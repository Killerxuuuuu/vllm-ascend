# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile the unchanged top-k kernel; do not infer bottlenecks from time alone."""

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_K = 512
DEFAULT_N = 192
DEFAULT_WIDTH = 1024


@dataclass(frozen=True)
class Case:
    name: str
    n: int
    k: int
    width: int
    valid: int
    kind: str = "topk"


def make_cases(suite):
    baseline = Case("baseline", DEFAULT_N, DEFAULT_K, DEFAULT_WIDTH, DEFAULT_N)
    if suite == "baseline":
        return [baseline]
    cases = [baseline]
    cases += [Case(f"k_{k}", 192, k, 1024, 192) for k in (1, 8, 32, 128)]
    cases += [Case(f"n_{n}", n, 512, 1024, n) for n in (64, 512, 1024)]
    cases += [Case(f"width_{w}", 192, 512, w, 192) for w in (256, 512)]
    cases += [Case("masked_192", 1024, 512, 1024, 192)]
    cases += [Case("masked_zero", 1024, 512, 1024, 0)]
    cases += [Case("io_control", 192, 512, 1024, 192, "io")]
    return cases


def validate_case(case):
    if not 0 < case.n <= case.width:
        raise ValueError("single-chunk probe requires 0 < n <= width")
    if case.width & (case.width - 1):
        raise ValueError("width must be a power of two")
    if not 1 <= case.k <= DEFAULT_K or not 0 <= case.valid <= case.n:
        raise ValueError("invalid k or valid count")
    if case.kind not in ("topk", "io"):
        raise ValueError("unknown probe kind")


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def extract_durations(folder, kernel_name):
    """Use one task CSV only: op_summary and kernel_details overlap."""
    files = sorted(folder.rglob("kernel_details.csv"))
    if not files:
        files = sorted(folder.rglob("op_summary*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one task CSV, found {len(files)} in {folder}")
    with files[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matched = [r for r in rows if r.get("Name", r.get("Op Name")) == kernel_name]
    durations = [float(r.get("Duration(us)", r.get("Task Duration(us)"))) for r in matched]
    if not durations or any(not (0 < d < float("inf")) for d in durations):
        raise RuntimeError(f"No valid durations for {kernel_name} in {files[0]}")
    return files[0], durations


def profiler_config(profiler, metric):
    level = profiler.ProfilerLevel.Level0 if metric == "none" else profiler.ProfilerLevel.Level1
    metric_name = "AiCoreNone" if metric == "none" else metric
    if not hasattr(profiler.AiCMetrics, metric_name):
        raise RuntimeError(f"Installed profiler has no metric {metric_name}; use --metrics none for timing only")
    return profiler._ExperimentalConfig(
        profiler_level=level,
        aic_metrics=getattr(profiler.AiCMetrics, metric_name),
        export_type=profiler.ExportType.Text,
    )


def prepare_case(case, torch, kernel, io_kernel, device):
    # CPU RNG, unique exactly-representable scores: no device RNG or tie ambiguity.
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    cpu_values = torch.randperm(case.n, generator=generator).to(torch.float32)
    values = cpu_values.to(device)
    counts = torch.tensor([case.valid], dtype=torch.int32, device=device)
    out_values = torch.empty((1, case.k), dtype=torch.float32, device=device)
    out_indices = torch.empty((1, case.k), dtype=torch.int32, device=device)
    scratch = torch.empty_like(values)

    def launch():
        if case.kind == "topk":
            kernel[(1, 1)](
                values,
                values,
                counts,
                out_values,
                out_indices,
                case.n,
                case.k,
                case.k,
                HAS_INPUT_INDICES=False,
                HAS_VALID_COUNTS=True,
                CHUNK_SIZE=case.width,
            )
        else:
            io_kernel[(1,)](
                values,
                scratch,
                out_values,
                out_indices,
                N=case.n,
                K=case.k,
                WIDTH=case.width,
                OUT_WIDTH=1 << (case.k - 1).bit_length(),
            )

    def check():
        if case.kind == "io":
            torch.testing.assert_close(scratch.cpu(), cpu_values, rtol=0, atol=0)
            count = min(case.n, case.k)
            selected = torch.arange(count, dtype=torch.int64)
        else:
            count = min(case.valid, case.k)
            selected = torch.argsort(cpu_values[: case.valid], descending=True)[:count]
        expected_values = torch.full((case.k,), -float("inf"))
        expected_indices = torch.full((case.k,), -1, dtype=torch.int32)
        expected_values[:count] = cpu_values[selected]
        expected_indices[:count] = selected.to(torch.int32)
        torch.testing.assert_close(out_values.cpu()[0], expected_values, rtol=0, atol=0)
        torch.testing.assert_close(out_indices.cpu()[0], expected_indices, rtol=0, atol=0)

    return launch, check


def run_case(case, args, folder, torch, profiler, kernel, io_kernel):
    validate_case(case)
    launch, check = prepare_case(case, torch, kernel, io_kernel, args.device)
    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    check()
    # Compilation, allocation, H2D, correctness checks: all outside the capture.
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
        on_trace_ready=profiler.tensorboard_trace_handler(str(folder), analyse_flag=True),
        experimental_config=profiler_config(profiler, args.metrics),
    ):
        for _ in range(args.repeats):
            launch()
        torch.npu.synchronize()
    check()
    name = "_indexer_topk_chunk_kernel" if case.kind == "topk" else "_topk_io_control_kernel"
    task_file, durations = extract_durations(folder, name)
    if len(durations) != args.repeats:
        raise RuntimeError(f"Expected {args.repeats} calls, captured {len(durations)} for {case.name}")
    return {
        **asdict(case),
        "calls": len(durations),
        "median_us": statistics.median(durations),
        "p10_us": percentile(durations, 0.1),
        "p90_us": percentile(durations, 0.9),
        "min_us": min(durations),
        "max_us": max(durations),
        "task_csv": str(task_file),
        "op_summary_files": [str(p) for p in sorted(folder.rglob("op_summary*.csv"))],
        "durations_us": durations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("topk_diagnostics"))
    parser.add_argument("--suite", choices=("all", "baseline"), default="all")
    parser.add_argument(
        "--metrics", choices=("none", "PipeUtilization", "MemoryAccess", "MemoryUB"), default="PipeUtilization"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="npu:0", help="logical device; select physical card before launching")
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.repeats < 3:
        parser.error("warmup must be >= 1, repeats must be >= 3")
    if not args.device.startswith("npu:"):
        parser.error("device must be npu:<logical index>")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="topk-", dir=args.output_root)).resolve()
    report = {
        "status": "running",
        "metrics_requested": args.metrics,
        "arguments": {**vars(args), "output_root": str(args.output_root)},
        "results": [],
        "errors": [],
        "limitations": [
            "Synthetic single-row single-chunk probe, not a full-service benchmark.",
            "Profiler kernel duration excludes host gaps but profiling can perturb execution.",
            "Repeated buffers may be cache-resident; this is not a cold-HBM bandwidth test.",
            "K also changes output stores; width changes may change compilation and occupancy.",
            "IO control has different access patterns and extra scratch writes; "
            "no time subtraction or compute percentage is valid.",
            "Requested hardware metrics are not guaranteed available on this hardware/toolchain; "
            "inspect raw CSV fields.",
        ],
    }
    print(f"Output: {output}", flush=True)
    try:
        # Lazy imports keep --help and CPU tests independent of vLLM/NPU.
        import torch
        import torch_npu
        from topk_diagnostic_kernels import _topk_io_control_kernel

        from vllm_ascend.ops.triton.indexer_block_topk import _indexer_topk_chunk_kernel
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

        torch.npu.set_device(args.device)
        init_device_properties_triton()
        source = Path(inspect.getfile(_indexer_topk_chunk_kernel.fn)).resolve()
        report["source"] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        report["device_name"] = torch.npu.get_device_name(torch.npu.current_device())
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
        for package in ("triton-ascend", "vllm", "vllm-ascend"):
            try:
                report["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["versions"][package] = "unknown"
        with torch.inference_mode():
            for case in make_cases(args.suite):
                print(
                    f"Running {case.name}: n={case.n}, k={case.k}, width={case.width}, valid={case.valid}", flush=True
                )
                result = run_case(
                    case,
                    args,
                    output / case.name,
                    torch,
                    torch_npu.profiler,
                    _indexer_topk_chunk_kernel,
                    _topk_io_control_kernel,
                )
                report["results"].append(result)
                (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"  median={result['median_us']:.3f} us", flush=True)
        report["status"] = "durations_collected_metrics_need_review"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if report["results"]:
            fields = ["name", "kind", "n", "k", "width", "valid", "calls", "median_us", "p10_us", "p90_us"]
            with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(report["results"])
        print(f"Report: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
