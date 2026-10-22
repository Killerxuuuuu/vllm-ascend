# SPDX-License-Identifier: Apache-2.0
"""Collect A3 TQBMM/CANN Eager and Graph profiles, without changing defaults.

Run in the vLLM NPU environment. --help/--dry-run need only Python's stdlib.
Each case runs in a fresh, serial subprocess. Capture/compilation/activation
quantization are outside the collection interval. Profiler times do not replace
the unprofiled benchmark. Only readable analysis copies and logs are archived.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROFILE_CASES = ("triton_eager", "triton_graph", "qbmm_eager", "qbmm_graph")
ANALYSIS_PATTERNS = ("kernel_details.csv", "op_summary*.csv", "trace_view.json")
DEFAULT_GRAPH_BATCH_SIZE = 16
DEFAULT_REPLAYS = 4
GRAPH_WARMUP_REPLAYS = 3
READABLE_DIRECTORY_MODE = 0o755
READABLE_FILE_MODE = 0o644


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-shape", type=int, nargs=3, default=[64, 64, 128], metavar=("M", "N", "K"))
    parser.add_argument("--num-stages", type=int, choices=(1, 2), default=2)
    parser.add_argument("--transpose-weight-tile", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--graph-batch-size", type=int, default=DEFAULT_GRAPH_BATCH_SIZE)
    parser.add_argument("--replays", type=int, default=DEFAULT_REPLAYS)
    parser.add_argument("--timeout", type=int, default=600, help="Per-case seconds, including analysis")
    parser.add_argument("--cases", nargs="+", choices=PROFILE_CASES, default=list(PROFILE_CASES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-case", choices=PROFILE_CASES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.tokens, args.tp_size, args.graph_batch_size, args.replays, args.timeout) <= 0 or args.warmup < 0:
        parser.error("tokens/TP/batch/replays/timeout must be positive; warmup must be nonnegative")
    if args.layer < 0 or not 0 <= args.tp_rank < args.tp_size:
        parser.error("layer must be nonnegative and tp-rank must be in [0,tp-size)")
    if any(size < 16 or size & (size - 1) for size in args.block_shape) or args.block_shape[2] < 32:
        parser.error("block-shape must contain powers of two >=16, with K>=32")
    if args.worker_case and args.output is None:
        parser.error("worker requires an output directory")
    args.cases = list(dict.fromkeys(args.cases))
    return args


def case_plan(args):
    calls = args.graph_batch_size * args.replays
    return [
        {
            "case": name,
            "implementation": name.split("_")[0],
            "mode": name.split("_")[1],
            "operator_calls": calls,
            "graph_replays": args.replays if name.endswith("graph") else 0,
            "status": "pending",
        }
        for name in args.cases
    ]


def worker_command(args, name, output):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker-case",
        name,
        "--quant-model",
        str(args.quant_model.resolve()),
        "--output",
        str(output.resolve()),
        "--device",
        args.device,
        "--tokens",
        str(args.tokens),
        "--layer",
        str(args.layer),
        "--tp-size",
        str(args.tp_size),
        "--tp-rank",
        str(args.tp_rank),
        "--seed",
        str(args.seed),
        "--block-shape",
        *map(str, args.block_shape),
        "--num-stages",
        str(args.num_stages),
        "--warmup",
        str(args.warmup),
        "--graph-batch-size",
        str(args.graph_batch_size),
        "--replays",
        str(args.replays),
    ]
    return command + (["--transpose-weight-tile"] if args.transpose_weight_tile else [])


def run_profile_calls(function, graph, batch_size, replays):
    if graph is None:
        for _ in range(batch_size * replays):
            function()
    else:
        for _ in range(replays):
            graph.replay()


def capture_case(args):
    # Runtime-only imports isolate NPU initialization to the worker process;
    # parent orchestration, --help, and --dry-run must remain CPU-only.
    import torch
    import torch_npu

    from tools.dsv4_a3_wo.benchmark_tqbmm import load_wo_a, qbmm_reference
    from vllm_ascend.ops.triton.transpose_quant_batch_matmul import transpose_quant_batch_matmul
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if not torch.npu.is_available() or get_ascend_device_type() != AscendDeviceType.A3:
        raise RuntimeError("This profiling script requires an Ascend A3 NPU")
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    case_dir = args.output / "raw" / args.worker_case
    case_dir.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        weight, sw = load_wo_a(args)
        groups, inner, columns = weight.shape
        activations = torch.randn((args.tokens, groups, inner), dtype=torch.bfloat16, device=args.device)
        x, sx = torch_npu.npu_dynamic_quant(activations.reshape(-1, inner), dst_type=torch.int8)
        x = x.reshape(args.tokens, groups, inner)
        sx = sx.reshape(args.tokens, groups).float()

        def fused():
            return transpose_quant_batch_matmul(
                x,
                weight,
                sx,
                sw,
                output_dtype=torch.bfloat16,
                block_shape=tuple(args.block_shape),
                num_stages=args.num_stages,
                transpose_weight_tile=args.transpose_weight_tile,
            )

        def reference():
            return qbmm_reference(x, weight, sx, sw, torch.bfloat16)

        expected = reference()
        torch.testing.assert_close(fused(), expected, rtol=0, atol=0, equal_nan=False)
        function = fused if args.worker_case.startswith("triton") else reference
        for _ in range(args.warmup):
            function()
        torch.npu.synchronize()
        graph = None
        if args.worker_case.endswith("graph"):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(args.graph_batch_size):
                    captured = function()
            for _ in range(GRAPH_WARMUP_REPLAYS):
                graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0, equal_nan=False)
            torch.npu.synchronize()

        print(f"Capture: {args.worker_case}; calls={args.graph_batch_size * args.replays}", flush=True)
        with (
            torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                    data_simplification=False,
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(case_dir / "profiler"), analyse_flag=True
                ),
            ),
            torch.profiler.record_function(args.worker_case),
        ):
            run_profile_calls(function, graph, args.graph_batch_size, args.replays)
            torch.npu.synchronize()

        metadata = {
            "case": args.worker_case,
            "strict_correctness": True,
            "shape_T_G_K_N": [args.tokens, groups, inner, columns],
            "weight_stride": list(weight.stride()),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "operator_calls": args.graph_batch_size * args.replays,
            "input_scope": "prequantized; dynamic quantization and graph capture are outside profiling",
            "graph_output_checked": graph is not None,
        }
        write_manifest(case_dir / "capture.json", metadata)
    return 0


def write_manifest(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.chmod(READABLE_FILE_MODE)
    temporary.replace(path)


def export_case(case_dir, export_dir, name):
    """Copy only nonempty analysis files; never change raw-tree permissions."""
    artifacts = []
    for pattern in ANALYSIS_PATTERNS:
        sources = sorted(case_dir.rglob(pattern))
        if not sources:
            raise RuntimeError(f"{name}: missing {pattern}; keep the raw profile and case log")
        for source in sources:
            if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(f"{name}: invalid/empty analysis file: {source}")
            source.resolve().relative_to(case_dir.resolve())
            destination = export_dir / f"{name}__{source.name}"
            if destination.exists():
                destination = export_dir / f"{name}__{len(artifacts)}__{source.name}"
            if destination.exists():
                raise RuntimeError(f"Refusing to overwrite export: {destination}")
            shutil.copyfile(source, destination)
            destination.chmod(READABLE_FILE_MODE)
            artifacts.append({"file": destination.name, "bytes": destination.stat().st_size})
    return artifacts


def announce(output, message):
    print(message, flush=True)
    with (output / "collect.log").open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def execute_case(args, output, case):
    name = case["case"]
    log = output / f"{name}.log"
    command = worker_command(args, name, output)
    case.update(status="running", command=command, log=log.name)
    started = time.perf_counter()
    try:
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
        if process.returncode:
            raise RuntimeError(f"{name}: worker exit={process.returncode}; see {log}")
        case_dir = output / "raw" / name
        metadata = json.loads((case_dir / "capture.json").read_text(encoding="utf-8"))
        if metadata.get("case") != name or metadata.get("strict_correctness") is not True:
            raise ValueError(f"{name}: missing strict correctness confirmation")
        if metadata.get("operator_calls") != case["operator_calls"]:
            raise ValueError(f"{name}: unexpected operator call count")
        case["capture"] = metadata
        case["artifacts"] = export_case(case_dir, output / "export", name)
        case["status"] = "completed"
    except KeyboardInterrupt:
        case["status"] = "interrupted"
        raise
    except Exception as error:
        case.update(status="failed", error=str(error))
        raise
    finally:
        case["wall_seconds"] = time.perf_counter() - started
        if log.exists():
            log.chmod(READABLE_FILE_MODE)
            with (
                (output / "collect.log").open("a", encoding="utf-8") as collected,
                log.open(encoding="utf-8", errors="replace") as worker_log,
            ):
                shutil.copyfileobj(worker_log, collected)


def archive_exports(output):
    archive = output / "export.tar.gz"
    temporary = output / "export.tar.gz.tmp"
    with tarfile.open(temporary, "w:gz") as bundle:
        for path in sorted((output / "export").iterdir()):
            bundle.add(path, arcname=path.name, recursive=False)
        for path in (output / "manifest.json", output / "collect.log"):
            bundle.add(path, arcname=path.name, recursive=False)
    temporary.chmod(READABLE_FILE_MODE)
    temporary.replace(archive)
    return archive


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "parameters": {"block_shape": args.block_shape, "num_stages": args.num_stages},
                    "cases": case_plan(args),
                },
                indent=2,
            )
        )
        return 0
    if args.worker_case:
        return capture_case(args)
    if not (args.quant_model / "config.json").is_file():
        raise ValueError("quant-model must point to an existing exported model")
    output = args.output or REPO_ROOT.parent / f"tqbmm_profile_best_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(READABLE_DIRECTORY_MODE)
    (output / "export").mkdir(mode=READABLE_DIRECTORY_MODE)
    (output / "export").chmod(READABLE_DIRECTORY_MODE)
    settings = {**vars(args), "quant_model": str(args.quant_model.resolve()), "output": str(output)}
    manifest = {
        "status": "running",
        "settings": settings,
        "cases": case_plan(args),
        "note": "Profiling is diagnostic, not steady-state benchmarking. Raw data is not in the archive.",
    }
    write_manifest(output / "manifest.json", manifest)
    announce(output, f"Output: {output}")
    try:
        for case in manifest["cases"]:
            announce(output, f"START {case['case']} ({case['operator_calls']} operator calls)")
            case["status"] = "running"
            write_manifest(output / "manifest.json", manifest)
            execute_case(args, output, case)
            write_manifest(output / "manifest.json", manifest)
            announce(output, f"DONE {case['case']}: {len(case['artifacts'])} analysis files")
        manifest["status"] = "completed"
        write_manifest(output / "manifest.json", manifest)
        (output / "collect.log").chmod(READABLE_FILE_MODE)
        archive = archive_exports(output)
        print(f"Archive: {archive} ({archive.stat().st_size} bytes)", flush=True)
        return 0
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        announce(output, "Interrupted; logs, manifest and available raw data are preserved. No complete archive.")
        return 130
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        announce(output, traceback.format_exc())
        announce(output, "FAILED: keep collect.log and raw; no complete archive is claimed.")
        return 1
    finally:
        write_manifest(output / "manifest.json", manifest)
        (output / "collect.log").chmod(READABLE_FILE_MODE)


if __name__ == "__main__":
    raise SystemExit(main())
