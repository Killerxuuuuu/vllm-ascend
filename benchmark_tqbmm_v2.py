# SPDX-License-Identifier: Apache-2.0
"""Isolated A3 V2 experiment. Parent/--dry-run need only the Python stdlib."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

from tools.dsv4_a3_wo.runtime_options import internal_format_enabled
from tools.dsv4_a3_wo.tune_tqbmm import run_worker, write_json

ROOT = Path(__file__).resolve().parent


def v1_options():
    return {"block_shape": (128, 64, 128), "num_stages": 2, "reuse_m": 1, "reuse_n": 2}


def configure_internal_format(torch_npu):
    # On torch_npu 2.10 the Python config is a setter, not a getter.
    torch_npu.npu.config.allow_internal_format = True
    enabled = internal_format_enabled(torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT"))
    if not enabled:
        raise RuntimeError("V2 benchmark requires ALLOW_INTERNAL_FORMAT=enable, but the runtime reports it disabled")
    return enabled


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory, never overwrite an existing run")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 8192])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--block-shape", type=int, nargs=3, default=[64, 64, 64], metavar=("M", "N", "DOT_K"))
    parser.add_argument("--panel-k", type=int, default=128)
    parser.add_argument("--num-stages", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--programs", type=int, default=0, help="0 queries num_aicore; otherwise explicit persistent programs"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--graph-batch-size", type=int, default=4)
    parser.add_argument("--timing-mode", choices=["eager", "graph", "both"], default="both")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per fresh T/seed process, seconds")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if any(value <= 0 for value in (*args.tokens, args.iterations, args.repeats, args.graph_batch_size, args.timeout)):
        parser.error("tokens/iterations/repeats/graph-batch-size/timeout must be positive")
    if (
        args.warmup < 1
        or args.programs < 0
        or args.layer < 0
        or args.tp_size < 1
        or not 0 <= args.tp_rank < args.tp_size
    ):
        parser.error("Require warmup>=1, programs>=0, layer>=0 and 0<=tp-rank<tp-size")
    args.tokens = list(dict.fromkeys(args.tokens))
    args.seeds = list(dict.fromkeys(args.seeds))
    if args.worker and (len(args.tokens) != 1 or len(args.seeds) != 1):
        parser.error("worker takes exactly one T and seed")
    return args


def worker_command(args, tokens, seed):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--quant-model",
        str(args.quant_model.resolve()),
        "--output",
        str(args.output.resolve()),
        "--tokens",
        str(tokens),
        "--seeds",
        str(seed),
        "--device",
        args.device,
        "--layer",
        str(args.layer),
        "--tp-size",
        str(args.tp_size),
        "--tp-rank",
        str(args.tp_rank),
        "--block-shape",
        *map(str, args.block_shape),
        "--panel-k",
        str(args.panel_k),
        "--num-stages",
        str(args.num_stages),
        "--programs",
        str(args.programs),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--repeats",
        str(args.repeats),
        "--graph-batch-size",
        str(args.graph_batch_size),
        "--timing-mode",
        args.timing_mode,
    ]
    if args.check_only:
        command.append("--check-only")
    return command


def measure(functions, args, timer):
    """Rotate order to avoid always timing the same implementation first."""
    samples = {name: [] for name in functions}
    items = list(functions.items())
    for repeat in range(args.repeats):
        rotated = items[repeat % len(items) :] + items[: repeat % len(items)]
        for name, function in rotated:
            duration = timer(
                function,
                args.warmup,
                args.iterations,
                timing_mode=args.current_mode,
                graph_batch_size=args.graph_batch_size,
            )
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid duration for {name}: {duration}")
            samples[name].append(duration)
    return {"median_ms": {name: median(values) for name, values in samples.items()}, "samples_ms": samples}


def source_hashes():
    names = (
        "benchmark_tqbmm_v2.py",
        "vllm_ascend/ops/triton/transpose_quant_batch_matmul_v2.py",
        "vllm_ascend/ops/triton/transpose_quant_batch_matmul.py",
        "tools/dsv4_a3_wo/benchmark_tqbmm.py",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def worker(args):
    # Worker isolation: no torch/vLLM imports in the orchestration process.
    import torch
    import torch_npu

    from tools.dsv4_a3_wo.benchmark_tqbmm import (
        dynamic_qbmm_reference,
        elapsed_ms,
        load_wo_a,
        qbmm_reference,
    )
    from tools.dsv4_a3_wo.qbmm_probe import error_metrics
    from vllm_ascend.ops.triton.transpose_quant_batch_matmul import PreparedDynamicWoA, PreparedTQBMM
    from vllm_ascend.ops.triton.transpose_quant_batch_matmul_v2 import (
        PreparedDynamicWoAV2,
        PreparedTQBMMV2,
        V2Config,
    )
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if not torch.npu.is_available() or get_ascend_device_type() != AscendDeviceType.A3:
        raise RuntimeError("V2 experiment requires an A3 NPU with Triton-Ascend")
    hooks = [
        value
        for value in os.environ.get("LD_PRELOAD", "").replace(":", " ").split()
        if any(key in value.lower() for key in ("msmemscope", "leaks_ascend", "ascend_kernel_hook"))
    ]
    modules = [name for name in sys.modules if "msmemscope" in name.lower()]
    if hooks or modules:
        raise RuntimeError(f"Disable memory instrumentation first: {hooks}, {modules}")
    torch.npu.set_device(args.device)
    allow_internal_format = configure_internal_format(torch_npu)
    torch.manual_seed(args.seeds[0])
    config = V2Config(tuple(args.block_shape), args.panel_k, True, args.programs, args.num_stages)
    control = replace(config, prefetch=False)
    with torch.inference_mode():
        weight, sw = load_wo_a(args)
        groups, inner, columns = weight.shape
        tokens = args.tokens[0]
        activation = torch.randn(tokens, groups, inner, device=args.device, dtype=torch.bfloat16)
        q, sx = torch_npu.npu_dynamic_quant(activation.view(tokens * groups, inner), dst_type=torch.int8)
        q, sx = q.view(tokens, groups, inner), sx.view(tokens, groups)
        prequantized = {
            "composition": lambda: qbmm_reference(q, weight, sx, sw, activation.dtype),
            "v1": PreparedTQBMM(q, weight, sx, sw, output_dtype=activation.dtype, **v1_options()),
            "v2_no_prefetch": PreparedTQBMMV2(q, weight, sx, sw, config=control),
            "v2": PreparedTQBMMV2(q, weight, sx, sw, config=config),
        }
        dynamic_plans = {
            "v1": PreparedDynamicWoA(weight, sw, **v1_options()),
            "v2_no_prefetch": PreparedDynamicWoAV2(weight, sw, config=control),
            "v2": PreparedDynamicWoAV2(weight, sw, config=config),
        }
        dynamic = {"composition": lambda: dynamic_qbmm_reference(activation, weight, sw)}
        dynamic.update({name: lambda plan=plan: plan(activation) for name, plan in dynamic_plans.items()})
        correctness = {}
        for domain, functions in (("prequantized", prequantized), ("including_quant", dynamic)):
            expected = functions["composition"]()
            for name, function in functions.items():
                actual = function()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=False)
                correctness[f"{domain}/{name}"] = error_metrics(actual, expected)
                print(f"Strict check passed: {domain}/{name}", flush=True)
        # Independent selected INT64 CPU dots: not just agreement with CANN.
        q_cpu, w_cpu, sx_cpu, sw_cpu = q.cpu(), weight.cpu(), sx.cpu(), sw.cpu()
        sample_rows = sorted({0, tokens // 2, tokens - 1})
        sample_cols = sorted({0, columns // 2, columns - 1})
        for name, function in prequantized.items():
            actual_cpu = function().cpu()
            for group in range(groups):
                for row in sample_rows:
                    for col in sample_cols:
                        integer = (q_cpu[row, group].long() * w_cpu[group, :, col].long()).sum()
                        golden = ((integer.float() * sw_cpu[group, col]) * sx_cpu[row, group]).to(activation.dtype)
                        torch.testing.assert_close(actual_cpu[row, group, col], golden, rtol=0, atol=0)
        # Dynamic bindings must read changed data, not cached activation ptrs.
        for name, function in dynamic.items():
            function()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = function()
            activation.mul_(0.5)
            try:
                graph.replay()
                torch.npu.synchronize()
                expected = dynamic_qbmm_reference(activation, weight, sw)
                torch.testing.assert_close(captured, expected, rtol=0, atol=0, equal_nan=False)
            finally:
                activation.mul_(2)
                torch.npu.synchronize()
            del captured, graph
            print(f"Changed-input graph check passed: {name}", flush=True)
        timings = {}
        if not args.check_only:
            modes = ("eager", "graph") if args.timing_mode == "both" else (args.timing_mode,)
            for mode in modes:
                args.current_mode = mode
                timings[mode] = {
                    domain: measure(functions, args, elapsed_ms)
                    for domain, functions in (("prequantized", prequantized), ("including_quant", dynamic))
                }
        return {
            "v2_experiment": "passed",
            "T": tokens,
            "seed": args.seeds[0],
            "shape_T_G_K_N": [tokens, groups, inner, columns],
            "weight_stride": list(weight.stride()),
            "v1": v1_options(),
            "v2": asdict(config),
            "v2_prefetch_scheme": (
                "paired full panels"
                if tokens % config.block_shape[0] == columns % config.block_shape[1] == inner % config.panel_k == 0
                else "paired direct dot-width loads"
            ),
            "persistent_cores": prequantized["v2"]._cores,
            "logical_accumulator_bytes": config.block_shape[0] * config.block_shape[1] * 4,
            "memory_placement": "compiler-managed; panels/prefetch do not guarantee physical L1/overlap",
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "allow_internal_format": allow_internal_format,
            "storage_formats": {"weight": torch_npu.get_npu_format(weight), "q": torch_npu.get_npu_format(q)},
            "correctness": correctness,
            "cpu_sample_check": "passed",
            "changed_input_graph": "passed",
            "timings": timings,
            "source_sha256": source_hashes(),
        }


def timing_rows(report):
    for mode, domains in report.get("timings", {}).items():
        for domain, measurements in domains.items():
            durations = measurements["median_ms"]
            for name, duration in durations.items():
                yield {
                    "T": report["T"],
                    "seed": report["seed"],
                    "mode": mode,
                    "domain": domain,
                    "implementation": name,
                    "median_ms": duration,
                    "speedup_vs_composition": durations["composition"] / duration,
                    "speedup_vs_v1": durations["v1"] / duration,
                }


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        print(json.dumps(worker(args), indent=2, allow_nan=False))
        return
    commands = [worker_command(args, tokens, seed) for tokens in args.tokens for seed in args.seeds]
    if args.dry_run:
        print(json.dumps({"commands": commands, "source_sha256": source_hashes()}, indent=2))
        return
    if not args.quant_model.is_dir():
        raise SystemExit("--quant-model must be an existing quantized checkpoint directory")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    payload = {"run_status": "running", "commands": commands, "source_sha256": source_hashes(), "reports": []}
    write_json(args.output / "results.json", payload)
    try:
        for index, command in enumerate(commands):
            print(f"[{index + 1}/{len(commands)}] {command}", flush=True)
            report = run_worker(command, args.output / "logs" / f"case-{index:03}.log", args.timeout, "v2_experiment")
            payload["reports"].append(report)
            write_json(args.output / "results.json", payload)
        rows = [row for report in payload["reports"] for row in timing_rows(report)]
        if rows:
            with (args.output / "timings.csv").open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        payload["run_status"] = "completed"
    except BaseException as error:
        payload.update(run_status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "results.json", payload)
        print(args.output / "results.json", flush=True)


if __name__ == "__main__":
    main()
