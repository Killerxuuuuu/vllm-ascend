# SPDX-License-Identifier: Apache-2.0
"""A3 TQBMM host-overhead isolation. No profiler or production changes.

Diagnostic B/C bypass safety checks ONLY after strict fixed-input validation.
Their output reuse is single-stream, serial and not a production API.
--help and --dry-run are CPU-only; use the working vLLM Python for real runs.
"""

import argparse
import csv
import io
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent
CASES = ("A_full", "B_unchecked_alloc", "C_unchecked_reuse", "D_graph_full")
TIMING_FIELDS = ("host_submit_us", "stream_us", "wall_us")
GRAPH_WARMUP_REPLAYS = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 64, 512])
    parser.add_argument("--block-shape", type=int, nargs=3, default=[64, 64, 128], metavar=("M", "N", "K"))
    parser.add_argument("--num-stages", type=int, choices=(1, 2), default=2)
    parser.add_argument("--transpose-weight-tile", action="store_true")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--graph-batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(*args.tokens, args.tp_size, args.iterations, args.repeats, args.graph_batch_size) <= 0:
        parser.error("tokens/TP/iterations/repeats/graph-batch-size must be positive")
    if args.warmup < 0 or args.layer < 0 or not 0 <= args.tp_rank < args.tp_size:
        parser.error("warmup/layer must be nonnegative; tp-rank must be in [0,tp-size)")
    if any(size < 16 or size & (size - 1) for size in args.block_shape) or args.block_shape[2] < 32:
        parser.error("block sizes must be powers of two >=16, K>=32")
    args.tokens = list(dict.fromkeys(args.tokens))
    return args


def launch_plan(iterations, batch_size):
    return math.ceil(iterations / batch_size), math.ceil(iterations / batch_size) * batch_size


def launch_unchecked(torch_module, triton_module, kernel, x, weight, sx, sw, args, group_size_m, output=None):
    """Mirror the current host launch, omitting validation only (and optionally allocation).

    Shapes, strides, grid and constexpr arguments are rebuilt on EVERY call,
    just like the real wrapper. No compiled-handle or metadata caching here.
    The existing JIT kernel is called directly, never copied or modified.
    """
    tokens, groups, inner = x.shape
    columns = weight.shape[-1]
    block_m, block_n, block_k = args.block_shape
    if output is None:
        output = torch_module.empty((tokens, groups, columns), dtype=torch_module.bfloat16, device=x.device)
    grid = (triton_module.cdiv(tokens, block_m) * triton_module.cdiv(columns, block_n), groups)
    kernel[grid](
        x,
        weight,
        sx,
        sw,
        output,
        tokens,
        groups,
        inner,
        columns,
        *x.stride(),
        *weight.stride(),
        *sx.stride(),
        *sw.stride(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=group_size_m,
        TRANSPOSE_WEIGHT_TILE=args.transpose_weight_tile,
        num_stages=args.num_stages,
    )
    return output


def measure(function, npu, iterations, calls_per_invocation=1):
    """Measure separately: host enqueue loop, stream span, synchronized wall time.

    Stream events include launch starvation; they are NOT kernel-only time.
    Host and device execution overlap, so never add/subtract these as an
    exact decomposition. No per-iteration synchronization or profiler.
    """
    invocations, actual_calls = launch_plan(iterations, calls_per_invocation)
    npu.synchronize()
    start, end = npu.Event(enable_timing=True), npu.Event(enable_timing=True)
    start.record()
    before = time.perf_counter()
    for _ in range(invocations):
        function()
    submitted = time.perf_counter()
    end.record()
    end.synchronize()
    finished = time.perf_counter()
    return {
        "host_submit_us": (submitted - before) * 1e6 / actual_calls,
        "stream_us": start.elapsed_time(end) * 1e3 / actual_calls,
        "wall_us": (finished - before) * 1e6 / actual_calls,
        "actual_calls": actual_calls,
        "invocations": invocations,
    }


def summarize(samples):
    return {
        name: {metric: median(sample[metric] for sample in measurements) for metric in TIMING_FIELDS}
        for name, measurements in samples.items()
        if measurements
    }


def case_order(repeat):
    # Deterministic rotation reduces fixed case-order bias across repeats.
    offset = repeat % len(CASES)
    return CASES[offset:] + CASES[:offset]


def save_report(output, report):
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.chmod(0o644)
    temporary.replace(output / "result.json")
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=["T", "case", "repeat", "actual_calls", "invocations", *TIMING_FIELDS])
    writer.writeheader()
    for result in report["results"]:
        for name, samples in result["samples"].items():
            for repeat, sample in enumerate(samples):
                writer.writerow({"T": result["shape_T_G_K_N"][0], "case": name, "repeat": repeat, **sample})
    temporary = output / "timings.csv.tmp"
    temporary.write_text(text.getvalue(), encoding="utf-8")
    temporary.chmod(0o644)
    temporary.replace(output / "timings.csv")


def run_device(args, report, output):
    # Lazy runtime imports keep CLI parsing and CPU control tests independent
    # of torch, vLLM, Triton and NPU initialization.
    import torch
    import torch_npu
    from vllm.triton_utils import triton

    from tools.dsv4_a3_wo.benchmark_tqbmm import load_wo_a, qbmm_reference
    from vllm_ascend.ops.triton.transpose_quant_batch_matmul import (
        DEFAULT_GROUP_SIZE_M,
        _transpose_quant_batch_matmul_kernel,
        transpose_quant_batch_matmul,
    )
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if not torch.npu.is_available() or get_ascend_device_type() != AscendDeviceType.A3:
        raise RuntimeError("Requires an A3 NPU and the working vLLM/Triton-Ascend environment")
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    report["software"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
    with torch.inference_mode():
        weight, sw = load_wo_a(args)
        groups, inner, columns = weight.shape
        for tokens in args.tokens:

            def quantize(token_count=tokens):
                activation = torch.randn((token_count * groups, inner), device=args.device, dtype=torch.bfloat16)
                values, scales = torch_npu.npu_dynamic_quant(activation, dst_type=torch.int8)
                return values.reshape(token_count, groups, inner), scales.reshape(token_count, groups).float()

            x, sx = quantize()
            preallocated = torch.empty((tokens, groups, columns), dtype=torch.bfloat16, device=args.device)

            def full(x=x, sx=sx):
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

            def unchecked_alloc(x=x, sx=sx):
                return launch_unchecked(
                    torch,
                    triton,
                    _transpose_quant_batch_matmul_kernel,
                    x,
                    weight,
                    sx,
                    sw,
                    args,
                    DEFAULT_GROUP_SIZE_M,
                )

            def unchecked_reuse(x=x, sx=sx, preallocated=preallocated):
                return launch_unchecked(
                    torch,
                    triton,
                    _transpose_quant_batch_matmul_kernel,
                    x,
                    weight,
                    sx,
                    sw,
                    args,
                    DEFAULT_GROUP_SIZE_M,
                    output=preallocated,
                )

            functions = dict(zip(CASES[:3], (full, unchecked_alloc, unchecked_reuse)))
            expected = qbmm_reference(x, weight, sx, sw, torch.bfloat16)
            # Validate with the public wrapper first. B/C are diagnostic only.
            for function in functions.values():
                preallocated.fill_(float("nan"))
                torch.testing.assert_close(function(), expected, rtol=0, atol=0, equal_nan=False)
                for _ in range(args.warmup):
                    function()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(args.graph_batch_size):
                    captured_output = full()
            for _ in range(GRAPH_WARMUP_REPLAYS):
                graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured_output, expected, rtol=0, atol=0, equal_nan=False)

            # Replay must consume current buffers, not a baked-in input.
            original_x, original_sx = x.clone(), sx.clone()
            changed_x, changed_sx = quantize()
            x.copy_(changed_x)
            sx.copy_(changed_sx)
            changed_expected = qbmm_reference(x, weight, sx, sw, torch.bfloat16)
            for function in functions.values():
                preallocated.fill_(float("nan"))
                torch.testing.assert_close(function(), changed_expected, rtol=0, atol=0, equal_nan=False)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured_output, changed_expected, rtol=0, atol=0, equal_nan=False)
            x.copy_(original_x)
            sx.copy_(original_sx)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured_output, expected, rtol=0, atol=0, equal_nan=False)
            functions[CASES[3]] = graph.replay

            result = {
                "shape_T_G_K_N": [tokens, groups, inner, columns],
                "weight_stride": list(weight.stride()),
                "strict_correctness": True,
                "changed_input_checks": "passed for A/B/C/D",
                "samples": {name: [] for name in CASES},
            }
            report["results"].append(result)
            print(f"T={tokens}: strict A/B/C/D and changed-input checks passed", flush=True)
            for repeat in range(args.repeats):
                for name in case_order(repeat):
                    sample = measure(
                        functions[name],
                        torch.npu,
                        args.iterations,
                        args.graph_batch_size if name == CASES[3] else 1,
                    )
                    result["samples"][name].append(sample)
                result["median_us"] = summarize(result["samples"])
                save_report(output, report)
            print(json.dumps({"T": tokens, "median_us": result["median_us"]}, indent=2), flush=True)


def main(argv=None):
    args = parse_args(argv)
    settings = {
        **vars(args),
        "quant_model": str(args.quant_model.resolve()),
        "output": str(args.output) if args.output else None,
    }
    report = {
        "status": "running",
        "settings": settings,
        "results": [],
        "cases": {
            "A_full": "Unmodified public wrapper: checks + allocation + JIT launch",
            "B_unchecked_alloc": "Diagnostic: no checks; same shapes/strides/grid assembly + allocation + JIT launch",
            "C_unchecked_reuse": "Diagnostic: same as B but reuses one serial output buffer",
            "D_graph_full": "Graph replay of A; not of the unchecked diagnostic paths",
        },
        "note": "Prequantized fixed-shape diagnostic; not whole-model speed or original-model accuracy. "
        "Stream time includes submission gaps. Host and stream times overlap. "
        "A-B also includes wrapper implementation differences, not exact check-only cost. "
        "Graph is amortized over calls per replay; no profiler is enabled.",
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    if not (args.quant_model / "config.json").is_file():
        raise ValueError("quant-model must be an existing exported checkpoint")
    output = (args.output or REPO_ROOT.parent / f"tqbmm_eager_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o755)
    report["settings"]["output"] = str(output)
    save_report(output, report)
    print(f"Output: {output}", flush=True)
    try:
        run_device(args, report, output)
        report["status"] = "completed"
        return 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        return 130
    except Exception as error:
        report.update(status="failed", error=str(error))
        log = output / "error.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        log.chmod(0o644)
        print(f"Failed: {error}; see {log}", file=sys.stderr, flush=True)
        return 1
    finally:
        save_report(output, report)
        print(f"Status: {report['status']}; retained result.json and timings.csv", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
