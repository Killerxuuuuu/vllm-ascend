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
from functools import partial
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent
CASES = ("A_full", "B_unchecked_alloc", "C_unchecked_reuse", "D_graph_full")
NATIVE_CASES = ("R_qbmm_eager", "R_qbmm_graph")
PREPARED_CASES = ("S_safe_out", "P_prepared_reuse", "P_graph_prepared")
VALIDATION_CASES = ("V_control", "V_input_metadata", "V_option_metadata", "V_format", "V_all")
ADDBACK_CHECKS = ("empty_call_control", "input_metadata", "option_metadata", "format_all_four", "all_validation")
GRAPH_CASES = ("D_graph_full", "R_qbmm_graph", "P_graph_prepared")
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
    parser.add_argument("--check-breakdown", action="store_true", help="Separately time validation/format queries")
    parser.add_argument("--check-iterations", type=int, default=2048)
    parser.add_argument("--include-qbmm", action="store_true", help="Include same-run native composition Eager/Graph")
    parser.add_argument("--prepared", action="store_true", help="Measure safe out and validated fixed-buffer APIs")
    parser.add_argument(
        "--validation-addback", action="store_true", help="Add checks back to the identical B launch path"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(*args.tokens, args.tp_size, args.iterations, args.repeats, args.graph_batch_size, args.check_iterations)
        <= 0
    ):
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


def case_order(repeat, cases=CASES):
    # Deterministic rotation reduces fixed case-order bias across repeats.
    offset = repeat % len(cases)
    return cases[offset:] + cases[:offset]


def check_format(get_format, tensor):
    if get_format(tensor) not in (0, 2):
        raise ValueError("Diagnostic inputs must use ordinary ND storage")


def validation_functions(torch_module, torch_npu, validate_inputs, validate_stages, x, weight, sx, sw, args):
    """Call the actual input validator, and mirror remaining option/format checks.

    These do not launch matmul, allocate output, or alter tensors. Query costs
    are measured rather than assuming get_npu_format causes synchronization.
    """

    def empty_call():
        pass

    def input_metadata():
        validate_inputs(x, weight, sx, sw, torch_module.bfloat16)

    def option_metadata():
        validate_stages(args.num_stages)
        if type(args.transpose_weight_tile) is not bool:
            raise TypeError("transpose_weight_tile must be bool")
        if x.device.type != "npu":
            raise ValueError("Requires NPU tensors")
        block_shape = tuple(args.block_shape)
        if (
            len(block_shape) != 3
            or any(not isinstance(size, int) or size < 16 or size & (size - 1) for size in block_shape)
            or block_shape[2] < 32
        ):
            raise ValueError("Invalid block shape")

    def format_all():
        # Same short-circuit expression and tensor order as the public wrapper.
        if any(torch_npu.get_npu_format(tensor) not in (0, 2) for tensor in (x, weight, sx, sw)):
            raise ValueError("Diagnostic inputs must use ordinary ND storage")

    def all_validation():
        input_metadata()
        option_metadata()
        format_all()

    return {
        "empty_call_control": empty_call,
        "input_metadata": input_metadata,
        "option_metadata": option_metadata,
        "format_all_four": format_all,
        "format_x": partial(check_format, torch_npu.get_npu_format, x),
        "format_weight": partial(check_format, torch_npu.get_npu_format, weight),
        "format_x_scale": partial(check_format, torch_npu.get_npu_format, sx),
        "format_weight_scale": partial(check_format, torch_npu.get_npu_format, sw),
        "all_validation": all_validation,
    }


def measure_host(function, npu, iterations):
    # No per-call timers, events or synchronization; include any blocking
    # inside the query itself, but exclude the boundary synchronization.
    npu.synchronize()
    before = time.perf_counter()
    for _ in range(iterations):
        function()
    elapsed = time.perf_counter() - before
    npu.synchronize()
    return elapsed * 1e6 / iterations


def check_then_launch(check, launch):
    # All V variants share this exact wrapper and the exact B allocation/JIT
    # path. Include a no-op control to expose the extra wrapper's own overhead.
    check()
    return launch()


def validation_addback_functions(launch, checks):
    return {
        name: partial(check_then_launch, checks[check_name], launch)
        for name, check_name in zip(VALIDATION_CASES, ADDBACK_CHECKS)
    }


def capture_graph(function, npu, batch_size):
    npu.synchronize()
    graph = npu.NPUGraph()
    with npu.graph(graph):
        for _ in range(batch_size):
            output = function()
    for _ in range(GRAPH_WARMUP_REPLAYS):
        graph.replay()
    npu.synchronize()
    return graph, output


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
    if report.get("settings", {}).get("check_breakdown"):
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=["T", "check", "repeat", "iterations", "host_call_us"])
        writer.writeheader()
        for result in report["results"]:
            for name, samples in result.get("check_samples_us", {}).items():
                for repeat, sample in enumerate(samples):
                    writer.writerow(
                        {
                            "T": result["shape_T_G_K_N"][0],
                            "check": name,
                            "repeat": repeat,
                            "iterations": report["settings"]["check_iterations"],
                            "host_call_us": sample,
                        }
                    )
        temporary = output / "check_timings.csv.tmp"
        temporary.write_text(text.getvalue(), encoding="utf-8")
        temporary.chmod(0o644)
        temporary.replace(output / "check_timings.csv")


def run_device(args, report, output):
    # Lazy runtime imports keep CLI parsing and CPU control tests independent
    # of torch, vLLM, Triton and NPU initialization.
    import torch
    import torch_npu
    from vllm.triton_utils import triton

    from tools.dsv4_a3_wo.benchmark_tqbmm import load_wo_a, qbmm_reference
    from vllm_ascend.ops.triton.transpose_quant_batch_matmul import (
        DEFAULT_GROUP_SIZE_M,
        PreparedTQBMM,
        _transpose_quant_batch_matmul_kernel,
        _validate_inputs,
        _validate_num_stages,
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

            def reference(x=x, sx=sx):
                return qbmm_reference(x, weight, sx, sw, torch.bfloat16)

            functions = dict(zip(CASES[:3], (full, unchecked_alloc, unchecked_reuse)))
            prepared = None
            if args.prepared:
                safe_output = torch.empty((tokens, groups, columns), dtype=torch.bfloat16, device=args.device)

                def safe_out(x=x, sx=sx, out=safe_output):
                    return transpose_quant_batch_matmul(
                        x,
                        weight,
                        sx,
                        sw,
                        output_dtype=torch.bfloat16,
                        block_shape=tuple(args.block_shape),
                        num_stages=args.num_stages,
                        transpose_weight_tile=args.transpose_weight_tile,
                        out=out,
                    )

                prepared = PreparedTQBMM(
                    x,
                    weight,
                    sx,
                    sw,
                    output_dtype=torch.bfloat16,
                    block_shape=tuple(args.block_shape),
                    num_stages=args.num_stages,
                    transpose_weight_tile=args.transpose_weight_tile,
                )
                prepared.output.fill_(float("nan"))
                functions[PREPARED_CASES[0]] = safe_out
                functions[PREPARED_CASES[1]] = prepared
            checks = {}
            if args.check_breakdown or args.validation_addback:
                checks = validation_functions(
                    torch,
                    torch_npu,
                    _validate_inputs,
                    _validate_num_stages,
                    x,
                    weight,
                    sx,
                    sw,
                    args,
                )
            if args.validation_addback:
                functions.update(validation_addback_functions(unchecked_alloc, checks))
            if args.include_qbmm:
                functions[NATIVE_CASES[0]] = reference
            expected = qbmm_reference(x, weight, sx, sw, torch.bfloat16)
            # Validate with the public wrapper first. B/C are diagnostic only.
            for function in functions.values():
                preallocated.fill_(float("nan"))
                torch.testing.assert_close(function(), expected, rtol=0, atol=0, equal_nan=False)
                for _ in range(args.warmup):
                    function()
            torch.npu.synchronize()
            graph_sources = {CASES[3]: full}
            if args.include_qbmm:
                graph_sources[NATIVE_CASES[1]] = reference
            if prepared is not None:
                graph_sources[PREPARED_CASES[2]] = prepared
            graphs = {}
            for name, function in graph_sources.items():
                graphs[name] = capture_graph(function, torch.npu, args.graph_batch_size)
                torch.testing.assert_close(graphs[name][1], expected, rtol=0, atol=0, equal_nan=False)

            # Replay must consume current buffers, not a baked-in input.
            original_x, original_sx = x.clone(), sx.clone()
            changed_x, changed_sx = quantize()
            x.copy_(changed_x)
            sx.copy_(changed_sx)
            changed_expected = qbmm_reference(x, weight, sx, sw, torch.bfloat16)
            for function in functions.values():
                preallocated.fill_(float("nan"))
                torch.testing.assert_close(function(), changed_expected, rtol=0, atol=0, equal_nan=False)
            for graph, captured_output in graphs.values():
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(captured_output, changed_expected, rtol=0, atol=0, equal_nan=False)
            x.copy_(original_x)
            sx.copy_(original_sx)
            for name, (graph, captured_output) in graphs.items():
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(captured_output, expected, rtol=0, atol=0, equal_nan=False)
                functions[name] = graph.replay

            names = (
                CASES
                + (NATIVE_CASES if args.include_qbmm else ())
                + (VALIDATION_CASES if args.validation_addback else ())
                + (PREPARED_CASES if args.prepared else ())
            )
            if args.check_breakdown:
                for function in checks.values():
                    for _ in range(max(1, args.warmup)):
                        function()

            result = {
                "shape_T_G_K_N": [tokens, groups, inner, columns],
                "weight_stride": list(weight.stride()),
                "strict_correctness": True,
                "changed_input_checks": "passed for all selected cases",
                "samples": {name: [] for name in names},
            }
            if args.check_breakdown:
                result["check_samples_us"] = {name: [] for name in checks}
            report["results"].append(result)
            print(f"T={tokens}: strict {','.join(names)} and changed-input checks passed", flush=True)
            for repeat in range(args.repeats):
                for name in case_order(repeat, names):
                    sample = measure(
                        functions[name],
                        torch.npu,
                        args.iterations,
                        args.graph_batch_size if name in GRAPH_CASES else 1,
                    )
                    result["samples"][name].append(sample)
                result["median_us"] = summarize(result["samples"])
                for name in case_order(repeat, tuple(checks)) if args.check_breakdown else ():
                    result["check_samples_us"][name].append(
                        measure_host(checks[name], torch.npu, args.check_iterations)
                    )
                if args.check_breakdown:
                    result["check_median_us"] = {
                        name: median(samples) for name, samples in result["check_samples_us"].items()
                    }
                save_report(output, report)
            print(
                json.dumps(
                    {"T": tokens, "median_us": result["median_us"], "check_median_us": result.get("check_median_us")},
                    indent=2,
                ),
                flush=True,
            )


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
    if args.include_qbmm:
        report["cases"].update(
            {
                "R_qbmm_eager": "Same-run native composition, including per-call transpose/contiguous and stack",
                "R_qbmm_graph": "Graph replay of the same native composition; no outside prepacking",
            }
        )
    if args.validation_addback:
        report["cases"].update(
            {
                "V_control": "No-op check then the identical B allocation/JIT path; wrapper control",
                "V_input_metadata": "Actual input validator then the identical B allocation/JIT path",
                "V_option_metadata": "Option validator then the identical B allocation/JIT path",
                "V_format": "Four-tensor ND format checks then the identical B allocation/JIT path",
                "V_all": "All validation checks then the identical B allocation/JIT path",
            }
        )
        report["addback_note"] = (
            "All V cases use the same check_then_launch wrapper and unchanged B allocation/launch. "
            "Compare V_* with V_control, and V_control with B to expose instrumentation overhead. "
            "V_all with A estimates remaining wrapper/order differences, not a proved runtime defect. "
            "V format checks precede allocation; the public wrapper checks format after allocation. "
            "All V cases pass the same strict original/changed-input gates before timing."
        )
    if args.prepared:
        report["cases"].update(
            {
                "S_safe_out": "Public out API: full validation including allocation-alias rejection on every call",
                "P_prepared_reuse": "Validated fixed-buffer binding: cached geometry/config, serial owned output",
                "P_graph_prepared": "Graph replay of the same fixed-buffer binding, reusing its one serial output",
            }
        )
        report["prepared_note"] = (
            "Preparation validates all inputs/output and geometry before timing. No unchecked diagnostic shortcut. "
            "Bindings retain detached tensor views; input copy_ updates remain visible. No metadata/storage mutation. "
            "Output is consumed before serial reuse. Not a concurrent serving benchmark or automatic model wiring. "
            "Graph capture and JIT compilation are excluded; strict changed-input gates apply to all paths."
        )
    if args.check_breakdown:
        report["check_note"] = (
            "Check-only CPU wall-time microbenchmarks use the same fixed NPU tensors, no matmul or output allocation. "
            "empty_call_control is reported, not subtracted. Query costs may include runtime blocking. "
            "Isolated costs need not sum exactly to the full-wrapper difference."
        )
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
