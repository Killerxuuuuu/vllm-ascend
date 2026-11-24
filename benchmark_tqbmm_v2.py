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
LAYOUT_CASES = ("composition", "native_prepacked", "v2", "v2_prepacked")
MACRO_CASES = ("v2_macro", "v2_macro_prepacked")
TILE_MAJOR_CASE = "v2_macro_tile_major"


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
    parser.add_argument("--layout-abcd", action="store_true", help="G=1 prequantized native/layout comparison, no V1")
    parser.add_argument(
        "--macro-tile",
        type=int,
        nargs=2,
        metavar=("BM", "BN"),
        help="Add original/packed block-IO kernels to ABCD. BK and stages match the V2 control.",
    )
    parser.add_argument(
        "--tile-major-weight",
        action="store_true",
        help="Add G: compare F with the same macro kernel reading a static tile-major W (setup excluded).",
    )
    parser.add_argument(
        "--profile-metric",
        choices=("PipeUtilization", "MemoryAccess", "L2Cache"),
        help="Profile Graph replays instead of timing; requires --layout-abcd --timing-mode graph",
    )
    parser.add_argument(
        "--profile-cases",
        nargs="+",
        choices=(*LAYOUT_CASES, *MACRO_CASES, TILE_MAJOR_CASE),
        help="Default: all enabled layouts in the same worker",
    )
    parser.add_argument("--profile-replays", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.macro_tile and (
        not args.layout_abcd
        or any(size not in (16, 32, 64, 128) for size in args.macro_tile)
        or args.block_shape[2] not in (32, 64, 128)
    ):
        parser.error("--macro-tile requires --layout-abcd, BM/BN in 16,32,64,128 and BK in 32,64,128")
    if not args.macro_tile and any(name in MACRO_CASES for name in args.profile_cases or ()):
        parser.error("Macro profile cases require --macro-tile")
    if args.tile_major_weight and not args.macro_tile:
        parser.error("--tile-major-weight requires --macro-tile and --layout-abcd")
    if TILE_MAJOR_CASE in (args.profile_cases or ()) and not args.tile_major_weight:
        parser.error(f"{TILE_MAJOR_CASE} requires --tile-major-weight")
    if args.profile_metric and (not args.layout_abcd or args.timing_mode != "graph" or args.check_only):
        parser.error("--profile-metric requires --layout-abcd --timing-mode graph and is separate from --check-only")
    if args.profile_cases and not args.profile_metric:
        parser.error("--profile-cases requires --profile-metric")
    if args.profile_replays < 1:
        parser.error("--profile-replays must be positive")
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
    if args.layout_abcd:
        command.append("--layout-abcd")
    if args.macro_tile:
        command.extend(["--macro-tile", *map(str, args.macro_tile)])
    if args.tile_major_weight:
        command.append("--tile-major-weight")
    if args.profile_metric:
        command.extend(["--profile-metric", args.profile_metric, "--profile-replays", str(args.profile_replays)])
        if args.profile_cases:
            command.extend(["--profile-cases", *dict.fromkeys(args.profile_cases)])
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
        "profile_tqbmm.py",
        "analyze_tqbmm_profile.py",
        "vllm_ascend/ops/triton/transpose_quant_batch_matmul_v2.py",
        "vllm_ascend/ops/triton/transpose_quant_batch_matmul.py",
        "tools/dsv4_a3_wo/benchmark_tqbmm.py",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def layout_cases(q, weight, sx, sw, dtype, config, reference, native, prepare):
    """Pack once. B's 2-D weight is a view of D's 3-D allocation, not another copy."""
    if weight.shape[0] != 1 or weight.is_contiguous() or weight.stride(1) != 1:
        raise ValueError("ABCD requires G=1 and the original noncontiguous transposed [G,K,N] weight")
    packed = weight.contiguous()
    q2, w2, sx1, sw1 = q[:, 0], packed[0], sx[:, 0], sw[0]
    if not all(tensor.is_contiguous() for tensor in (q2, w2, sx1, sw1)):
        raise ValueError("Native B requires contiguous input/scale views; no hidden packing")
    return {
        "composition": lambda: reference(q, weight, sx, sw, dtype),
        "native_prepacked": lambda: native(q2, w2, sw1, pertoken_scale=sx1, output_dtype=dtype).unsqueeze(1),
        "v2": prepare(q, weight, sx, sw, config=config, output_dtype=dtype),
        "v2_prepacked": prepare(q, packed, sx, sw, config=config, output_dtype=dtype),
    }, packed


def check_layout_graphs(torch, functions, q, *, original_cpu=None):
    """Replay on the NPU, but keep validation buffers and reductions off the device."""
    golden_cpu = functions["composition"]().cpu()
    if not golden_cpu.any().item():
        raise ValueError("Changed-input gate needs a nonzero reference")
    zero_cpu = torch.zeros_like(golden_cpu)
    if original_cpu is None:
        original_cpu = q.cpu()
    for name, function in functions.items():
        function()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = function()
        try:
            q.zero_()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured.cpu(), zero_cpu, atol=0, rtol=0)
        finally:
            q.copy_(original_cpu)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured.cpu(), golden_cpu, atol=0, rtol=0)
        del captured, graph
        print(f"Changed-input graph check passed: {name}", flush=True)


def tile_cost(t, g, k, n, bm, bn, bk):
    """Source-level INT8 operand bytes (valid lanes); not HBM transactions or allocated UB."""
    mt, nt, kt = (t + bm - 1) // bm, (n + bn - 1) // bn, (k + bk - 1) // bk
    x = g * t * k * nt
    w = g * k * n * mt
    return {
        "x_bytes": x,
        "w_bytes": w,
        "input_bytes": x + w,
        "tiles": g * mt * nt,
        "logical_dots": g * mt * nt * kt,
        "acc_bytes_per_tile": bm * bn * 4,
        "int8_operand_bytes_per_k_step": (bm + bn) * bk,
    }


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
        PreparedMacroTQBMMV2,
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
        output_dtype = activation.dtype
        q, sx = torch_npu.npu_dynamic_quant(activation.view(tokens * groups, inner), dst_type=torch.int8)
        q, sx = q.view(tokens, groups, inner), sx.view(tokens, groups)
        layout_metadata = {}
        if args.layout_abcd:
            # Layout cases only consume the quantized activation from here on.
            del activation
            prequantized, packed = layout_cases(
                q, weight, sx, sw, output_dtype, config, qbmm_reference, torch_npu.npu_quant_matmul, PreparedTQBMMV2
            )
            dynamic = {}
            layout_metadata = {
                "packed_weight_stride": list(packed.stride()),
                "packed_weight_format": torch_npu.get_npu_format(packed),
                "scope": "G=1 prequantized only; B/D share packed ND weight; A retains copies and stack",
            }
            if args.macro_tile:
                macro_shape = (*args.macro_tile, config.block_shape[2])
                for name, operand in zip(MACRO_CASES, (weight, packed)):
                    prequantized[name] = PreparedMacroTQBMMV2(
                        q,
                        operand,
                        sx,
                        sw,
                        block_shape=macro_shape,
                        num_stages=config.num_stages,
                        output_dtype=output_dtype,
                    )
                if args.tile_major_weight:
                    prequantized[TILE_MAJOR_CASE] = PreparedMacroTQBMMV2(
                        q,
                        packed,
                        sx,
                        sw,
                        block_shape=macro_shape,
                        num_stages=config.num_stages,
                        output_dtype=output_dtype,
                        tile_major_weight=True,
                    )
                    bn = macro_shape[1]
                    layout_metadata["tile_major_weight"] = {
                        "physical_shape_G_Npanels_K_BN": [groups, (columns + bn - 1) // bn, inner, bn],
                        "w_row_stride_elements": bn,
                        "packing_in_timing_or_capture": False,
                    }
                layout_metadata.update(
                    macro_shape=macro_shape,
                    macro_schedule="one program per output rectangle",
                    predicted_v2=tile_cost(tokens, groups, inner, columns, *config.block_shape),
                    predicted_macro=tile_cost(tokens, groups, inner, columns, *macro_shape),
                )
        else:
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
        domains = {"prequantized": prequantized}
        if dynamic:
            domains["including_quant"] = dynamic
        correctness = {}
        for domain, functions in domains.items():
            expected = functions["composition"]()
            for name, function in functions.items():
                actual = function()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=False)
                correctness[f"{domain}/{name}"] = error_metrics(actual, expected)
                print(f"Strict check passed: {domain}/{name}", flush=True)
                del actual
            del expected
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
                        golden = ((integer.float() * sw_cpu[group, col]) * sx_cpu[row, group]).to(output_dtype)
                        torch.testing.assert_close(actual_cpu[row, group, col], golden, rtol=0, atol=0)
        # Dynamic bindings must read changed data, not cached activation ptrs.
        if args.layout_abcd:
            check_layout_graphs(torch, prequantized, q, original_cpu=q_cpu)
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
        profiles = {}
        if args.profile_metric:
            from profile_tqbmm import capture_layout_graph

            expected = prequantized["composition"]()
            for name in dict.fromkeys(args.profile_cases or prequantized):
                case_dir = args.output / f"T{tokens}-seed{args.seeds[0]}" / name
                profiles[name] = capture_layout_graph(
                    torch, torch_npu.profiler, prequantized[name], expected, name, case_dir, args
                )
        elif not args.check_only:
            modes = ("eager", "graph") if args.timing_mode == "both" else (args.timing_mode,)
            for mode in modes:
                args.current_mode = mode
                timings[mode] = {domain: measure(functions, args, elapsed_ms) for domain, functions in domains.items()}
        return {
            "v2_experiment": "passed",
            "T": tokens,
            "seed": args.seeds[0],
            "shape_T_G_K_N": [tokens, groups, inner, columns],
            "weight_stride": list(weight.stride()),
            **({"layout_abcd": layout_metadata} if args.layout_abcd else {"v1": v1_options()}),
            **({"profiles": profiles} if args.profile_metric else {}),
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
                    **(
                        {"time_over_native_prepacked": duration / durations["native_prepacked"]}
                        if "native_prepacked" in durations
                        else {"speedup_vs_v1": durations["v1"] / duration}
                    ),
                }


def layout_comparisons(report):
    if "layout_abcd" not in report:
        return
    for mode, domains in report["timings"].items():
        ms = domains["prequantized"]["median_ms"]
        a, b, c, d = (ms[name] for name in LAYOUT_CASES)
        yield {
            "T": report["T"],
            "seed": report["seed"],
            "mode": mode,
            "A_ms": a,
            "B_ms": b,
            "C_ms": c,
            "D_ms": d,
            "A_over_B": a / b,
            "C_over_D": c / d,
            "D_over_B": d / b,
            "A_over_C": a / c,
            "A_minus_B_ms": a - b,
            "D_minus_B_ms": d - b,
            **(
                {
                    "E_macro_ms": ms["v2_macro"],
                    "F_macro_packed_ms": ms["v2_macro_prepacked"],
                    "A_over_E": a / ms["v2_macro"],
                    "C_over_E": c / ms["v2_macro"],
                    "D_over_F": d / ms["v2_macro_prepacked"],
                    "F_over_B": ms["v2_macro_prepacked"] / b,
                }
                if "v2_macro" in ms
                else {}
            ),
            **(
                {
                    "G_tile_major_ms": ms[TILE_MAJOR_CASE],
                    "F_over_G": ms["v2_macro_prepacked"] / ms[TILE_MAJOR_CASE],
                    "A_over_G": a / ms[TILE_MAJOR_CASE],
                }
                if TILE_MAJOR_CASE in ms
                else {}
            ),
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
        for filename, extract in (("timings.csv", timing_rows), ("layout_comparison.csv", layout_comparisons)):
            rows = [row for report in payload["reports"] for row in extract(report)]
            if rows:
                with (args.output / filename).open("x", newline="", encoding="utf-8") as handle:
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
