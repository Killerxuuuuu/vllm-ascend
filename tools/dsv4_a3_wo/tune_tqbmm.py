# SPDX-License-Identifier: Apache-2.0
"""Offline, correctness-gated A3 tiling search. Never changes kernel defaults.

Workers use the current interpreter in fresh processes; compiler failures do
not trigger a fallback or enter the rankings. Run in the vLLM NPU environment,
not ModelSlim's venv. UB estimates are screening heuristics, not a guarantee
about compiler allocation or a queried device capacity.
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENS = (1, 16, 64, 512)
SCREEN_UB_BUDGET_KIB = 192
VECTOR_PADDING_BYTES = 32
WEIGHT_SCRATCH_BYTES_PER_ELEMENT = 4
ACTIVATION_SCRATCH_BYTES_PER_ELEMENT = 4
EPILOGUE_SCRATCH_BYTES_PER_ELEMENT = 24
UB_RESERVE_BYTES = 8 * 1024
AUTOMATIC_PIPELINE_SCREEN_FACTOR = 2
COSINE_ABS_TOLERANCE = 1e-6
MAX_SEARCH_CANDIDATES = 256
MAX_SEARCH_BLOCK_SIZE = 128
DEFAULT_GRAPH_BATCH_SIZE = 16
BALANCED_BLOCK_SHAPES = (
    (16, 32, 32),
    (32, 32, 32),
    (64, 32, 32),
    (16, 32, 64),
    (32, 32, 64),
    (32, 16, 64),
    (16, 64, 32),
    (64, 32, 64),
    (32, 64, 32),
)
TIMING_KEYS = (("eager", "timing_ms"), ("graph", "graph_timing_ms"))
DEVICE_FAILURE_MARKERS = (
    "illegal memory access",
    "device-side assert",
    "acl_error_rt_aicore_exception",
    "acl_error_rt_aicore_timeout",
    "device lost",
)


@dataclass(frozen=True)
class Candidate:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    transpose_weight_tile: bool = False

    @property
    def tag(self):
        suffix = "-nk" if self.transpose_weight_tile else ""
        return f"m{self.block_m}n{self.block_n}k{self.block_k}s{self.num_stages}{suffix}"

    @property
    def estimated_ub_bytes(self):
        # Reserve two padded INT8 weight tiles, cast/deinterleave scratch,
        # input scratch and a conservative FP32 epilogue allowance. Automatic
        # multi-buffer mode gets another factor of two for screening only.
        weight = (2 * VECTOR_PADDING_BYTES + WEIGHT_SCRATCH_BYTES_PER_ELEMENT) * self.block_n * self.block_k
        activation = ACTIVATION_SCRATCH_BYTES_PER_ELEMENT * self.block_m * self.block_k
        epilogue = EPILOGUE_SCRATCH_BYTES_PER_ELEMENT * self.block_m * self.block_n
        factor = AUTOMATIC_PIPELINE_SCREEN_FACTOR if self.num_stages == 2 else 1
        return (weight + activation + epilogue + UB_RESERVE_BYTES) * factor

    def options(self):
        options = [
            "--block-shape",
            str(self.block_m),
            str(self.block_n),
            str(self.block_k),
            "--num-stages",
            str(self.num_stages),
        ]
        return options + (["--transpose-weight-tile"] if self.transpose_weight_tile else [])


def candidates(extended=False, budget_kib=SCREEN_UB_BUDGET_KIB, transpose_weight_tile=False):
    # First increase M to reuse weights, then test pipelining at the smallest
    # known tile. Larger N/K tiles are explicitly opt-in; never try 128x128.
    choices = [Candidate(m, 32, 32, 1) for m in (16, 32, 64)]
    choices.append(Candidate(16, 32, 32, 2))
    if extended:
        choices.extend([Candidate(16, 32, 64, 1), Candidate(32, 32, 64, 1), Candidate(16, 64, 32, 1)])
    if transpose_weight_tile:
        choices.extend(
            [
                Candidate(choice.block_m, choice.block_n, choice.block_k, choice.num_stages, True)
                for choice in tuple(choices)
            ]
        )
    return [choice for choice in choices if choice.estimated_ub_bytes <= budget_kib * 1024]


def search_candidates(args):
    """Keep the old search available; new searches do not trust UB estimates.

    A known, direct-load control always runs first, even when it is outside a
    custom grid. Extremely large N/K combinations remain explicitly excluded.
    """
    custom_grid = any(value is not None for value in (args.block_m, args.block_n, args.block_k, args.stages))
    if custom_grid and args.search_space != "grid":
        raise ValueError("--block-m/--block-n/--block-k/--stages require --search-space grid")
    if args.candidate and custom_grid:
        raise ValueError("Use either --candidate or grid axes, not both")
    if args.candidate:
        choices = [Candidate(*values) for values in args.candidate]
    elif args.search_space == "legacy":
        if args.tile_loads == "nk":
            raise ValueError("Use balanced/grid or --candidate for NK-only searches")
        return candidates(args.extended, args.ub_budget_kib, args.transpose_weight_tile or args.tile_loads == "both")
    elif args.search_space == "balanced":
        shapes = list(BALANCED_BLOCK_SHAPES)
        if args.extended:
            shapes.extend(((32, 16, 128), (16, 32, 128)))
        choices = [Candidate(*shape, stage) for shape in shapes for stage in (1, 2)]
    else:
        choices = [
            Candidate(*values)
            for values in product(
                args.block_m or (16, 32, 64),
                args.block_n or (16, 32, 64),
                args.block_k or (32, 64),
                args.stages or (1, 2),
            )
        ]
    load_mode = args.tile_loads or ("both" if args.transpose_weight_tile else "direct")
    load_modes = (False, True) if load_mode == "both" else (load_mode == "nk",)
    choices = [
        Candidate(choice.block_m, choice.block_n, choice.block_k, choice.num_stages, tile_load)
        for choice in choices
        for tile_load in load_modes
    ]
    for choice in choices:
        sizes = (choice.block_m, choice.block_n, choice.block_k)
        if any(size < 16 or size > MAX_SEARCH_BLOCK_SIZE or size & (size - 1) for size in sizes):
            raise ValueError("Tile sizes must be powers of two from 16 to 128")
        if choice.block_k < 32 or choice.num_stages not in (1, 2):
            raise ValueError("INT8 K tile must be >=32; stages must be 1 or 2")
        if choice.block_n >= 128 and choice.block_k >= 128:
            raise ValueError("N>=128 together with K>=128 is excluded after the historical UB overflow")
    control = Candidate(16, 32, 32, 1)
    choices = list(dict.fromkeys([control, *choices]))
    measured_candidate = Candidate(32, 32, 64, 2)
    if measured_candidate in choices:
        choices.remove(measured_candidate)
        choices.insert(1, measured_candidate)
    return choices


class WorkerFailure(RuntimeError):
    def __init__(self, message, kind):
        super().__init__(message)
        self.kind = kind


def failure_kind(output):
    lowered = output.lower()
    if any(marker in lowered for marker in DEVICE_FAILURE_MARKERS):
        return "device_error"
    if any(marker in lowered for marker in ("ub overflow", "ub buffer", "ub out of", "out of memory")):
        return "resource_error"
    if "assertionerror" in lowered or "tensor-likes are not close" in lowered:
        return "correctness_error"
    return "worker_error"


def extract_report(output, expected_key):
    """Ignore import logs and per-case JSON lines, but require a final report."""
    decoder = json.JSONDecoder()
    for offset, character in enumerate(output):
        if character != "{":
            continue
        try:
            report, _ = decoder.raw_decode(output[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict) and expected_key in report:
            return report
    raise ValueError(f"No benchmark JSON report containing {expected_key!r}")


def validate_exact_report(report, timing_mode="both"):
    if report.get("strict_correctness") is not True:
        raise ValueError("Worker did not enable strict correctness")
    for name in ("same_int8_inputs", "including_dynamic_quant"):
        metrics = report[name]
        if metrics["max_abs"] != 0 or metrics["nrmse"] != 0:
            raise ValueError(f"Nonzero or nonfinite error metrics: {name}={metrics}")
        cosine = metrics["cosine"]
        if not math.isfinite(cosine) or not math.isclose(cosine, 1.0, abs_tol=COSINE_ABS_TOLERANCE):
            raise ValueError(f"Invalid cosine: {name}={cosine}")
    if report.get("graph_replay_with_changed_input") != "passed":
        raise ValueError("Changed-input graph replay was not validated")
    keys = (
        ("timing_ms", "graph_timing_ms")
        if timing_mode == "both"
        else ("timing_ms" if timing_mode == "eager" else "graph_timing_ms",)
    )
    for key in keys:
        values = report[key]
        required = (
            "triton_prequantized",
            "qbmm_composition_prequantized",
            "prequantized_speedup",
            "triton_including_quant",
            "qbmm_composition_including_quant",
            "including_quant_speedup",
        )
        if any(not math.isfinite(values[name]) or values[name] <= 0 for name in required):
            raise ValueError(f"Invalid timing results: {key}")


def run_worker(command, log, timeout, expected_key):
    try:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log.write_text(output + "\nWORKER TIMEOUT\n", encoding="utf-8")
        raise WorkerFailure(f"Worker timed out; see {log}", "timeout") from error
    log.write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise WorkerFailure(f"Worker exit={process.returncode}; see {log}", failure_kind(process.stdout))
    return extract_report(process.stdout, expected_key)


def rankings(results, tokens, metric="triton_prequantized"):
    winners = {}
    valid = [result for result in results if result["status"] == "passed"]
    for count in tokens:
        winners[str(count)] = {}
        for mode, key in TIMING_KEYS:
            available = [
                (result, result["reports"][str(count)]) for result in valid if key in result["reports"][str(count)]
            ]
            if available:
                winner, report = min(available, key=lambda pair: pair[1][key][metric])
                winners[str(count)][mode] = {
                    "candidate": winner["tag"],
                    "parameters": winner.get("parameters"),
                    **report[key],
                }
    return winners


def best_configuration(results, tokens, mode="graph", objective="prequantized"):
    """One shared configuration, with equal weight for each requested T.

    Geometric-mean latency is not a claim about the deployment request mix.
    A fast partial candidate is never eligible, even if one shape passed.
    """
    key = "graph_timing_ms" if mode == "graph" else "timing_ms"
    metric = "triton_prequantized" if objective == "prequantized" else "triton_including_quant"
    eligible = []
    for result in results:
        if result["status"] != "passed":
            continue
        values = [result["reports"].get(str(count), {}).get(key, {}).get(metric) for count in tokens]
        if not values or any(value is None or not math.isfinite(value) or value <= 0 for value in values):
            continue
        score = math.exp(sum(math.log(value) for value in values) / len(values))
        eligible.append((score, result, values))
    if not eligible:
        return None
    score, winner, values = min(eligible, key=lambda entry: entry[0])
    return {
        "candidate": winner["tag"],
        "parameters": winner.get("parameters"),
        "benchmark_options": winner.get("benchmark_options"),
        "selection_mode": mode,
        "objective": objective,
        "score_geometric_mean_ms": score,
        "latency_ms_by_T": dict(zip(map(str, tokens), values)),
        "selection_note": "Measured best among fully passed candidates; equal weight per T, not a production default.",
    }


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    files = (
        "autotune_tqbmm.py",
        "tools/dsv4_a3_wo/tune_tqbmm.py",
        "tools/dsv4_a3_wo/benchmark_tqbmm.py",
        "vllm_ascend/ops/triton/transpose_quant_batch_matmul.py",
    )
    return {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in files}


def save_reports(output, report):
    # Checkpoint recommendations after every trial, not only after a long scan.
    tokens = report.get("tokens") or sorted({int(count) for result in report["results"] for count in result["reports"]})
    selection = report.get("selection", {"mode": "graph", "objective": "prequantized"})
    best = best_configuration(report["results"], tokens, selection["mode"], selection["objective"])
    report["best_measured"] = rankings(report["results"], tokens)
    write_json(output / "results.json", report)
    write_json(
        output / "best_config.json",
        {
            "run_status": report.get("run_status", "unknown"),
            "selection": selection,
            "overall_best": best,
            "best_by_T": report["best_measured"],
            "best_by_T_including_quant": rankings(report["results"], tokens, "triton_including_quant"),
            "note": "No kernel or model defaults are modified; interrupted scans are incomplete searches.",
        },
    )
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "candidate",
                "status",
                "M",
                "N",
                "K",
                "stages",
                "transpose_weight_tile",
                "estimated_ub_bytes",
                "wall_seconds",
                "failure_kind",
                "error",
            ]
        )
        for result in report["results"]:
            params = result.get("parameters") or {}
            writer.writerow(
                [
                    result["tag"],
                    result["status"],
                    *[
                        params.get(name)
                        for name in ("block_m", "block_n", "block_k", "num_stages", "transpose_weight_tile")
                    ],
                    result.get("estimated_ub_bytes"),
                    result.get("wall_seconds"),
                    result.get("failure_kind"),
                    result.get("error"),
                ]
            )
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate", "phase", "T", "status", "wall_seconds", "log", "error"])
        for result in report["results"]:
            for trial in result.get("trials", []):
                writer.writerow(
                    [
                        result["tag"],
                        *[trial.get(name) for name in ("phase", "T", "status", "wall_seconds", "log", "error")],
                    ]
                )
    with (output / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "candidate",
                "status",
                "T",
                "mode",
                "triton_ms",
                "qbmm_ms",
                "speedup",
                "triton_with_quant_ms",
                "qbmm_with_quant_ms",
                "with_quant_speedup",
            ]
        )
        for result in report["results"]:
            for count, measurements in result.get("reports", {}).items():
                for mode, key in (("eager", "timing_ms"), ("graph", "graph_timing_ms")):
                    if key not in measurements:
                        continue
                    values = measurements[key]
                    writer.writerow(
                        [
                            result["tag"],
                            result["status"],
                            count,
                            mode,
                            values["triton_prequantized"],
                            values["qbmm_composition_prequantized"],
                            values["prequantized_speedup"],
                            values["triton_including_quant"],
                            values["qbmm_composition_including_quant"],
                            values["including_quant_speedup"],
                        ]
                    )


def checked_trial(command, output, report, result, timeout, expected_key, tokens=None):
    phase = expected_key if tokens is None else "benchmark"
    suffix = expected_key if tokens is None else f"T{tokens}"
    log = output / f"{result['tag']}-{suffix}.log"
    trial = {"phase": phase, "T": tokens, "status": "running", "command": command, "log": str(log)}
    result["trials"].append(trial)
    save_reports(output, report)
    started = time.perf_counter()
    try:
        measurements = run_worker(command, log, timeout, expected_key)
        if tokens is None:
            if measurements[expected_key] != "passed":
                raise ValueError(f"Gate failed: {expected_key}")
        else:
            validate_exact_report(measurements, report["settings"]["timing_mode"])
            if measurements["shape_T_G_K_N"][0] != tokens:
                raise ValueError("Worker returned the wrong token count")
            result["reports"][str(tokens)] = measurements
        trial["status"] = "passed"
        return measurements
    except KeyboardInterrupt:
        trial["status"] = "interrupted"
        raise
    except (RuntimeError, ValueError, KeyError, TypeError) as error:
        trial["status"] = "failed"
        trial["error"] = str(error)
        raise
    finally:
        trial["wall_seconds"] = time.perf_counter() - started
        save_reports(output, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tokens", nargs="+", type=int, default=list(DEFAULT_TOKENS))
    parser.add_argument("--search-space", choices=("legacy", "balanced", "grid"), default="legacy")
    parser.add_argument("--block-m", nargs="+", type=int)
    parser.add_argument("--block-n", nargs="+", type=int)
    parser.add_argument("--block-k", nargs="+", type=int)
    parser.add_argument("--stages", nargs="+", type=int)
    parser.add_argument("--candidate", nargs=4, type=int, action="append", metavar=("M", "N", "K", "STAGES"))
    parser.add_argument("--tile-loads", choices=("direct", "nk", "both"))
    parser.add_argument("--max-candidates", type=int, default=MAX_SEARCH_CANDIDATES)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--transpose-weight-tile", action="store_true", help="Also test opt-in [N,K] tile loads")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ub-budget-kib", type=int, default=SCREEN_UB_BUDGET_KIB)
    parser.add_argument("--timeout", type=int, default=600, help="Per-worker seconds")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timing-mode", choices=("eager", "graph", "both"), default="both")
    parser.add_argument("--selection-mode", choices=("auto", "eager", "graph"), default="auto")
    parser.add_argument("--objective", choices=("prequantized", "including_quant"), default="prequantized")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph-batch-size", type=int, default=DEFAULT_GRAPH_BATCH_SIZE)
    args = parser.parse_args(argv)
    if (
        min(
            *args.tokens,
            args.ub_budget_kib,
            args.timeout,
            args.iterations,
            args.repeats,
            args.tp_size,
            args.graph_batch_size,
            args.max_candidates,
        )
        <= 0
    ):
        parser.error("Counts, timeout and budget must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if args.layer < 0 or not 0 <= args.tp_rank < args.tp_size:
        parser.error("layer must be nonnegative and tp-rank must be in [0,tp-size)")
    args.tokens = list(dict.fromkeys(args.tokens))
    selection_mode = args.selection_mode
    if selection_mode == "auto":
        selection_mode = "eager" if args.timing_mode == "eager" else "graph"
    if args.timing_mode != "both" and selection_mode != args.timing_mode:
        parser.error("selection-mode must be present in timing-mode")
    try:
        choices = search_candidates(args)
    except ValueError as error:
        parser.error(str(error))
    if not choices:
        parser.error("UB screening rejected every candidate")
    if len(choices) > args.max_candidates:
        parser.error(f"Search contains {len(choices)} candidates; reduce the grid or increase --max-candidates")
    plan = [
        {"tag": choice.tag, **asdict(choice), "estimated_ub_bytes": choice.estimated_ub_bytes} for choice in choices
    ]
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not (args.quant_model / "config.json").is_file():
        parser.error("quant-model must point to an existing exported model")
    output = args.output or REPO_ROOT.parent / "tqbmm-tuning" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "quant_model": str(args.quant_model.resolve()),
        "plan": plan,
        "tokens": args.tokens,
        "run_status": "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.executable,
        "source_sha256": source_hashes(),
        "settings": {**vars(args), "quant_model": str(args.quant_model.resolve()), "output": str(output.resolve())},
        "selection": {"mode": selection_mode, "objective": args.objective},
        "note": (
            "No defaults changed. New searches record UB estimates but do not filter by them. A3 validation required."
        ),
        "results": [
            {
                "tag": choice.tag,
                "parameters": asdict(choice),
                "benchmark_options": choice.options(),
                "estimated_ub_bytes": choice.estimated_ub_bytes,
                "status": "pending",
                "reports": {},
                "trials": [],
            }
            for choice in choices
        ],
    }
    save_reports(output, report)
    print(f"Results: {output.resolve()}", flush=True)
    print(f"Selection: {selection_mode}, {args.objective}; {len(choices)} candidates, serial execution", flush=True)
    base = [sys.executable, "-m", "tools.dsv4_a3_wo.benchmark_tqbmm", "--device", args.device]
    exit_code = 0
    for index, choice in enumerate(choices):
        print(f"[{index + 1}/{len(choices)}] {choice.tag}: correctness gates first", flush=True)
        result = report["results"][index]
        result["status"] = "checking"
        started = time.perf_counter()
        common = base + choice.options() + ["--seed", str(args.seed)]
        try:
            for flag, key in (("--layout-check", "layout_check"), ("--k-loop-check", "k_loop_check")):
                checked_trial(common + [flag], output, report, result, args.timeout, key)
            for count in args.tokens:
                command = common + [
                    "--quant-model",
                    str(args.quant_model.resolve()),
                    "--layer",
                    str(args.layer),
                    "--tp-size",
                    str(args.tp_size),
                    "--tp-rank",
                    str(args.tp_rank),
                    "--tokens",
                    str(count),
                    "--strict",
                    "--graph",
                    "--timing-mode",
                    args.timing_mode,
                    "--warmup",
                    str(args.warmup),
                    "--iterations",
                    str(args.iterations),
                    "--repeats",
                    str(args.repeats),
                    "--graph-batch-size",
                    str(args.graph_batch_size),
                ]
                checked_trial(command, output, report, result, args.timeout, "shape_T_G_K_N", count)
                print(f"  T={count}: exact outputs + graph replay passed, timings recorded", flush=True)
            result["status"] = "passed"
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            report["run_status"] = "interrupted"
            exit_code = 130
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            result["status"] = "rejected"
            result["error"] = str(error)
            result["failure_kind"] = getattr(error, "kind", "validation_error")
            print(f"  REJECTED: {error}", flush=True)
        result["wall_seconds"] = time.perf_counter() - started
        save_reports(output, report)
        if report["run_status"] == "interrupted":
            print("Interrupted; completed measurements and provisional best parameters were saved.", flush=True)
            break
        if index == 0 and result["status"] != "passed":
            print("Known baseline failed; stop and inspect its log before tuning.", flush=True)
            report["run_status"] = "baseline_failed"
            exit_code = 1
            break
        if result.get("failure_kind") == "device_error":
            print("Device execution fault; stop before testing more candidates on an unhealthy device.", flush=True)
            report["run_status"] = "device_error"
            exit_code = 1
            break
    if report["run_status"] == "running":
        report["run_status"] = "completed"
    save_reports(output, report)
    winner = best_configuration(report["results"], args.tokens, selection_mode, args.objective)
    if winner:
        print(f"Best shared configuration: {winner['candidate']} (see best_config.json)", flush=True)
    print(f"Results: {output.resolve()}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
