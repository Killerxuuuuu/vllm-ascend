# SPDX-License-Identifier: Apache-2.0
"""Offline, correctness-gated A3 tiling search. Never changes kernel defaults.

Workers use the current interpreter in fresh processes; compiler failures do
not trigger a fallback or enter the rankings. Run in the vLLM NPU environment,
not ModelSlim's venv. UB estimates are screening heuristics, not a guarantee
about compiler allocation or a queried device capacity.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
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
        raise RuntimeError(f"Worker timed out; see {log}") from error
    log.write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Worker exit={process.returncode}; see {log}")
    return extract_report(process.stdout, expected_key)


def rankings(results, tokens):
    winners = {}
    valid = [result for result in results if result["status"] == "passed"]
    for count in tokens:
        winners[str(count)] = {}
        for mode, key in (("eager", "timing_ms"), ("graph", "graph_timing_ms")):
            available = [
                (result, result["reports"][str(count)]) for result in valid if key in result["reports"][str(count)]
            ]
            if available:
                winner, report = min(available, key=lambda pair: pair[1][key]["triton_prequantized"])
                winners[str(count)][mode] = {"candidate": winner["tag"], **report[key]}
    return winners


def save_reports(output, report):
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tokens", nargs="+", type=int, default=list(DEFAULT_TOKENS))
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--transpose-weight-tile", action="store_true", help="Also test opt-in [N,K] tile loads")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ub-budget-kib", type=int, default=SCREEN_UB_BUDGET_KIB)
    parser.add_argument("--timeout", type=int, default=600, help="Per-worker seconds")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timing-mode", choices=("eager", "graph", "both"), default="both")
    args = parser.parse_args(argv)
    if min(*args.tokens, args.ub_budget_kib, args.timeout, args.iterations, args.repeats, args.tp_size) <= 0:
        parser.error("Counts, timeout and budget must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    choices = candidates(args.extended, args.ub_budget_kib, args.transpose_weight_tile)
    if not choices:
        parser.error("UB screening rejected every candidate")
    plan = [
        {"tag": choice.tag, **asdict(choice), "estimated_ub_bytes": choice.estimated_ub_bytes} for choice in choices
    ]
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not (args.quant_model / "config.json").is_file():
        parser.error("quant-model must point to an existing exported model")
    output = args.output or Path.cwd() / "tqbmm-tuning" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "quant_model": str(args.quant_model.resolve()),
        "plan": plan,
        "note": "No defaults changed. UB screening is heuristic; timings need A3 validation.",
        "results": [],
    }
    base = [sys.executable, "-m", "tools.dsv4_a3_wo.benchmark_tqbmm", "--device", args.device]
    for index, choice in enumerate(choices):
        print(f"[{index + 1}/{len(choices)}] {choice.tag}: correctness gates first", flush=True)
        result = {"tag": choice.tag, "status": "checking", "reports": {}}
        report["results"].append(result)
        common = base + choice.options()
        try:
            for flag, key in (("--layout-check", "layout_check"), ("--k-loop-check", "k_loop_check")):
                check = run_worker(common + [flag], output / f"{choice.tag}-{key}.log", args.timeout, key)
                if check[key] != "passed":
                    raise ValueError(f"Gate failed: {key}")
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
                ]
                measurements = run_worker(command, output / f"{choice.tag}-T{count}.log", args.timeout, "shape_T_G_K_N")
                validate_exact_report(measurements, args.timing_mode)
                result["reports"][str(count)] = measurements
                print(f"  T={count}: exact outputs + graph replay passed, timings recorded", flush=True)
            result["status"] = "passed"
        except (RuntimeError, ValueError, KeyError) as error:
            result["status"] = "rejected"
            result["error"] = str(error)
            print(f"  REJECTED: {error}", flush=True)
        report["best_measured"] = rankings(report["results"], args.tokens)
        save_reports(output, report)
        if index == 0 and result["status"] != "passed":
            print("Known baseline failed; stop and inspect its log before tuning.", flush=True)
            return 1
    print(f"Results: {output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
