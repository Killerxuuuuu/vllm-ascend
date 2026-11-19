# SPDX-License-Identifier: Apache-2.0
"""Replay deployment shapes, screen large-T tiles, then verify a shortlist.

No model/kernel defaults are changed. Workers are serial, isolated processes
in the current Python environment. The native reference is the existing
dynamic-quant + per-group quant-matmul + stack composition, not another layer
with a coincidentally identical shape in a whole-model profiler report.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

from tools.dsv4_a3_wo.tune_tqbmm import (
    ACTIVATION_SCRATCH_BYTES_PER_ELEMENT,
    AUTOMATIC_PIPELINE_SCREEN_FACTOR,
    EPILOGUE_SCRATCH_BYTES_PER_ELEMENT,
    VECTOR_PADDING_BYTES,
    WEIGHT_SCRATCH_BYTES_PER_ELEMENT,
    WorkerFailure,
    run_worker,
    validate_exact_report,
    write_json,
)
from tools.dsv4_a3_wo.tune_tqbmm import (
    Candidate as TileCandidate,
)


@dataclass(frozen=True)
class Candidate(TileCandidate):
    reuse_m: int = 1
    reuse_n: int = 1

    @property
    def tag(self):
        return (
            super().tag
            + (f"-rm{self.reuse_m}" if self.reuse_m != 1 else "")
            + (f"-rn{self.reuse_n}" if self.reuse_n != 1 else "")
        )

    def options(self):
        return (
            super().options()
            + (["--reuse-m", str(self.reuse_m)] if self.reuse_m != 1 else [])
            + (["--reuse-n", str(self.reuse_n)] if self.reuse_n != 1 else [])
        )

    @property
    def estimated_ub_bytes(self):
        # Account for live A/B tiles and all independent accumulators. This is a
        # conservative heuristic, NOT a queried capacity or compiler allocation.
        factor = AUTOMATIC_PIPELINE_SCREEN_FACTOR if self.num_stages == 2 else 1
        extra = (
            (self.reuse_m - 1) * ACTIVATION_SCRATCH_BYTES_PER_ELEMENT * self.block_m * self.block_k
            + (self.reuse_n - 1)
            * (2 * VECTOR_PADDING_BYTES + WEIGHT_SCRATCH_BYTES_PER_ELEMENT)
            * self.block_n
            * self.block_k
            + (self.reuse_m * self.reuse_n - 1) * EPILOGUE_SCRATCH_BYTES_PER_ELEMENT * self.block_m * self.block_n
        )
        return super().estimated_ub_bytes + extra * factor


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENS = (8, 32, 512, 1024, 4096, 7224, 8192)
CURRENT = Candidate(64, 64, 128, 2)
FOCUSED_SHAPES = (
    (64, 32, 128),
    (64, 64, 64),
    (64, 64, 128),
    (128, 32, 128),
    (128, 64, 64),
    (128, 64, 128),
    (64, 128, 64),
)
WIDE_EXTRA_SHAPES = ((128, 128, 64), (128, 32, 64), (32, 64, 128))
MODES = (("eager", "timing_ms"), ("graph", "graph_timing_ms"))
MIN_TIMING_REPEATS = 3
MAX_TILE_SIZE = 128
MAX_CANDIDATES = 64


def build_candidates(args):
    shapes = FOCUSED_SHAPES + (WIDE_EXTRA_SHAPES if args.search_space == "wide" else ())
    choices = (
        [Candidate(*values) for values in args.candidate]
        if args.candidate
        else [Candidate(*shape, stage) for shape in shapes for stage in (1, 2)]
    )
    if args.tile_loads == "both":
        choices += [Candidate(c.block_m, c.block_n, c.block_k, c.num_stages, True) for c in tuple(choices)]
    choices = [
        Candidate(c.block_m, c.block_n, c.block_k, c.num_stages, c.transpose_weight_tile, reuse_m, reuse_n)
        for c in choices
        for reuse_m in args.reuse_m
        for reuse_n in args.reuse_n
    ]
    control = Candidate(*args.control)
    for c in [control, *choices]:
        if any(v < 16 or v > MAX_TILE_SIZE or v & (v - 1) for v in (c.block_m, c.block_n, c.block_k)):
            raise ValueError("Tile sizes must be powers of two between 16 and 128")
        if c.block_k < 32 or c.num_stages not in (1, 2):
            raise ValueError("K tile must be >=32, stages must be 1 or 2")
        if c.block_n >= 128 and c.block_k >= 128:
            raise ValueError("N>=128 with K>=128 remains excluded after historical UB overflow")
    choices = list(dict.fromkeys([control, *choices]))
    if len(choices) > MAX_CANDIDATES:
        raise ValueError(f"At most {MAX_CANDIDATES} candidates may be tested")
    return choices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("baseline", "all"), default="all")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tokens", nargs="+", type=int, default=list(DEFAULT_TOKENS))
    parser.add_argument("--screen-tokens", nargs="+", type=int, default=[8192])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--search-space", choices=("focused", "wide"), default="focused")
    parser.add_argument("--candidate", nargs=4, type=int, action="append", metavar=("M", "N", "K", "STAGES"))
    parser.add_argument("--tile-loads", choices=("direct", "both"), default="direct")
    parser.add_argument("--reuse-m", type=int, nargs="+", choices=(1, 2), default=[1])
    parser.add_argument("--reuse-n", type=int, nargs="+", choices=(1, 2), default=[1])
    parser.add_argument("--control", type=int, nargs=4, default=[64, 64, 128, 2], metavar=("M", "N", "K", "STAGES"))
    parser.add_argument("--top-k", type=int, default=3, help="Non-control finalists selected by screening")
    parser.add_argument("--selection-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--objective", choices=("including_quant", "prequantized"), default="including_quant")
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--allow-internal-format", choices=("inherit", "on", "off"), default="inherit")
    parser.add_argument("--expected-gkn", nargs=3, type=int, default=[1, 4096, 1024])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--graph-batch-size", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("tokens", "screen_tokens", "seeds"):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    positive = [
        *args.tokens,
        *args.screen_tokens,
        *args.expected_gkn,
        args.tp_size,
        args.top_k,
        args.iterations,
        args.graph_batch_size,
        args.timeout,
    ]
    if min(positive) <= 0 or args.warmup < 0 or args.repeats < MIN_TIMING_REPEATS:
        parser.error("Counts must be positive, warmup >=0, repeats >=3")
    if args.layer < 0 or not 0 <= args.tp_rank < args.tp_size:
        parser.error("Invalid layer or TP rank")
    if not math.isfinite(args.min_speedup) or args.min_speedup < 1:
        parser.error("min-speedup must be finite and >=1")
    if not set(args.screen_tokens).issubset(args.tokens):
        parser.error("Every screen T must also be in --tokens for full verification")
    try:
        build_candidates(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def benchmark_command(args, candidate, tokens=None, seed=0):
    command = [
        sys.executable,
        "-m",
        "tools.dsv4_a3_wo.benchmark_tqbmm",
        "--device",
        args.device,
        *candidate.options(),
        "--seed",
        str(seed),
    ]
    if tokens is not None:
        command += [
            "--quant-model",
            str(args.quant_model.resolve()),
            "--layer",
            str(args.layer),
            "--tp-size",
            str(args.tp_size),
            "--tp-rank",
            str(args.tp_rank),
            "--tokens",
            str(tokens),
            "--strict",
            "--graph",
            "--timing-mode",
            "both",
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--repeats",
            str(args.repeats),
            "--graph-batch-size",
            str(args.graph_batch_size),
            "--allow-internal-format",
            args.allow_internal_format,
            "--prepared-dynamic",
            "--reject-instrumentation",
        ]
    return command


def validate_measurement(report, args, candidate, tokens, expected_format=None):
    validate_exact_report(report, "both")
    expected_source = REPO_ROOT / "tools/dsv4_a3_wo/benchmark_tqbmm.py"
    if Path(report.get("benchmark_source", "")).resolve() != expected_source.resolve():
        raise ValueError("Worker imported a different benchmark source tree")
    if report["shape_T_G_K_N"] != [tokens, *args.expected_gkn]:
        raise ValueError("Worker shape differs from the deployment G,K,N or requested T")
    if report.get("requested_block_shape") != [candidate.block_m, candidate.block_n, candidate.block_k]:
        raise ValueError("Worker did not use the requested tile")
    if report.get("num_stages") != candidate.num_stages:
        raise ValueError("Worker stages differ")
    if report.get("transpose_weight_tile") != candidate.transpose_weight_tile:
        raise ValueError("Worker tile-load mode differs")
    if report.get("reuse_m", 1) != candidate.reuse_m:
        raise ValueError("Worker M-tile reuse differs")
    if report.get("reuse_n", 1) != candidate.reuse_n:
        raise ValueError("Worker N-tile reuse differs")
    if report.get("prepared_dynamic") is not True:
        raise ValueError("Worker did not use the server's PreparedDynamicWoA path")
    if "allow_internal_format" not in report:
        raise ValueError("Worker did not record the effective internal-format setting")
    actual_format = report["allow_internal_format"]
    if args.allow_internal_format != "inherit" and actual_format != (args.allow_internal_format == "on"):
        raise ValueError("Worker ignored the explicit internal-format setting")
    if expected_format is not None and actual_format != expected_format:
        raise ValueError("Effective internal-format setting changed between workers")
    if any(report.get("instrumentation", {}).values()):
        raise ValueError("Instrumented measurements cannot enter this timing comparison")
    for mode, _ in MODES:
        samples = report["timing_samples_ms"][mode]
        for metric in (
            "triton_prequantized",
            "qbmm_composition_prequantized",
            "triton_including_quant",
            "qbmm_composition_including_quant",
        ):
            values = samples[metric]
            if len(values) != args.repeats or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Missing, nonfinite or invalid timing samples")


def measurements(result, phase, tokens):
    return [
        trial["report"]
        for trial in result["trials"]
        if trial["status"] == "passed" and trial["phase"] == phase and trial["T"] == tokens
    ]


def score(result, tokens, mode, objective, phase="screen"):
    metric = "triton_including_quant" if objective == "including_quant" else "triton_prequantized"
    values = []
    for count in tokens:
        reports = measurements(result, phase, count)
        if not reports:
            return math.inf
        values.append(median(v for r in reports for v in r["timing_samples_ms"][mode][metric]))
    return math.exp(sum(math.log(v) for v in values) / len(values))


def build_recommendations(report):
    """Per-shape/mode evidence only, never an automatically applied dispatcher."""
    candidates = [r for r in report["results"] if r["verification_status"] == "passed"]
    settings = report["settings"]
    dynamic = settings["objective"] == "including_quant"
    triton_key = "triton_including_quant" if dynamic else "triton_prequantized"
    native_key = "qbmm_composition_including_quant" if dynamic else "qbmm_composition_prequantized"
    per_shape = {}
    for count in settings["tokens"]:
        per_shape[str(count)] = {}
        for mode, _ in MODES:
            eligible = []
            for result in candidates:
                reports = measurements(result, "verify", count)
                if len(reports) != len(settings["seeds"]):
                    continue
                triton = [v for r in reports for v in r["timing_samples_ms"][mode][triton_key]]
                native = [v for r in reports for v in r["timing_samples_ms"][mode][native_key]]
                eligible.append((median(triton), result, triton, native))
            if not eligible:
                continue
            _, winner, triton, native = min(eligible, key=lambda x: (x[0], x[1]["tag"]))
            conservative = min(native) / max(triton)
            per_shape[str(count)][mode] = {
                "fastest_measured_triton": winner["tag"],
                "parameters": winner["parameters"],
                "triton_median_ms": median(triton),
                "composition_median_ms": median(native),
                "median_speedup": median(native) / median(triton),
                "triton_sample_range_ms": [min(triton), max(triton)],
                "composition_sample_range_ms": [min(native), max(native)],
                "conservative_sample_speedup": conservative,
                "suggested_path": "triton" if conservative >= settings["min_speedup"] else "composition",
                "sample_count_per_path": len(triton),
            }
    return {
        "run_status": report["run_status"],
        "objective": settings["objective"],
        "effective_internal_format": report.get("effective_internal_format"),
        "baseline_candidate": report["results"][0]["tag"],
        "by_T_and_mode": per_shape,
        "production_dispatch_applied": False,
        "note": (
            "Recommendation only at measured T/G/K/N and settings. Sample-range margin is not statistical "
            "significance, an interpolated threshold, a whole-model speedup or an A5-equivalent implementation."
        ),
    }


def save(output, report):
    write_json(output / "results.json", report)
    write_json(output / "best_config.json", build_recommendations(report))
    with (output / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["candidate", "phase", "T", "seed", "status", "wall_seconds", "failure_kind", "log", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in report["results"]:
            for trial in result["trials"]:
                writer.writerow({**trial, "candidate": result["tag"]})
    with (output / "timings.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "candidate",
                "phase",
                "T",
                "seed",
                "mode",
                "triton_ms",
                "composition_ms",
                "speedup",
                "triton_with_quant_ms",
                "composition_with_quant_ms",
                "with_quant_speedup",
                "allow_internal_format",
                "prepared_dynamic",
                "candidate_verification_status",
            ]
        )
        for result in report["results"]:
            for trial in result["trials"]:
                if trial["status"] != "passed" or trial["T"] is None:
                    continue
                for mode, key in MODES:
                    r = trial["report"]
                    v = r[key]
                    writer.writerow(
                        [
                            result["tag"],
                            trial["phase"],
                            trial["T"],
                            trial["seed"],
                            mode,
                            v["triton_prequantized"],
                            v["qbmm_composition_prequantized"],
                            v["prequantized_speedup"],
                            v["triton_including_quant"],
                            v["qbmm_composition_including_quant"],
                            v["including_quant_speedup"],
                            r["allow_internal_format"],
                            r["prepared_dynamic"],
                            result["verification_status"],
                        ]
                    )


def trial(args, output, report, result, candidate, phase, tokens=None, seed=0, gate=None):
    suffix = gate or f"T{tokens}-seed{seed}"
    log = output / "logs" / f"{candidate.tag}-{phase}-{suffix}.log"
    command = benchmark_command(args, candidate, tokens, seed)
    if tokens is not None:
        command += ["--failure-dir", str(output / "failures" / candidate.tag)]
    if gate:
        command.append("--" + gate.replace("_", "-"))
    record = {
        "phase": phase,
        "T": tokens,
        "seed": seed,
        "status": "running",
        "command": command,
        "log": str(log),
    }
    result["trials"].append(record)
    save(output, report)
    started = time.perf_counter()
    print(f"{phase}: {candidate.tag} {suffix}", flush=True)
    try:
        r = run_worker(command, log, args.timeout, gate or "shape_T_G_K_N")
        if gate:
            if r[gate] != "passed":
                raise ValueError(f"Correctness gate failed: {gate}")
            if r.get("reuse_m", 1) != candidate.reuse_m:
                raise ValueError("Correctness gate used a different reuse_m")
            if r.get("reuse_n", 1) != candidate.reuse_n:
                raise ValueError("Correctness gate used a different reuse_n")
        else:
            validate_measurement(r, args, candidate, tokens, report.get("effective_internal_format"))
            report["effective_internal_format"] = r["allow_internal_format"]
        record["report"] = r
        record["status"] = "passed"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        raise
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as error:
        record["status"] = "failed"
        record["error"] = str(error)
        record["failure_kind"] = getattr(error, "kind", "validation_error")
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        save(output, report)


def evaluate(args, output, report, result, candidate, phase, tokens, seeds):
    field = "verification_status" if phase == "verify" else "screen_status"
    result[field] = "running"
    try:
        if result["gate_status"] != "passed":
            for gate in ("layout_check", "k_loop_check"):
                trial(args, output, report, result, candidate, "gate", gate=gate)
            result["gate_status"] = "passed"
        for seed in seeds:
            for count in tokens:
                trial(args, output, report, result, candidate, phase, count, seed)
        result[field] = "passed"
    except KeyboardInterrupt:
        result[field] = "interrupted"
        raise
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as error:
        result[field] = "rejected"
        if result["gate_status"] != "passed":
            result["gate_status"] = "rejected"
        print(f"REJECTED {candidate.tag}: {error}", flush=True)
        if isinstance(error, WorkerFailure) and error.kind in ("device_error", "timeout"):
            raise
    finally:
        save(output, report)


def main(argv=None):
    args = parse_args(argv)
    choices = build_candidates(args)
    settings = {**vars(args), "quant_model": str(args.quant_model.resolve()), "output": None}
    plan = {
        "settings": settings,
        "candidates": [{"tag": c.tag, **asdict(c), "estimated_ub_bytes": c.estimated_ub_bytes} for c in choices],
        "phases": [
            "verify current server tile on every T/seed",
            "screen candidates on large T with the first seed",
            "verify top-k finalists on every T/seed; save advisory per-T/mode winners",
        ]
        if args.phase == "all"
        else ["verify current server tile on every T/seed"],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not (args.quant_model / "config.json").is_file():
        raise ValueError("quant-model must be an existing exported ModelSlim checkpoint")
    output = args.output or REPO_ROOT.parent / ("tqbmm-large-t-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    settings["output"] = str(output.resolve())
    sources = (
        "autotune_tqbmm_large_t.py",
        "tools/dsv4_a3_wo/tune_tqbmm_large_t.py",
        "tools/dsv4_a3_wo/tune_tqbmm.py",
        "tools/dsv4_a3_wo/benchmark_tqbmm.py",
        "vllm_ascend/ops/triton/transpose_quant_batch_matmul.py",
        "vllm_ascend/attention/dsa_v1.py",
        "autotune_tqbmm_reuse.py",
        "autotune_tqbmm_data_reuse.py",
    )
    report = {
        "schema_version": 1,
        "run_status": "running",
        "settings": settings,
        "python": sys.executable,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_sha256": {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in sources},
        "results": [
            {
                "tag": c.tag,
                "parameters": asdict(c),
                "estimated_ub_bytes": c.estimated_ub_bytes,
                "gate_status": "pending",
                "screen_status": "pending",
                "verification_status": "pending",
                "trials": [],
            }
            for c in choices
        ],
        "note": "Serial standalone operator timings, no profiler; no production settings or math changes.",
    }
    save(output, report)
    print(f"Results: {output.resolve()}", flush=True)
    code = 0
    try:
        baseline = report["results"][0]
        evaluate(args, output, report, baseline, choices[0], "verify", args.tokens, args.seeds)
        if baseline["verification_status"] != "passed":
            report["run_status"] = "baseline_failed"
            return 1
        if args.phase == "all":
            for candidate, result in zip(choices[1:], report["results"][1:]):
                evaluate(args, output, report, result, candidate, "screen", args.screen_tokens, args.seeds[:1])
            eligible = [r for r in report["results"][1:] if r["screen_status"] == "passed"]
            ordered = sorted(
                eligible,
                key=lambda r: score(r, args.screen_tokens, args.selection_mode, args.objective),
            )
            report["shortlist"] = [r["tag"] for r in ordered[: args.top_k]]
            for result in ordered[: args.top_k]:
                candidate = Candidate(**result["parameters"])
                evaluate(args, output, report, result, candidate, "verify", args.tokens, args.seeds)
        report["run_status"] = "completed"
    except KeyboardInterrupt:
        report["run_status"] = "interrupted"
        code = 130
    except WorkerFailure as error:
        report["run_status"] = error.kind
        report["error"] = str(error)
        code = 1
    finally:
        save(output, report)
        print(f"Status: {report['run_status']}; see {output / 'best_config.json'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
