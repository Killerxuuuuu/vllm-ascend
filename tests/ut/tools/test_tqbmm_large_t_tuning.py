# SPDX-License-Identifier: Apache-2.0
"""CPU-only orchestration tests; run --noconftest, no NPU performance claims."""

import ast
import json
import subprocess
import sys
from copy import deepcopy
from unittest.mock import patch

import pytest

from tools.dsv4_a3_wo import tune_tqbmm_large_t as large


def arguments(*options):
    return large.parse_args(["--quant-model", "unused", *options])


def measurement(args, candidate, tokens, triton=0.2, native=0.3):
    metrics = {"max_abs": 0.0, "nrmse": 0.0, "cosine": 0.99999994}
    samples = {
        "triton_prequantized": [triton] * args.repeats,
        "qbmm_composition_prequantized": [native] * args.repeats,
        "triton_including_quant": [triton + 0.01] * args.repeats,
        "qbmm_composition_including_quant": [native + 0.01] * args.repeats,
    }
    timings = {
        **{k: v[0] for k, v in samples.items()},
        "prequantized_speedup": native / triton,
        "including_quant_speedup": (native + 0.01) / (triton + 0.01),
    }
    return {
        "benchmark_source": str(large.REPO_ROOT / "tools/dsv4_a3_wo/benchmark_tqbmm.py"),
        "shape_T_G_K_N": [tokens, *args.expected_gkn],
        "strict_correctness": True,
        "same_int8_inputs": metrics.copy(),
        "including_dynamic_quant": metrics.copy(),
        "graph_replay_with_changed_input": "passed",
        "requested_block_shape": [candidate.block_m, candidate.block_n, candidate.block_k],
        "num_stages": candidate.num_stages,
        "transpose_weight_tile": candidate.transpose_weight_tile,
        "prepared_dynamic": True,
        "allow_internal_format": True,
        "instrumentation": {"memscope_modules": [], "preload_hooks": []},
        "timing_ms": timings.copy(),
        "graph_timing_ms": timings.copy(),
        "timing_samples_ms": {"eager": deepcopy(samples), "graph": deepcopy(samples)},
    }


def test_plan_covers_real_prefill_shapes_and_preserves_known_control():
    args = arguments()
    assert {8, 32, 1024, 4096, 7224, 8192} <= set(args.tokens)
    assert args.selection_mode == "eager"
    assert args.objective == "including_quant"
    choices = large.build_candidates(args)
    assert choices[0] == large.CURRENT
    assert len(choices) == 14
    assert all(not (c.block_n >= 128 and c.block_k >= 128) for c in choices)
    assert any(c.block_m == 128 for c in choices)
    assert all(c.num_stages in (1, 2) for c in choices)


def test_wide_and_tile_load_modes_are_explicit():
    regular = large.build_candidates(arguments())
    wide = large.build_candidates(arguments("--search-space", "wide", "--tile-loads", "both"))
    assert set(regular) < set(wide)
    assert len(set(wide)) == len(wide) == 40
    assert any(c.transpose_weight_tile for c in wide)


@pytest.mark.parametrize(
    "options",
    [
        ["--tokens", "8"],
        ["--tokens", "0", "8192"],
        ["--repeats", "2"],
        ["--min-speedup", "nan"],
        ["--min-speedup", "0.9"],
        ["--tp-size", "8", "--tp-rank", "8"],
        ["--candidate", "64", "128", "128", "2"],
        ["--candidate", "256", "32", "128", "1"],
        ["--candidate", "64", "32", "128", "3"],
    ],
)
def test_bad_options_fail_before_launch(options):
    with patch.object(large, "run_worker") as worker, pytest.raises(SystemExit):
        large.main(["--quant-model", "unused", "--dry-run", *options])
    worker.assert_not_called()


def test_root_dry_run_has_no_torch_dependency_or_output_write(tmp_path):
    output = tmp_path / "not_created"
    process = subprocess.run(
        [
            sys.executable,
            str(large.REPO_ROOT / "autotune_tqbmm_large_t.py"),
            "--quant-model",
            "unused",
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=large.REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(process.stdout)
    assert plan["settings"]["tokens"][-1] == 8192
    assert plan["candidates"][0]["tag"] == large.CURRENT.tag
    assert not output.exists()
    assert "torch" not in process.stderr


def test_worker_command_is_explicit_and_matches_prepared_server_path():
    args = arguments("--allow-internal-format", "on")
    command = large.benchmark_command(args, large.CURRENT, 8192, 1)
    for flag in ("--strict", "--graph", "--prepared-dynamic", "--reject-instrumentation"):
        assert flag in command
    assert command[command.index("--allow-internal-format") + 1] == "on"
    assert command[command.index("--tokens") + 1] == "8192"
    assert command[command.index("--timing-mode") + 1] == "both"


@pytest.mark.parametrize(
    "mutation",
    [
        "shape",
        "source",
        "tile",
        "stage",
        "load_mode",
        "prepared",
        "format",
        "injected",
        "error",
        "missing_samples",
        "nan",
    ],
)
def test_validation_rejects_wrong_inexact_or_uncontrolled_reports(mutation):
    args = arguments("--allow-internal-format", "on")
    r = measurement(args, large.CURRENT, 8192)
    if mutation == "shape":
        r["shape_T_G_K_N"][1] = 8
    elif mutation == "source":
        r["benchmark_source"] = "another_checkout/benchmark.py"
    elif mutation == "tile":
        r["requested_block_shape"] = [16, 32, 32]
    elif mutation == "stage":
        r["num_stages"] = 1
    elif mutation == "load_mode":
        r["transpose_weight_tile"] = True
    elif mutation == "prepared":
        r["prepared_dynamic"] = False
    elif mutation == "format":
        r["allow_internal_format"] = False
    elif mutation == "injected":
        r["instrumentation"]["memscope_modules"] = ["msmemscope"]
    elif mutation == "error":
        r["including_dynamic_quant"]["max_abs"] = 0.015625
    elif mutation == "missing_samples":
        r["timing_samples_ms"]["graph"]["triton_including_quant"] = []
    else:
        r["timing_samples_ms"]["eager"]["triton_prequantized"][0] = float("nan")
    with pytest.raises((ValueError, KeyError)):
        large.validate_measurement(r, args, large.CURRENT, 8192)


def test_inherit_format_must_remain_identical_between_workers():
    args = arguments()
    r = measurement(args, large.CURRENT, 8192)
    large.validate_measurement(r, args, large.CURRENT, 8192, True)
    with pytest.raises(ValueError, match="changed"):
        large.validate_measurement(r, args, large.CURRENT, 8192, False)


def run_options(tmp_path, *extra):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    return [
        "--quant-model",
        str(model),
        "--output",
        str(tmp_path / "scan"),
        "--tokens",
        "8",
        "8192",
        "--seeds",
        "0",
        "1",
        "--repeats",
        "3",
        "--candidate",
        "128",
        "64",
        "128",
        "2",
        *extra,
    ]


def worker_fixture(options, fail=None):
    args = large.parse_args(options)

    def worker(command, log, timeout, key):
        if key in ("layout_check", "k_loop_check"):
            return {key: "passed"}
        candidate = large.Candidate(
            *map(int, command[command.index("--block-shape") + 1 : command.index("--block-shape") + 4]),
            int(command[command.index("--num-stages") + 1]),
            "--transpose-weight-tile" in command,
        )
        tokens = int(command[command.index("--tokens") + 1])
        seed = int(command[command.index("--seed") + 1])
        if fail:
            fail(candidate, tokens, seed)
        r = measurement(args, candidate, tokens, triton=0.2 if candidate == large.CURRENT else 0.1)
        log.write_text(json.dumps(r), encoding="utf-8")
        return r

    return worker


def test_end_to_end_staging_records_all_trials_and_verified_winners(tmp_path):
    options = run_options(tmp_path)
    with patch.object(large, "run_worker", side_effect=worker_fixture(options)) as worker:
        assert large.main(options) == 0
    assert worker.call_count == 13  # control:2 gates+4; screen:2 gates+1; verify:4
    output = tmp_path / "scan"
    report = json.loads((output / "results.json").read_text())
    best = json.loads((output / "best_config.json").read_text())
    assert report["run_status"] == "completed"
    assert report["shortlist"] == ["m128n64k128s2"]
    for modes in best["by_T_and_mode"].values():
        for row in modes.values():
            assert row["fastest_measured_triton"] == "m128n64k128s2"
            assert row["sample_count_per_path"] == 6
            assert row["suggested_path"] == "triton"
    assert best["production_dispatch_applied"] is False
    assert all(t["wall_seconds"] >= 0 for r in report["results"] for t in r["trials"])
    assert len((output / "trials.csv").read_text().splitlines()) == 14
    assert ",screen,8192," in (output / "timings.csv").read_text()


def test_baseline_mode_does_not_tune(tmp_path):
    options = run_options(tmp_path, "--phase", "baseline")
    with patch.object(large, "run_worker", side_effect=worker_fixture(options)) as worker:
        assert large.main(options) == 0
    assert worker.call_count == 6


def test_failed_real_weight_baseline_prevents_search(tmp_path):
    options = run_options(tmp_path)

    def fail(*_):
        raise ValueError("nonzero error")

    with patch.object(large, "run_worker", side_effect=worker_fixture(options, fail)) as worker:
        assert large.main(options) == 1
    assert worker.call_count == 3
    assert json.loads((tmp_path / "scan" / "best_config.json").read_text())["run_status"] == "baseline_failed"


def test_partial_fast_finalist_cannot_enter_any_recommendation(tmp_path):
    options = run_options(tmp_path)

    def fail(candidate, tokens, seed):
        if candidate != large.CURRENT and tokens == 8 and seed == 1:
            raise ValueError("correctness regression")

    with patch.object(large, "run_worker", side_effect=worker_fixture(options, fail)):
        assert large.main(options) == 0
    best = json.loads((tmp_path / "scan" / "best_config.json").read_text())
    assert all(
        row["fastest_measured_triton"] == large.CURRENT.tag
        for modes in best["by_T_and_mode"].values()
        for row in modes.values()
    )


@pytest.mark.parametrize("kind", ["device_error", "timeout"])
def test_device_fault_or_timeout_stops_scan(tmp_path, kind):
    options = run_options(tmp_path)

    def fail(candidate, *_):
        if candidate != large.CURRENT:
            raise large.WorkerFailure("stop", kind)

    with patch.object(large, "run_worker", side_effect=worker_fixture(options, fail)):
        assert large.main(options) == 1
    assert json.loads((tmp_path / "scan" / "results.json").read_text())["run_status"] == kind


def test_interrupt_preserves_completed_control_without_recommending_partial_candidate(tmp_path):
    options = run_options(tmp_path)

    def fail(candidate, *_):
        if candidate != large.CURRENT:
            raise KeyboardInterrupt

    with patch.object(large, "run_worker", side_effect=worker_fixture(options, fail)):
        assert large.main(options) == 130
    best = json.loads((tmp_path / "scan" / "best_config.json").read_text())
    assert best["run_status"] == "interrupted"
    assert best["by_T_and_mode"]["8192"]["eager"]["fastest_measured_triton"] == large.CURRENT.tag


def test_conservative_margin_rejects_a_noisy_median_win():
    args = arguments("--tokens", "8192", "--seeds", "0")
    r = measurement(args, large.CURRENT, 8192, triton=0.2, native=0.3)
    r["timing_samples_ms"]["eager"]["triton_including_quant"][-1] = 0.4
    report = {
        "settings": vars(args),
        "run_status": "completed",
        "results": [
            {
                "tag": large.CURRENT.tag,
                "parameters": {},
                "verification_status": "passed",
                "trials": [
                    {"phase": "verify", "T": 8192, "status": "passed", "report": r},
                ],
            }
        ],
    }
    row = large.build_recommendations(report)["by_T_and_mode"]["8192"]["eager"]
    assert row["median_speedup"] > 1
    assert row["suggested_path"] == "composition"


def test_existing_directory_is_never_overwritten(tmp_path):
    options = run_options(tmp_path)
    (tmp_path / "scan").mkdir()
    with patch.object(large, "run_worker") as worker, pytest.raises(FileExistsError):
        large.main(options)
    worker.assert_not_called()


def test_benchmark_uses_prepared_plan_for_correctness_capture_and_timing():
    path = large.REPO_ROOT / "tools/dsv4_a3_wo/benchmark_tqbmm.py"
    tree = ast.parse(path.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    text = ast.unparse(run)
    assert "PreparedDynamicWoA(" in text
    assert "dynamic_actual = dynamic_fused()" in text
    assert "captured = dynamic_fused()" in text
    assert "'triton_including_quant': dynamic_fused" in text
