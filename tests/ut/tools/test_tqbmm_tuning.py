# SPDX-License-Identifier: Apache-2.0
"""Standalone tuning/measurement control tests; no torch, Triton or NPU.

Run with --noconftest. These do not validate NPU compilation/performance.
"""

import ast
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.dsv4_a3_wo.tune_tqbmm import (
    Candidate,
    WorkerFailure,
    best_configuration,
    candidates,
    extract_report,
    main,
    rankings,
    run_worker,
    save_reports,
    search_candidates,
    validate_exact_report,
)

BENCHMARK_PATH = Path(__file__).resolve().parents[3] / "tools/dsv4_a3_wo/benchmark_tqbmm.py"
REPO_ROOT = BENCHMARK_PATH.parents[2]


def search_args(**overrides):
    values = {
        "search_space": "balanced",
        "block_m": None,
        "block_n": None,
        "block_k": None,
        "stages": None,
        "candidate": None,
        "extended": False,
        "ub_budget_kib": 192,
        "transpose_weight_tile": False,
        "tile_loads": None,
    }
    return SimpleNamespace(**(values | overrides))


def exact_report():
    metrics = {"max_abs": 0.0, "nrmse": 0.0, "cosine": 0.99999994}
    timing = {
        "triton_prequantized": 0.1,
        "qbmm_composition_prequantized": 0.08,
        "prequantized_speedup": 0.8,
        "triton_including_quant": 0.15,
        "qbmm_composition_including_quant": 0.12,
        "including_quant_speedup": 0.8,
    }
    return {
        "strict_correctness": True,
        "same_int8_inputs": metrics.copy(),
        "including_dynamic_quant": metrics.copy(),
        "graph_replay_with_changed_input": "passed",
        "timing_ms": timing.copy(),
        "graph_timing_ms": timing.copy(),
    }


def test_candidate_plan_preserves_baseline_and_screens_large_tiles():
    plan = candidates(extended=True, transpose_weight_tile=True)
    assert plan[0] == Candidate(16, 32, 32, 1)
    assert len({choice.tag for choice in plan}) == len(plan)
    assert all(choice.block_n <= 64 and choice.block_k <= 64 for choice in plan)
    assert all(choice.estimated_ub_bytes <= 192 * 1024 for choice in plan)
    assert Candidate(64, 32, 32, 1) in plan
    assert Candidate(16, 32, 32, 2) in plan
    assert Candidate(64, 32, 32, 2) not in plan
    assert candidates(budget_kib=1) == []


def test_experimental_tile_load_is_explicit():
    assert all(not choice.transpose_weight_tile for choice in candidates())
    assert "--transpose-weight-tile" not in Candidate(16, 32, 32, 1).options()
    assert "--transpose-weight-tile" in Candidate(16, 32, 32, 1, True).options()


def test_balanced_scan_includes_measured_s2_despite_heuristic_budget():
    plan = search_candidates(search_args())
    measured = Candidate(32, 32, 64, 2)
    assert plan[:2] == [Candidate(16, 32, 32, 1), measured]
    assert measured.estimated_ub_bytes > 192 * 1024
    assert len(plan) == 18
    assert all(not choice.transpose_weight_tile for choice in plan)


def test_grid_scans_every_unique_axis_combination_and_keeps_control():
    plan = search_candidates(
        search_args(search_space="grid", block_m=[32, 32], block_n=[16, 32], block_k=[64], stages=[1, 2])
    )
    expected = {Candidate(32, n, 64, stage) for n in (16, 32) for stage in (1, 2)}
    assert set(plan) == expected | {Candidate(16, 32, 32, 1)}
    assert len(plan) == 5


@pytest.mark.parametrize("mode,multiplier", [("direct", 1), ("nk", 1), ("both", 2)])
def test_explicit_candidates_and_tile_load_modes(mode, multiplier):
    plan = search_candidates(search_args(candidate=[[32, 32, 64, 2]], tile_loads=mode))
    assert plan[0] == Candidate(16, 32, 32, 1)
    assert len(plan) == 1 + multiplier
    if mode == "nk":
        assert plan[1].transpose_weight_tile


def test_extended_128_k_trials_are_explicit_and_not_128_n_by_128_k():
    regular = search_candidates(search_args())
    extended = search_candidates(search_args(extended=True))
    assert all(choice.block_k <= 64 for choice in regular)
    assert Candidate(32, 16, 128, 1) in extended
    assert Candidate(16, 32, 128, 2) in extended
    assert all(choice.block_n < 128 or choice.block_k < 128 for choice in extended)


@pytest.mark.parametrize(
    "values", [(8, 32, 64, 1), (24, 32, 64, 1), (32, 32, 16, 1), (32, 32, 64, 3), (32, 128, 128, 1), (256, 32, 64, 1)]
)
def test_invalid_or_historical_overflow_tiles_rejected_before_launch(values):
    with pytest.raises(ValueError):
        search_candidates(search_args(candidate=[values]))


def test_grid_axes_not_silently_ignored_in_other_search_modes():
    with pytest.raises(ValueError, match="require"):
        search_candidates(search_args(block_m=[32]))
    with pytest.raises(ValueError, match="not both"):
        search_candidates(search_args(search_space="grid", block_m=[32], candidate=[[32, 32, 64, 2]]))


@pytest.mark.parametrize(
    "options",
    [
        ["--search-space", "balanced", "--max-candidates", "2"],
        ["--timing-mode", "eager", "--selection-mode", "graph"],
        ["--tp-size", "8", "--tp-rank", "8"],
        ["--layer", "-1"],
    ],
)
def test_invalid_scan_arguments_do_not_spawn_workers(options):
    with patch("tools.dsv4_a3_wo.tune_tqbmm.subprocess.run") as worker, pytest.raises(SystemExit):
        main(["--quant-model", "unused", "--dry-run", *options])
    worker.assert_not_called()


def test_root_launcher_dry_run_is_cpu_only_and_includes_new_best():
    process = subprocess.run(
        [sys.executable, str(REPO_ROOT / "autotune_tqbmm.py"), "--quant-model", "unused", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(process.stdout)
    assert len(plan) == 18
    assert plan[1]["tag"] == "m32n32k64s2"
    assert "torch" not in process.stderr


def test_extract_report_skips_import_logs_and_case_json():
    output = 'INFO {not JSON}\n{"K":32}\n' + json.dumps({"shape_T_G_K_N": [16, 1, 4096, 1024]})
    assert extract_report(output, "shape_T_G_K_N")["shape_T_G_K_N"][0] == 16
    with pytest.raises(ValueError):
        extract_report("Traceback: compile failed", "shape_T_G_K_N")


@pytest.mark.parametrize("mutation", ["strict", "max_abs", "nrmse", "nan", "cosine", "graph", "timing", "missing"])
def test_rejects_inexact_or_incomplete_results(mutation):
    report = exact_report()
    if mutation == "strict":
        report["strict_correctness"] = False
    elif mutation in ("max_abs", "nrmse"):
        report["same_int8_inputs"][mutation] = 1e-9
    elif mutation == "nan":
        report["including_dynamic_quant"]["max_abs"] = float("nan")
    elif mutation == "cosine":
        report["same_int8_inputs"]["cosine"] = 0.99
    elif mutation == "graph":
        del report["graph_replay_with_changed_input"]
    elif mutation == "timing":
        report["graph_timing_ms"]["triton_prequantized"] = -1
    else:
        del report["graph_timing_ms"]
    with pytest.raises((ValueError, KeyError)):
        validate_exact_report(report)


@pytest.mark.parametrize("mode", ["eager", "graph", "both"])
def test_accepts_exact_metrics_and_selected_timing_modes(mode):
    report = exact_report()
    if mode == "eager":
        del report["graph_timing_ms"]
    elif mode == "graph":
        del report["timing_ms"]
    validate_exact_report(report, mode)


def test_rankings_exclude_rejected_partial_candidates():
    valid = exact_report()
    invalid = deepcopy(valid)
    invalid["timing_ms"]["triton_prequantized"] = 0.001
    results = [
        {"tag": "safe", "status": "passed", "reports": {"16": valid}},
        {"tag": "wrong", "status": "rejected", "reports": {"16": invalid}},
    ]
    assert rankings(results, [16])["16"]["eager"]["candidate"] == "safe"


def test_single_best_uses_all_shapes_not_only_the_fastest_individual_case():
    def result(tag, durations):
        reports = {}
        for tokens, duration in zip((1, 512), durations):
            report = exact_report()
            report["graph_timing_ms"]["triton_prequantized"] = duration
            reports[str(tokens)] = report
        return {"tag": tag, "status": "passed", "reports": reports}

    results = [result("great_at_one_token", [0.01, 0.4]), result("balanced", [0.02, 0.1])]
    winner = best_configuration(results, [1, 512])
    assert winner["candidate"] == "balanced"
    assert winner["score_geometric_mean_ms"] == pytest.approx((0.02 * 0.1) ** 0.5)
    assert rankings(results, [1, 512])["1"]["graph"]["candidate"] == "great_at_one_token"


def test_single_best_obeys_objective_and_rejects_pending_partial_and_nonfinite():
    fast_prequant = exact_report()
    fast_dynamic = exact_report()
    fast_dynamic["graph_timing_ms"]["triton_prequantized"] = 0.2
    fast_dynamic["graph_timing_ms"]["triton_including_quant"] = 0.12
    results = [
        {"tag": "prequant", "status": "passed", "reports": {"16": fast_prequant}},
        {"tag": "dynamic", "status": "passed", "reports": {"16": fast_dynamic}},
        {"tag": "pending", "status": "pending", "reports": {"16": fast_prequant}},
        {"tag": "partial", "status": "rejected", "reports": {"16": fast_prequant}},
    ]
    assert best_configuration(results, [16])["candidate"] == "prequant"
    assert best_configuration(results, [16], objective="including_quant")["candidate"] == "dynamic"
    assert best_configuration(results, [16, 512]) is None
    fast_prequant["graph_timing_ms"]["triton_prequantized"] = float("nan")
    assert best_configuration(results, [16])["candidate"] == "dynamic"


@pytest.mark.parametrize("returncode", [0, 1])
def test_worker_preserves_log_and_rejects_compiler_failure(tmp_path, returncode):
    output = json.dumps({"layout_check": "passed"})
    process = SimpleNamespace(returncode=returncode, stdout=output)
    log = tmp_path / "worker.log"
    with patch("tools.dsv4_a3_wo.tune_tqbmm.subprocess.run", return_value=process) as run:
        if returncode:
            with pytest.raises(RuntimeError):
                run_worker(["python", "-m", "worker"], log, 10, "layout_check")
        else:
            assert run_worker(["python", "-m", "worker"], log, 10, "layout_check")["layout_check"] == "passed"
    assert log.read_text() == output
    assert "shell" not in run.call_args.kwargs


def test_worker_timeout_is_not_silently_timed(tmp_path):
    with (
        patch(
            "tools.dsv4_a3_wo.tune_tqbmm.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["worker"], 1, output=b"partial IR"),
        ),
        pytest.raises(RuntimeError, match="timed out"),
    ):
        run_worker(["worker"], tmp_path / "timeout.log", 1, "shape_T_G_K_N")
    assert "partial IR" in (tmp_path / "timeout.log").read_text()


def test_dry_run_does_not_create_output_or_spawn_workers(tmp_path, capsys):
    output = tmp_path / "not-created"
    with patch("tools.dsv4_a3_wo.tune_tqbmm.subprocess.run") as worker:
        assert main(["--quant-model", "not-required-for-plan", "--output", str(output), "--dry-run"]) == 0
    assert not output.exists()
    worker.assert_not_called()
    assert json.loads(capsys.readouterr().out)[0]["num_stages"] == 1


def test_results_include_rejected_status_and_separate_modes(tmp_path):
    report = {
        "results": [
            {"tag": "safe", "status": "passed", "reports": {"16": exact_report()}},
            {"tag": "bad", "status": "rejected", "reports": {}},
        ]
    }
    save_reports(tmp_path, report)
    assert json.loads((tmp_path / "results.json").read_text())["results"][1]["status"] == "rejected"
    csv = (tmp_path / "timings.csv").read_text()
    assert ",eager," in csv and ",graph," in csv


def test_failed_baseline_stops_search_and_saves_reason(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    output = tmp_path / "results"
    with patch("tools.dsv4_a3_wo.tune_tqbmm.run_worker", side_effect=RuntimeError("baseline compile failed")) as worker:
        assert main(["--quant-model", str(model), "--output", str(output)]) == 1
    assert worker.call_count == 1
    saved = json.loads((output / "results.json").read_text())
    assert saved["results"][0]["status"] == "rejected"
    assert "baseline compile failed" in saved["results"][0]["error"]


def test_partial_fast_candidate_does_not_win_when_later_shape_fails(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    output = tmp_path / "results"

    def worker(command, log, timeout, expected_key):
        if expected_key in ("layout_check", "k_loop_check"):
            return {expected_key: "passed"}
        assert "--strict" in command and "--graph" in command
        count = int(command[command.index("--tokens") + 1])
        block_m = int(command[command.index("--block-shape") + 1])
        report = exact_report()
        if block_m == 32:
            if count == 512:
                report["same_int8_inputs"]["max_abs"] = 0.015625
            else:
                report["timing_ms"]["triton_prequantized"] = 0.001
        report["shape_T_G_K_N"] = [count, 1, 4096, 1024]
        return report

    with patch("tools.dsv4_a3_wo.tune_tqbmm.run_worker", side_effect=worker):
        assert main(["--quant-model", str(model), "--output", str(output)]) == 0
    saved = json.loads((output / "results.json").read_text())
    rejected = next(result for result in saved["results"] if result["tag"] == "m32n32k32s1")
    assert rejected["status"] == "rejected"
    assert "512" not in rejected["reports"]
    assert all(value["eager"]["candidate"] != rejected["tag"] for value in saved["best_measured"].values())


def successful_worker(command, log, timeout, expected_key):
    if expected_key in ("layout_check", "k_loop_check"):
        measurements = {expected_key: "passed"}
    else:
        measurements = exact_report()
        tokens = int(command[command.index("--tokens") + 1])
        measurements["shape_T_G_K_N"] = [tokens, 1, 4096, 1024]
        stages = int(command[command.index("--num-stages") + 1])
        if stages == 2:
            measurements["graph_timing_ms"]["triton_prequantized"] = 0.07
    log.write_text(json.dumps(measurements), encoding="utf-8")
    return measurements


def scan_options(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    return [
        "--quant-model",
        str(model),
        "--output",
        str(tmp_path / "scan"),
        "--candidate",
        "32",
        "32",
        "64",
        "2",
        "--tokens",
        "1",
        "512",
    ]


def test_complete_scan_records_every_trial_wall_time_and_reusable_winner(tmp_path):
    options = scan_options(tmp_path)
    with patch("tools.dsv4_a3_wo.tune_tqbmm.run_worker", side_effect=successful_worker) as worker:
        assert main(options) == 0
    assert worker.call_count == 8
    output = tmp_path / "scan"
    report = json.loads((output / "results.json").read_text())
    assert report["run_status"] == "completed"
    assert len(report["results"]) == 2
    assert all(result["status"] == "passed" for result in report["results"])
    assert all(result["wall_seconds"] >= 0 for result in report["results"])
    assert all(trial["wall_seconds"] >= 0 for result in report["results"] for trial in result["trials"])
    best = json.loads((output / "best_config.json").read_text())
    assert best["overall_best"]["candidate"] == "m32n32k64s2"
    assert best["overall_best"]["parameters"] == {
        "block_m": 32,
        "block_n": 32,
        "block_k": 64,
        "num_stages": 2,
        "transpose_weight_tile": False,
    }
    assert best["overall_best"]["benchmark_options"] == ["--block-shape", "32", "32", "64", "--num-stages", "2"]
    assert len((output / "candidates.csv").read_text().splitlines()) == 3
    assert len((output / "trials.csv").read_text().splitlines()) == 9
    assert not list(output.glob("*.tmp"))


def test_interrupt_keeps_completed_best_and_marks_remaining_work(tmp_path):
    options = scan_options(tmp_path)

    def worker(command, log, timeout, expected_key):
        if command[command.index("--num-stages") + 1] == "2":
            raise KeyboardInterrupt
        return successful_worker(command, log, timeout, expected_key)

    with patch("tools.dsv4_a3_wo.tune_tqbmm.run_worker", side_effect=worker):
        assert main(options) == 130
    saved = json.loads((tmp_path / "scan" / "results.json").read_text())
    assert saved["run_status"] == "interrupted"
    assert saved["results"][1]["status"] == "interrupted"
    assert saved["results"][1]["trials"][0]["status"] == "interrupted"
    best = json.loads((tmp_path / "scan" / "best_config.json").read_text())
    assert best["overall_best"]["candidate"] == "m16n32k32s1"


def test_device_error_stops_search_without_timing_a_fallback(tmp_path):
    options = scan_options(tmp_path)
    options.extend(["--candidate", "64", "32", "32", "1"])

    def worker(command, log, timeout, expected_key):
        if command[command.index("--num-stages") + 1] == "2":
            raise WorkerFailure("device lost", "device_error")
        return successful_worker(command, log, timeout, expected_key)

    with patch("tools.dsv4_a3_wo.tune_tqbmm.run_worker", side_effect=worker):
        assert main(options) == 1
    saved = json.loads((tmp_path / "scan" / "results.json").read_text())
    assert saved["run_status"] == "device_error"
    assert saved["results"][1]["status"] == "rejected"
    assert saved["results"][2]["status"] == "pending"


def test_existing_output_is_never_overwritten(tmp_path):
    options = scan_options(tmp_path)
    output = tmp_path / "scan"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user output")
    with patch("tools.dsv4_a3_wo.tune_tqbmm.run_worker") as worker, pytest.raises(FileExistsError):
        main(options)
    worker.assert_not_called()
    assert sentinel.read_text() == "user output"


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_event_timing_denominator_and_capture_outside_events(mode):
    events = []

    class Event:
        def __init__(self, **kwargs):
            self.name = f"event{sum(item.startswith('new_event') for item in events)}"
            events.append("new_" + self.name)

        def record(self):
            events.append(self.name)

        def synchronize(self):
            events.append("end_sync")

        def elapsed_time(self, end):
            return 20.0

    class Graph:
        def __enter__(self):
            events.append("capture_start")

        def __exit__(self, *args):
            events.append("capture_end")

        def replay(self):
            events.append("replay")

    npu = SimpleNamespace(
        synchronize=lambda: events.append("sync"), Event=Event, NPUGraph=Graph, graph=lambda graph: graph
    )
    tree = ast.parse(BENCHMARK_PATH.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "elapsed_ms")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": SimpleNamespace(npu=npu), "DEFAULT_GRAPH_BATCH_SIZE": 16}
    exec(compile(module, str(BENCHMARK_PATH), "exec"), namespace)
    duration = namespace["elapsed_ms"](lambda: events.append("call"), 2, 17, timing_mode=mode, graph_batch_size=4)
    if mode == "eager":
        assert duration == 20.0 / 17
        assert events.count("call") == 19
        assert "capture_start" not in events
    else:
        # ceil(17/4)=5 timed replays, 4 calls each, plus one untimed replay.
        assert duration == 20.0 / 20
        assert events.count("call") == 6
        assert events.count("replay") == 6
        assert events.index("capture_end") < events.index("event0")
