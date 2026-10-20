# SPDX-License-Identifier: Apache-2.0
"""Standalone tuning/measurement control tests; no torch, Triton or NPU.

Run with --noconftest. These do not validate NPU compilation/performance.
"""

import ast
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.dsv4_a3_wo.tune_tqbmm import (
    Candidate,
    candidates,
    extract_report,
    main,
    rankings,
    run_worker,
    save_reports,
    validate_exact_report,
)

BENCHMARK_PATH = Path(__file__).resolve().parents[3] / "tools/dsv4_a3_wo/benchmark_tqbmm.py"


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
