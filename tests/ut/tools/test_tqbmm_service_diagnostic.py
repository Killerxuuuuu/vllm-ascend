# SPDX-License-Identifier: Apache-2.0
"""CPU-only short-profile collector, streaming trace and config regression tests."""

import ast
import contextlib
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import diagnose_tqbmm_service as diagnostic
from tools.dsv4_a3_wo import service_trace_analysis as trace


def write_trace(path, events, as_array=False):
    content = json.dumps(
        events if as_array else {"displayTimeUnit": "ms", "traceEvents": events, "metadata": {"end": True}}
    )
    path.write_text(content, encoding="utf-8")
    return path


def metadata():
    return [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Python"}},
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}},
    ]


def event(name, ts, duration, pid=2, tid=10, category=""):
    return {"ph": "X", "name": name, "ts": ts, "dur": duration, "pid": pid, "tid": tid, "cat": category}


@pytest.mark.parametrize("as_array", [False, True])
def test_streaming_tiny_chunks_and_unicode(tmp_path, monkeypatch, as_array):
    monkeypatch.setattr(trace, "CHUNK_SIZE", 7)
    events = [event("多字节事件", 1234567.125, 0.25), *metadata()]
    path = write_trace(tmp_path / "trace.json", events, as_array)
    assert list(trace.iter_events(path)) == events


def test_streaming_gzip(tmp_path):
    events = [event("a", 1, 2)]
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)
    assert list(trace.iter_events(path)) == events


@pytest.mark.parametrize(
    "contents",
    ["", "{}", '{"traceEvents":[', '[{"x":1}', '{"traceEvents":[]}trailing', '{"traceEvents":[],"traceEvents":[]}'],
)
def test_malformed_or_truncated_trace_rejected(tmp_path, contents):
    path = tmp_path / "bad.json"
    path.write_text(contents)
    with pytest.raises(ValueError):
        list(trace.iter_events(path))


def test_large_single_event_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "CHUNK_SIZE", 16)
    monkeypatch.setattr(trace, "MAX_EVENT_CHARS", 64)
    path = write_trace(tmp_path / "huge-event.json", [{"name": "x" * 300}])
    with pytest.raises(ValueError, match="size guard"):
        list(trace.iter_events(path))


def test_interval_union_not_sum_and_actual_gaps():
    result = trace.merged_interval_report([(20, 30, "a"), (25, 45, "b"), (50, 55, "c")])
    assert result["observed_span_us"] == 35
    assert result["union_covered_us"] == 30
    assert result["uncovered_between_tasks_us"] == 5
    assert result["largest_uncovered_intervals"][0]["previous_task"] == "b"


def test_zero_and_no_intervals():
    assert not trace.merged_interval_report([])["available"]
    assert trace.merged_interval_report([(1, 1, "zero")])["union_covered_us"] == 0


def test_metadata_after_events_shape_and_unique_flow(tmp_path):
    host = event("aten::fill_", 10, 5, pid=1, tid=1, category="cpu_op")
    host["args"] = {"Input Dims": [[1, 64]]}
    kernel = event("FillAiCore_Fill", 30, 4)
    events = [
        host,
        kernel,
        {"ph": "s", "name": "torch_to_npu", "id": 88, "pid": 1, "tid": 1, "ts": 10, "cat": "async_npu"},
        {"ph": "f", "name": "torch_to_npu", "id": 88, "pid": 2, "tid": 10, "ts": 30, "cat": "async_npu"},
        *metadata(),
    ]
    result = trace.analyze_trace(write_trace(tmp_path / "trace.json", events))
    assert result["device_focus"]["fill_or_zero_candidate"]["calls"] == 1
    assert result["host_focus_inclusive"]["fill_or_zero_candidate"]["mean_us"] == 5
    assert result["observed_input_shapes"][0]["input_dims"] == "[[1, 64]]"
    pair = result["flow_evidence"]["focus_associations"][0]
    assert pair["host_name"] == "aten::fill_"
    assert pair["host_start_to_device_start"]["mean_us"] == 20
    assert "NOT launch/queue latency" in result["flow_evidence"]["meaning"]


def test_ambiguous_flow_not_guessed(tmp_path):
    events = [
        event("a", 1, 2, 1, 1),
        event("b", 1, 3, 1, 1),
        event("Fill", 10, 1),
        *metadata(),
        {"ph": "s", "name": "torch_to_npu", "id": 7, "pid": 1, "tid": 1, "ts": 1},
        {"ph": "f", "name": "torch_to_npu", "id": 7, "pid": 2, "tid": 10, "ts": 10},
    ]
    result = trace.analyze_trace(write_trace(tmp_path / "trace.json", events))
    assert result["flow_evidence"]["matched_unique_pairs"] == 0
    assert result["flow_evidence"]["unmatched_or_ambiguous_pairs"] == 1


def test_timeline_cap_disables_partial_gap_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "MAX_TIMELINE_EVENTS", 1)
    events = [event("a", 0, 1), event("b", 5, 1), *metadata()]
    result = trace.analyze_trace(write_trace(tmp_path / "trace.json", events))
    assert result["omitted_interval_events"] == 1
    assert not result["device_task_coverage"]["available"]


def test_unknown_lanes_not_treated_as_hardware(tmp_path):
    result = trace.analyze_trace(write_trace(tmp_path / "trace.json", [event("unknown", 0, 50)]))
    assert result["unknown_top"][0]["name"] == "unknown"
    assert not result["device_task_coverage"]["available"]


def test_duration_buckets_are_exclusive_and_fill_kept():
    samples = trace.Durations()
    for value in [0, 1, 2, 7, 50000]:
        samples.add(value)
    result = samples.report()
    assert result["calls"] == 5
    assert sum(result["exclusive_duration_buckets_us"].values()) == 5
    assert result["exclusive_duration_buckets_us"][">10000"] == 1
    assert "task_time_share_pct" not in result


def test_rank_selection_matches_both_dp_and_tp(tmp_path):
    for dp, tp in [(0, 0), (1, 0), (1, 5)]:
        path = tmp_path / "trace" / f"dp{dp}_pp0_tp{tp}_ep0_ascend_pt" / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
        path.parent.mkdir(parents=True)
        path.touch()
    assert "dp1_pp0_tp5" in str(diagnostic.select_rank(tmp_path, 1, 5))
    with pytest.raises(ValueError, match="found 0"):
        diagnostic.select_rank(tmp_path, 0, 5)


def minimal_plan(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"vocab_size": 128}')
    return {
        "model": str(model),
        "model_name": "dsv4-int8",
        "base_url": "http://127.0.0.1:8000",
        "server_args": ["--max-model-len", "4096"],
        "workload": {"input_len": 8},
        "identity": {"python": "python"},
        "environment_hashes": {},
        "plan_sha256": "old",
    }


def test_new_plan_freezes_exact_token_ids_and_does_not_mutate_base(tmp_path, monkeypatch):
    base = minimal_plan(tmp_path)
    monkeypatch.setattr(diagnostic.bench, "runtime_identity", lambda model: {"python": "python"})
    monkeypatch.setattr(diagnostic.bench, "kernel_config", lambda: {"block_shape": [64, 64, 128], "num_stages": 2})
    plan, prompts = diagnostic.make_diagnostic_plan(base, 32)
    again, prompts_again = diagnostic.make_diagnostic_plan(base, 32)
    assert prompts == prompts_again
    assert plan == again
    assert base["plan_sha256"] == "old"
    assert "diagnostic_protocol" not in base
    assert plan["diagnostic_protocol"]["requests_by_c"] == {"1": 2, "64": 64}
    assert all(len(prompt) == 8 for prompt in prompts)
    assert len(prompts) == 64
    assert plan["source_plan_sha256"] == "old"


def test_profile_configuration_is_explicit_and_frontend_ignored(tmp_path):
    result = diagnostic.profile_config(tmp_path)
    assert result["ignore_frontend"]
    assert result["torch_profiler_record_shapes"]
    assert not result["torch_profiler_with_stack"]


@pytest.mark.parametrize("record_shapes", [True, False, None])
def test_actual_profiler_wrapper_forwards_shape_setting(record_shapes):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/profiler/torch_npu_profiler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_create_profiler")
    function.decorator_list = []
    namespace = {
        "ProfilerConfig": object,
        "Any": object,
        "suppress": contextlib.suppress,
        "torch_npu": SimpleNamespace(profiler=Mock()),
        "envs_ascend": SimpleNamespace(MSMONITOR_USE_DAEMON=False),
        "get_ascend_config": lambda: SimpleNamespace(msmonitor_use_daemon=False),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    config = SimpleNamespace(
        profiler="torch", torch_profiler_dir="trace", torch_profiler_with_memory=False, torch_profiler_with_stack=False
    )
    if record_shapes is not None:
        config.torch_profiler_record_shapes = record_shapes
    namespace["_create_profiler"](config, "rank0")
    assert namespace["torch_npu"].profiler.profile.call_args.kwargs["record_shapes"] is bool(record_shapes)


def prepare_collection(tmp_path, monkeypatch):
    folder = tmp_path / "sessions/composition-c1"
    folder.mkdir(parents=True)
    plan = minimal_plan(tmp_path)
    plan["plan_sha256"] = "diagnostic"
    plan["diagnostic_protocol"] = {
        "input_len": 8,
        "output_len": 32,
        "seed": 123,
        "requests_by_c": {"1": 2},
        "warmup_min_requests": 2,
        "request_body": {"ignore_eos": True, "stream": False},
    }
    prompts = [[1] * 8, [2] * 8]
    plan["diagnostic_protocol"]["prompts_sha256"] = diagnostic.bench.digest(prompts)
    diagnostic.bench.write_json(tmp_path / "prompts.json", prompts)
    monkeypatch.setattr(diagnostic.bench, "server_command", lambda plan: ["fixed-command"])
    monkeypatch.setattr(diagnostic.bench, "check_identity", lambda *args: None)
    monkeypatch.setattr(diagnostic.bench, "require_ready", lambda *args: None)
    monkeypatch.setattr(diagnostic.bench, "process_environment", lambda pid: {diagnostic.bench.VARIANT_ENV: "0"})
    monkeypatch.setattr(diagnostic.bench, "npu_snapshot", lambda *args: None)
    diagnostic.bench.write_json(
        folder / "server.json",
        {
            "command": ["fixed-command"],
            "variant": "composition",
            "pid": 123,
            "plan_sha256": "diagnostic",
            "wo_a_switch": "0",
        },
    )
    return plan, folder, SimpleNamespace(study=tmp_path, variant="composition", concurrency=1)


@pytest.mark.parametrize("failure", [None, "start", "request", "stop"])
def test_collection_control_order_single_stop_and_failure_retention(tmp_path, monkeypatch, failure):
    plan, folder, args = prepare_collection(tmp_path, monkeypatch)
    calls = []

    def post(url, body=None, request_id=None, **kwargs):
        if url.endswith("/start_profile"):
            calls.append("start")
            if failure == "start":
                raise ValueError("start failed")
            return {}
        if url.endswith("/stop_profile"):
            calls.append("stop")
            if failure == "stop":
                raise ValueError("stop failed")
            return {}
        calls.append(request_id.split("-")[0])
        assert isinstance(body["prompt"], list)
        if failure == "request" and request_id.startswith("capture"):
            return {"usage": {"prompt_tokens": 8, "completion_tokens": 1}, "choices": [{}]}
        return {"usage": {"prompt_tokens": 8, "completion_tokens": 32}, "choices": [{}]}

    monkeypatch.setattr(diagnostic, "post", post)
    if failure:
        with pytest.raises(ValueError):
            diagnostic.collect(args, plan)
    else:
        diagnostic.collect(args, plan)
    result = diagnostic.bench.read_json(folder / "capture.json")
    assert result["status"] == ("failed" if failure else "completed")
    assert calls.count("start") == 1
    assert calls.count("stop") == 1
    assert calls[:2] == ["warmup", "warmup"]
    assert calls[-1] == "stop"
    assert (folder / "capture-once.lock").exists()
    if not failure:
        assert len(result["requests"]) == 2


def test_profile_control_accepts_plain_text_success_response(monkeypatch):
    response = Mock()
    response.read.return_value = b"Profiler started"
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(
        diagnostic.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=lambda *a, **kw: response)
    )
    assert diagnostic.post("http://127.0.0.1/start_profile", expect_json=False) == {}


def test_csv_fill_histogram_not_hidden_by_top_n(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text("Name,Duration(us)\nFill,1\nFill,31000\nOther,999999\n", encoding="utf-8")
    result = diagnostic.focused_csv_durations(path)
    assert result["fill_or_zero_candidate"]["calls"] == 2
    assert result["fill_or_zero_candidate"]["exclusive_duration_buckets_us"][">10000"] == 1


@pytest.mark.parametrize("broken_trace", [False, True])
def test_offline_analysis_outputs_small_report_and_preserves_trace_failure(tmp_path, monkeypatch, broken_trace):
    plan, folder, args = prepare_collection(tmp_path, monkeypatch)
    args.dp, args.tp, args.output = 0, 0, None
    parsed = folder / "trace/dp0_pp0_tp0_ep0_ascend_pt/ASCEND_PROFILER_OUTPUT"
    parsed.mkdir(parents=True)
    (parsed / "kernel_details.csv").write_text("Name,Duration(us)\nFill,31000\nFill,1\n", encoding="utf-8")
    (folder / "server.log").write_text("Warning: The collected data has been lost\nReplaying aclgraph\n")
    diagnostic.bench.write_json(
        folder / "capture.json",
        {
            "status": "completed",
            "plan_sha256": "diagnostic",
            "expected_requests": 2,
            "expected_input_tokens": 16,
            "expected_output_tokens": 64,
            "requests": [{"prompt_tokens": 8, "completion_tokens": 32}] * 2,
        },
    )
    if broken_trace:
        (parsed / "trace_view.json").write_text('{"traceEvents":[')
    else:
        write_trace(parsed / "trace_view.json", [event("Fill", 1, 31000), *metadata()])
    diagnostic.analyze(args, plan)
    output = folder / "bottleneck-dp0-tp0.json"
    report = diagnostic.bench.read_json(output)
    assert output.stat().st_size <= diagnostic.MAX_REPORT_BYTES
    assert report["device_focus_duration_histograms"]["fill_or_zero_candidate"]["calls"] == 2
    assert report["actual_request_totals"]["output_tokens"] == 64
    assert "collected data has been lost" in report["server_log_evidence"]
    if broken_trace:
        assert report["trace"] is None
        assert any("incomplete" in warning for warning in report["warnings"])
    else:
        assert report["trace"]["device_focus"]["fill_or_zero_candidate"]["calls"] == 1


def test_output_size_limit_trims_only_detail_lists():
    report = {
        "device_csv": {"top_operators_by_sum": [{"name": "x" * 180}] * 400, "focus_shapes_by_sum": {}},
        "trace": None,
        "lists_trimmed_for_size": False,
    }
    payload = diagnostic.bounded_payload(report)
    assert len(payload.encode()) <= diagnostic.MAX_REPORT_BYTES
    assert report["lists_trimmed_for_size"]


@pytest.mark.parametrize("value", [0, 1, 129])
def test_overlong_diagnostic_capture_rejected(value):
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["init", "--study", "new", "--base-study", "old", "--output-len", str(value)])
