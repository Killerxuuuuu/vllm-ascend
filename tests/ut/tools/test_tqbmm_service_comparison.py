# SPDX-License-Identifier: Apache-2.0
"""CPU-only orchestrator tests, no NPU/vLLM imports; use --noconftest."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import compare_tqbmm_service as comparison


def make_study(tmp_path, monkeypatch, extra=None):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(comparison, "runtime_identity", lambda model: {"python": "python", "model": str(model)})
    args = comparison.parse_args(
        [
            "init",
            "--study",
            str(tmp_path / "study"),
            "--model",
            str(model),
            "--concurrencies",
            "1",
            "8",
            "--repeats",
            "2",
            "--num-prompts",
            "8",
            *(extra or []),
        ]
    )
    plan = comparison.make_plan(args)
    args.study.mkdir()
    comparison.write_json(args.study / "plan.json", plan)
    return args.study, plan


def good_result(plan, count):
    work = plan["workload"]
    return {
        "completed": count,
        "failed": 0,
        "total_input_tokens": count * work["input_len"],
        "total_output_tokens": count * work["output_len"],
        **dict.fromkeys(comparison.METRICS, 10.0),
    }


def prepare_session(study, plan, variant="composition", name="a1"):
    folder = study / "sessions" / name
    folder.mkdir(parents=True)
    manifest = {
        "session": name,
        "variant": variant,
        "pid": 123,
        "process_starttime": "12345",
        "plan_sha256": plan["plan_sha256"],
        "wo_a_switch": "1" if variant == "triton" else "0",
    }
    comparison.write_json(folder / "server.json", manifest)
    (folder / "server.log").write_text("Capturing CUDA graphs (decode, FULL): 100%\n", encoding="utf-8")
    return folder


def fake_runner(monkeypatch, plan, bad=False):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--result-dir") + 1]) / command[command.index("--result-filename") + 1]
        count = int(command[command.index("--num-prompts") + 1])
        result = good_result(plan, count)
        if bad and "warmup" not in output.name:
            result["total_output_tokens"] -= 1
        comparison.write_json(output, result)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(comparison, "require_ready", lambda *args: None)
    monkeypatch.setattr(comparison, "check_identity", lambda *args: None)
    monkeypatch.setattr(comparison, "npu_snapshot", lambda *args: None)
    monkeypatch.setattr(comparison, "process_environment", lambda pid: comparison.child_env("composition"))
    monkeypatch.setattr(comparison.subprocess, "run", run)
    return calls


def test_common_plan_has_no_profiler_prefix_cache_or_eager(tmp_path, monkeypatch):
    study, plan = make_study(tmp_path, monkeypatch)
    assert comparison.load_plan(study) == plan
    arguments = plan["server_args"]
    assert "--no-enable-prefix-caching" in arguments
    assert "--enforce-eager" not in arguments
    assert "--profiler-config" not in arguments
    assert "--speculative-config" not in arguments
    compilation = json.loads(arguments[arguments.index("--compilation-config") + 1])
    assert compilation == {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [8, 16, 24, 32]}
    assert plan["kernel_config"] == {"block_shape": [64, 64, 128], "num_stages": 2}


def test_two_child_environments_only_differ_in_existing_switch(monkeypatch):
    for name in comparison.PROFILE_ENV:
        monkeypatch.setenv(name, "enabled")
    a = comparison.child_env("composition")
    b = comparison.child_env("triton")
    assert {key for key in a if a[key] != b.get(key)} == {comparison.VARIANT_ENV}
    assert comparison.selected_env(a) == comparison.selected_env(b)
    assert all(name not in a for name in comparison.PROFILE_ENV)
    assert a["VLLM_LOGGING_LEVEL"] == "INFO"
    assert a["PYTHONPATH"].split(comparison.os.pathsep)[0] == str(comparison.REPO)


def test_secrets_are_not_stored_in_environment_manifest():
    value = "do-not-export-secret"
    hashed = comparison.selected_env({"VLLM_SECRET": value})
    assert value not in json.dumps(hashed)


def test_shared_workload_commands_reproduce_seed_and_token_lengths(tmp_path, monkeypatch):
    _, plan = make_study(tmp_path, monkeypatch)
    path = tmp_path / "c64-r1.json"
    command, count = comparison.bench_command(plan, 64, 0, path)
    assert count == 256
    assert command[command.index("--random-output-len") + 1] == "256"
    assert command[command.index("--seed") + 1] == "123"
    assert command[command.index("--temperature") + 1] == "0"
    assert "--ignore-eos" in command
    assert "--profile" not in command
    assert command == comparison.bench_command(plan, 64, 0, path)[0]
    warmup, warmup_count = comparison.bench_command(plan, 64, 0, path, warmup=True)
    assert warmup_count == 128
    assert warmup[warmup.index("--random-output-len") + 1] == "256"


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("completed", 7),
        ("failed", 1),
        ("total_input_tokens", 0),
        ("total_output_tokens", 0),
        ("mean_ttft_ms", float("nan")),
        ("duration", 0),
        ("output_throughput", 0),
        ("mean_tpot_ms", -1),
    ],
)
def test_strict_result_rejection(tmp_path, monkeypatch, key, bad):
    _, plan = make_study(tmp_path, monkeypatch)
    result = good_result(plan, 8)
    result[key] = bad
    with pytest.raises(ValueError):
        comparison.validate_result(result, plan["workload"], 8)


def test_edited_plan_is_rejected(tmp_path, monkeypatch):
    study, plan = make_study(tmp_path, monkeypatch)
    plan["workload"]["repeats"] += 1
    (study / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="edited"):
        comparison.load_plan(study)


def test_check_identity_and_env_rejection(tmp_path, monkeypatch):
    _, plan = make_study(tmp_path, monkeypatch)
    environment = comparison.child_env("triton")
    comparison.check_identity(plan, environment)
    environment["ASCEND_VISIBLE_DEVICES"] = "0"
    with pytest.raises(ValueError, match="environment"):
        comparison.check_identity(plan, environment)
    monkeypatch.setattr(comparison, "runtime_identity", lambda model: {})
    with pytest.raises(ValueError, match="changed"):
        comparison.check_identity(plan, comparison.child_env("triton"))


def test_complete_run_warmups_rotation_and_manifest(tmp_path, monkeypatch):
    study, plan = make_study(tmp_path, monkeypatch)
    folder = prepare_session(study, plan)
    calls = fake_runner(monkeypatch, plan)
    args = SimpleNamespace(study=study, session="a1", timeout=5)
    comparison.run_bench(args, plan)
    report = comparison.read_json(folder / "validated.json")
    assert report["status"] == "completed"
    assert [(r["concurrency"], r["repeat"]) for r in report["records"]] == [(1, 1), (8, 1), (8, 2), (1, 2)]
    assert len(calls) == 8  # Four identical-protocol warmups, four measured rounds.
    with pytest.raises(FileExistsError):
        comparison.run_bench(args, plan)


def test_bad_run_keeps_raw_data_but_has_no_validated_report(tmp_path, monkeypatch):
    study, plan = make_study(tmp_path, monkeypatch)
    folder = prepare_session(study, plan)
    fake_runner(monkeypatch, plan, bad=True)
    with pytest.raises(ValueError, match="total_output_tokens"):
        comparison.run_bench(SimpleNamespace(study=study, session="a1", timeout=5), plan)
    assert not (folder / "validated.json").exists()
    assert (folder / "bench/c1-r1.json").exists()
    assert comparison.read_json(folder / "bench/failure.json")["rankable"] is False


def test_missing_capture_prevents_client_requests(tmp_path, monkeypatch):
    study, plan = make_study(tmp_path, monkeypatch)
    folder = prepare_session(study, plan)
    (folder / "server.log").write_text("No capture\n")
    calls = fake_runner(monkeypatch, plan)
    with pytest.raises(ValueError, match="No Graph capture"):
        comparison.run_bench(SimpleNamespace(study=study, session="a1", timeout=5), plan)
    assert not calls


def test_pid_reuse_rejected(monkeypatch):
    monkeypatch.setattr(comparison, "process_token", lambda pid: "new")
    with pytest.raises(ValueError, match="reused"):
        comparison.check_server({"pid": 123, "process_starttime": "old"})


def test_serving_does_not_kill_existing_service(tmp_path, monkeypatch):
    study, plan = make_study(tmp_path, monkeypatch)
    monkeypatch.setattr(comparison.sys, "platform", "linux")
    monkeypatch.setattr(comparison, "check_identity", lambda *args: None)
    monkeypatch.setattr(comparison, "ensure_free_port", Mock(side_effect=OSError("already in use")))
    spawn = Mock()
    monkeypatch.setattr(comparison.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="already in use"):
        comparison.serve(SimpleNamespace(study=study, session="a1", variant="composition"), plan)
    spawn.assert_not_called()


def test_compare_separate_baseline_sessions_and_deltas(tmp_path, monkeypatch):
    study, plan = make_study(tmp_path, monkeypatch)
    for name, variant, speed in [("a1", "composition", 10), ("b1", "triton", 12), ("a2", "composition", 11)]:
        folder = prepare_session(study, plan, variant, name)
        records = []
        for concurrency in plan["workload"]["concurrencies"]:
            for repeat in (1, 2):
                result = good_result(plan, 8)
                result["output_throughput"] = speed
                records.append({"concurrency": concurrency, "repeat": repeat, **result})
        comparison.write_json(
            folder / "validated.json",
            {
                "status": "completed",
                "session": name,
                "variant": variant,
                "plan_sha256": plan["plan_sha256"],
                "records": records,
            },
        )
    comparison.compare(SimpleNamespace(study=study, sessions=["a1", "b1", "a2"]), plan)
    report = comparison.read_json(study / "comparison-a1-b1-a2.json")
    pairs = report["by_concurrency"][0]["comparisons"]
    assert len(pairs) == 2
    assert pairs[0]["change_pct"]["output_throughput"] == pytest.approx(20)
    assert (study / "comparison-a1-b1-a2.csv").is_file()


@pytest.mark.parametrize("extra", [["--enforce-eager"], ["--profiler-config", "{}"], ["--additional-config", "[]"]])
def test_fixed_server_args_cannot_be_overridden(tmp_path, monkeypatch, extra):
    with pytest.raises(ValueError, match="Only"):
        make_study(tmp_path, monkeypatch, ["--server-extra-json", json.dumps(extra)])


@pytest.mark.parametrize(
    "options",
    [
        ["--repeats", "0"],
        ["--waves", "0"],
        ["--concurrencies", "1", "1"],
        ["--capture-sizes", "1", "8"],
        ["--output-len", "1"],
        ["--memory-utilization", "1"],
    ],
)
def test_invalid_init_options(options):
    with pytest.raises(SystemExit):
        comparison.parse_args(["init", "--study", "unused", "--model", "unused", *options])


@pytest.mark.parametrize("name", ["../escape", "/tmp/a", "a b", "a" * 41])
def test_invalid_session_names(name):
    with pytest.raises(SystemExit):
        comparison.parse_args(["run", "--study", "unused", "--session", name])
