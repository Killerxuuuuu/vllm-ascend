# SPDX-License-Identifier: Apache-2.0
"""CPU-only orchestration tests. Run with --noconftest; no NPU validation."""

import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import profile_tqbmm as profiling


def model_args(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "profiles"
    return ["--quant-model", str(model), "--output", str(output)], output


def create_analysis(case_dir):
    analysis = case_dir / "profiler" / "ASCEND_PROFILER_OUTPUT"
    analysis.mkdir(parents=True)
    (analysis / "kernel_details.csv").write_text("Name,Duration(us)\nkernel,60\n", encoding="utf-8")
    (analysis / "trace_view.json").write_text('{"traceEvents": []}', encoding="utf-8")
    (analysis / "op_summary_1.csv").write_text("Op Name,Task Duration(us)\nkernel,60\n", encoding="utf-8")


def fake_worker(command, **kwargs):
    name = command[command.index("--worker-case") + 1]
    output = Path(command[command.index("--output") + 1])
    case_dir = output / "raw" / name
    create_analysis(case_dir)
    batch = int(command[command.index("--graph-batch-size") + 1])
    replays = int(command[command.index("--replays") + 1])
    profiling.write_manifest(
        case_dir / "capture.json",
        {"case": name, "strict_correctness": True, "operator_calls": batch * replays},
    )
    kwargs["stdout"].write(f"worker log: {name}\n")
    return SimpleNamespace(returncode=0)


def test_defaults_and_equal_call_counts():
    args = profiling.parse_args(["--quant-model", "unused"])
    assert args.block_shape == [64, 64, 128]
    assert args.num_stages == 2
    assert not args.transpose_weight_tile
    plan = profiling.case_plan(args)
    assert [case["case"] for case in plan] == list(profiling.PROFILE_CASES)
    assert all(case["operator_calls"] == 64 for case in plan)
    assert [case["graph_replays"] for case in plan] == [0, 4, 0, 4]


@pytest.mark.parametrize(
    "extra",
    [
        ["--tokens", "0"],
        ["--tp-size", "0"],
        ["--warmup", "-1"],
        ["--replays", "0"],
        ["--timeout", "0"],
        ["--graph-batch-size", "0"],
        ["--layer", "-1"],
        ["--tp-rank", "8"],
        ["--tp-rank", "-1"],
        ["--block-shape", "24", "32", "64"],
        ["--block-shape", "16", "32", "16"],
        ["--num-stages", "3"],
        ["--worker-case", "triton_graph"],
    ],
)
def test_invalid_options(extra):
    with pytest.raises(SystemExit):
        profiling.parse_args(["--quant-model", "unused", *extra])


def test_deduplicate_cases():
    args = profiling.parse_args(["--quant-model", "unused", "--cases", "triton_graph", "triton_graph"])
    assert args.cases == ["triton_graph"]


def test_profile_calls():
    function, graph = Mock(), Mock()
    profiling.run_profile_calls(function, None, 16, 4)
    assert function.call_count == 64
    function.reset_mock()
    profiling.run_profile_calls(function, graph, 16, 4)
    function.assert_not_called()
    assert graph.replay.call_count == 4


def test_worker_command_forwards_options(tmp_path):
    args = profiling.parse_args(
        [
            "--quant-model",
            str(tmp_path / "model"),
            "--tokens",
            "64",
            "--block-shape",
            "32",
            "64",
            "128",
            "--num-stages",
            "1",
            "--seed",
            "7",
            "--tp-rank",
            "2",
            "--replays",
            "3",
            "--graph-batch-size",
            "8",
            "--transpose-weight-tile",
        ]
    )
    command = profiling.worker_command(args, "qbmm_graph", tmp_path / "out")
    assert command[:2] == [sys.executable, "-u"]
    forwarded = profiling.parse_args(command[3:])
    assert forwarded.worker_case == "qbmm_graph"
    assert forwarded.quant_model == (tmp_path / "model").resolve()
    assert forwarded.output == (tmp_path / "out").resolve()
    assert forwarded.block_shape == [32, 64, 128]
    assert (forwarded.tokens, forwarded.num_stages, forwarded.seed, forwarded.tp_rank) == (64, 1, 7, 2)
    assert (forwarded.replays, forwarded.graph_batch_size) == (3, 8)
    assert forwarded.transpose_weight_tile


def test_dry_run_does_not_touch_files_or_launch_workers(tmp_path, capsys):
    output = tmp_path / "out"
    with patch.object(profiling.subprocess, "run") as runner:
        assert profiling.main(["--quant-model", "unused", "--output", str(output), "--dry-run"]) == 0
    runner.assert_not_called()
    assert not output.exists()
    assert len(json.loads(capsys.readouterr().out)["cases"]) == 4


def test_cli_dry_run_without_npu_imports():
    result = subprocess.run(
        [sys.executable, str(profiling.REPO_ROOT / "profile_tqbmm.py"), "--quant-model", "unused", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["parameters"]["block_shape"] == [64, 64, 128]


def test_export_preserves_raw_permissions_and_contents(tmp_path):
    case_dir, export = tmp_path / "raw", tmp_path / "export"
    create_analysis(case_dir)
    export.mkdir()
    before = {path: (path.stat().st_mode, path.read_bytes()) for path in case_dir.rglob("*") if path.is_file()}
    artifacts = profiling.export_case(case_dir, export, "triton_graph")
    assert len(artifacts) == 3
    assert all((export / entry["file"]).stat().st_size == entry["bytes"] > 0 for entry in artifacts)
    assert all(
        path.stat().st_mode == mode and path.read_bytes() == contents for path, (mode, contents) in before.items()
    )


@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_analysis_fails(tmp_path, empty):
    case_dir, export = tmp_path / "raw", tmp_path / "export"
    create_analysis(case_dir)
    export.mkdir()
    trace = next(case_dir.rglob("trace_view.json"))
    if empty:
        trace.write_text("", encoding="utf-8")
    else:
        trace.unlink()
    with pytest.raises(RuntimeError):
        profiling.export_case(case_dir, export, "triton_graph")


def test_duplicate_analysis_basenames_are_not_overwritten(tmp_path):
    case_dir, export = tmp_path / "raw", tmp_path / "export"
    create_analysis(case_dir)
    other = case_dir / "other"
    other.mkdir()
    (other / "op_summary_1.csv").write_text("other", encoding="utf-8")
    export.mkdir()
    artifacts = profiling.export_case(case_dir, export, "triton_graph")
    assert len(artifacts) == len({item["file"] for item in artifacts}) == 4


def test_complete_parent_run_and_archive(tmp_path):
    argv, output = model_args(tmp_path)
    with patch.object(profiling.subprocess, "run", side_effect=fake_worker) as runner:
        assert profiling.main(argv) == 0
    assert runner.call_count == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert all(case["status"] == "completed" and case["wall_seconds"] >= 0 for case in manifest["cases"])
    assert len(list((output / "export").iterdir())) == 12
    assert "worker log: qbmm_graph" in (output / "collect.log").read_text()
    with tarfile.open(output / "export.tar.gz") as archive:
        assert len(archive.getnames()) == 14
        assert not any("raw" in name for name in archive.getnames())
        packed = json.load(archive.extractfile("manifest.json"))
        assert packed["status"] == "completed"


@pytest.mark.parametrize("failure", ["exit", "timeout", "interrupt", "missing", "count", "incorrect"])
def test_failure_stops_and_preserves_results(tmp_path, failure):
    argv, output = model_args(tmp_path)

    def worker(command, **kwargs):
        name = command[command.index("--worker-case") + 1]
        if name == "triton_eager":
            return fake_worker(command, **kwargs)
        kwargs["stdout"].write("failed worker log\n")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 600)
        if failure == "interrupt":
            raise KeyboardInterrupt
        if failure == "exit":
            return SimpleNamespace(returncode=1)
        result = fake_worker(command, **kwargs)
        case_dir = output / "raw" / name
        if failure == "missing":
            next(case_dir.rglob("trace_view.json")).unlink()
        else:
            metadata = json.loads((case_dir / "capture.json").read_text())
            metadata["operator_calls" if failure == "count" else "strict_correctness"] = 0
            profiling.write_manifest(case_dir / "capture.json", metadata)
        return result

    with patch.object(profiling.subprocess, "run", side_effect=worker) as runner:
        assert profiling.main(argv) == (130 if failure == "interrupt" else 1)
    assert runner.call_count == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == ("interrupted" if failure == "interrupt" else "failed")
    assert manifest["cases"][0]["status"] == "completed"
    assert manifest["cases"][1]["status"] == manifest["status"]
    assert manifest["cases"][2]["status"] == "pending"
    assert "failed worker log" in (output / "collect.log").read_text()
    assert (output / "raw" / "triton_eager").is_dir()
    assert not (output / "export.tar.gz").exists()


def test_existing_output_is_not_overwritten(tmp_path):
    argv, output = model_args(tmp_path)
    output.mkdir()
    sentinel = output / "manifest.json"
    sentinel.write_text("keep", encoding="utf-8")
    with patch.object(profiling.subprocess, "run") as runner, pytest.raises(FileExistsError):
        profiling.main(argv)
    runner.assert_not_called()
    assert sentinel.read_text() == "keep"
