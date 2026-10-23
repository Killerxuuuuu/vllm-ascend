# SPDX-License-Identifier: Apache-2.0
"""CPU-only diagnostic control tests; no claim of NPU correctness/performance."""

import csv
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import diagnose_tqbmm_eager as diagnostic


def test_defaults():
    args = diagnostic.parse_args(["--quant-model", "unused"])
    assert args.tokens == [1, 16, 64, 512]
    assert args.block_shape == [64, 64, 128]
    assert args.num_stages == 2
    assert args.tp_size == 8 and args.tp_rank == 0
    assert not args.transpose_weight_tile
    assert diagnostic.launch_plan(args.iterations, args.graph_batch_size) == (16, 256)


@pytest.mark.parametrize(
    "options",
    [
        ["--tokens", "0"],
        ["--tokens", "1", "-1"],
        ["--warmup", "-1"],
        ["--iterations", "0"],
        ["--repeats", "0"],
        ["--graph-batch-size", "0"],
        ["--tp-size", "0"],
        ["--tp-rank", "8"],
        ["--tp-rank", "-1"],
        ["--layer", "-1"],
        ["--block-shape", "24", "32", "64"],
        ["--block-shape", "16", "32", "16"],
        ["--num-stages", "3"],
    ],
)
def test_invalid_options(options):
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["--quant-model", "unused", *options])


def test_dry_run_is_cpu_only(tmp_path, capsys):
    output = tmp_path / "not-created"
    with patch.object(diagnostic, "run_device") as runner:
        assert diagnostic.main(["--quant-model", "unused", "--output", str(output), "--dry-run"]) == 0
    runner.assert_not_called()
    assert not output.exists()
    report = json.loads(capsys.readouterr().out)
    assert tuple(report["cases"]) == diagnostic.CASES
    process = subprocess.run(
        [sys.executable, str(diagnostic.REPO_ROOT / "diagnose_tqbmm_eager.py"), "--quant-model", "unused", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(process.stdout)["results"] == []


def test_case_order_rotates_without_skipping():
    assert diagnostic.case_order(0) == diagnostic.CASES
    assert diagnostic.case_order(1)[-1] == diagnostic.CASES[0]
    for repeat in range(5):
        assert set(diagnostic.case_order(repeat)) == set(diagnostic.CASES)


@pytest.mark.parametrize("reuse", [False, True])
def test_unchecked_launch_preserves_grid_strides_and_configuration(reuse):
    args = diagnostic.parse_args(["--quant-model", "unused"])
    x = Mock(shape=(512, 1, 4096), device="npu:0")
    weight = Mock(shape=(1, 4096, 1024))
    sx, sw, output = Mock(), Mock(), Mock()
    x.stride.return_value = (4096, 4096, 1)
    weight.stride.return_value = (4194304, 1, 4096)
    sx.stride.return_value = (1, 1)
    sw.stride.return_value = (1024, 1)
    kernel = Mock()
    launch = Mock()
    kernel.__getitem__ = Mock(return_value=launch)
    torch_module = Mock(bfloat16="bf16")
    torch_module.empty.return_value = output
    triton = SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b)
    actual = diagnostic.launch_unchecked(
        torch_module,
        triton,
        kernel,
        x,
        weight,
        sx,
        sw,
        args,
        4,
        output if reuse else None,
    )
    assert actual is output
    assert torch_module.empty.call_count == (0 if reuse else 1)
    kernel.__getitem__.assert_called_once_with((128, 1))
    positional, named = launch.call_args
    assert positional == (
        x,
        weight,
        sx,
        sw,
        output,
        512,
        1,
        4096,
        1024,
        4096,
        4096,
        1,
        4194304,
        1,
        4096,
        1,
        1,
        1024,
        1,
    )
    assert named == {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 128,
        "GROUP_SIZE_M": 4,
        "TRANSPOSE_WEIGHT_TILE": False,
        "num_stages": 2,
    }


@pytest.mark.parametrize("batch", [1, 16])
def test_measure_counts_and_separates_timers(batch):
    function = Mock()
    start, end = Mock(), Mock()
    start.elapsed_time.return_value = 1.6
    npu = Mock()
    npu.Event.side_effect = [start, end]
    with patch.object(diagnostic.time, "perf_counter", side_effect=[1, 1.032, 1.064]):
        result = diagnostic.measure(function, npu, iterations=31, calls_per_invocation=batch)
    invocations, actual = diagnostic.launch_plan(31, batch)
    assert function.call_count == invocations
    assert result["actual_calls"] == actual
    assert result["invocations"] == invocations
    assert result["host_submit_us"] == pytest.approx(32000 / actual)
    assert result["stream_us"] == pytest.approx(1600 / actual)
    assert result["wall_us"] == pytest.approx(64000 / actual)
    npu.synchronize.assert_called_once()
    end.synchronize.assert_called_once()


def sample_result():
    return {
        "shape_T_G_K_N": [512, 1, 4096, 1024],
        "samples": {
            name: [
                {"host_submit_us": 10.0, "stream_us": 60.0, "wall_us": 61.0, "actual_calls": 256, "invocations": 256}
            ]
            for name in diagnostic.CASES
        },
        "strict_correctness": True,
    }


def test_report_summaries_csv_and_atomic_files(tmp_path):
    result = sample_result()
    result["median_us"] = diagnostic.summarize(result["samples"])
    assert result["median_us"]["A_full"]["stream_us"] == 60.0
    diagnostic.save_report(tmp_path, {"status": "running", "results": [result]})
    with (tmp_path / "timings.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[0]["T"] == "512" and rows[0]["actual_calls"] == "256"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_main_preserves_partial_results(tmp_path, status):
    model, output = tmp_path / "model", tmp_path / "output"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")

    def runtime(args, report, destination):
        report["results"].append(sample_result())
        diagnostic.save_report(destination, report)
        if status == "failed":
            raise RuntimeError("diagnostic failure")
        if status == "interrupted":
            raise KeyboardInterrupt

    with patch.object(diagnostic, "run_device", side_effect=runtime):
        code = diagnostic.main(["--quant-model", str(model), "--output", str(output)])
    assert code == {"completed": 0, "failed": 1, "interrupted": 130}[status]
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == status and len(report["results"]) == 1
    assert (output / "timings.csv").is_file()
    assert (output / "error.log").exists() == (status == "failed")
    with patch.object(diagnostic, "run_device") as runner, pytest.raises(FileExistsError):
        diagnostic.main(["--quant-model", str(model), "--output", str(output)])
    runner.assert_not_called()
