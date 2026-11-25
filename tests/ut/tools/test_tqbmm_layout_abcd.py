# SPDX-License-Identifier: Apache-2.0
"""Native/V2 diagnostic contract, no torch or A3 required."""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import benchmark_tqbmm_v2 as runner
import profile_tqbmm as profiling


@pytest.mark.parametrize("stale_replay", [None, "zero", "restored"])
def test_layout_graph_checks_both_replays_on_host_and_restores_input(stale_replay):
    torch = MagicMock()
    q = Mock(spec=["cpu", "zero_", "copy_"])
    golden = np.array([0, 7], dtype=np.float32)
    zeros = np.zeros_like(golden)
    ref, captured = (Mock(spec=["cpu"]) for _ in range(2))
    ref.cpu.return_value = golden
    captured.cpu.side_effect = [
        golden if stale_replay == "zero" else zeros,
        zeros if stale_replay == "restored" else golden,
    ]
    cases = {"composition": Mock(side_effect=[ref, None, captured])}  # reference, warmup, capture
    torch.zeros_like.side_effect = np.zeros_like

    def assert_exact(actual, expected, *, rtol, atol):
        assert rtol == atol == 0
        np.testing.assert_array_equal(actual, expected)

    torch.testing.assert_close = assert_exact
    original = np.array([1, 2], dtype=np.int8)
    q.cpu.return_value = original
    if stale_replay:
        with pytest.raises(AssertionError):
            runner.check_layout_graphs(torch, cases, q, original_cpu=original)
        q.cpu.assert_not_called()
    else:
        runner.check_layout_graphs(torch, cases, q)
        q.cpu.assert_called_once_with()
    q.zero_.assert_called_once_with()
    q.copy_.assert_called_once_with(original)
    assert torch.npu.NPUGraph.return_value.replay.call_count == (1 if stale_replay == "zero" else 2)


def test_layout_graph_rejects_zero_reference_before_capture():
    torch = MagicMock()
    ref = Mock(spec=["cpu"])
    ref.cpu.return_value = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="nonzero reference"):
        runner.check_layout_graphs(torch, {"composition": lambda: ref}, Mock())
    torch.npu.NPUGraph.assert_not_called()


def test_abcd_only_packs_once_and_passes_shared_views():
    q, w, sx, sw = [MagicMock() for _ in range(4)]
    w.shape = (1, 128, 64)
    w.is_contiguous.return_value = False
    w.stride.return_value = 1
    native, reference, prepare = Mock(), Mock(), Mock()
    cases, packed = runner.layout_cases(q, w, sx, sw, "bf16", "config", reference, native, prepare)
    assert tuple(cases) == runner.LAYOUT_CASES
    for _ in range(2):
        for function in cases.values():
            function()
    assert prepare.call_args_list[0].args == (q, w, sx, sw)
    assert prepare.call_args_list[1].args == (q, packed, sx, sw)
    assert prepare.call_args_list[0].kwargs == prepare.call_args_list[1].kwargs
    reference.assert_called_with(q, w, sx, sw, "bf16")
    native.assert_called_with(q[:, 0], packed[0], sw[0], pertoken_scale=sx[:, 0], output_dtype="bf16")
    native.return_value.unsqueeze.assert_called_with(1)
    w.contiguous.assert_called_once_with()
    w.shape = (2, 128, 64)
    with pytest.raises(ValueError, match="G=1"):
        runner.layout_cases(q, w, sx, sw, "bf16", "config", reference, native, prepare)


def test_profile_worker_preserves_cases_and_separation():
    base = ["--quant-model", ".", "--output", "unused"]
    for extra in ([], ["--layout-abcd"], ["--layout-abcd", "--timing-mode", "graph", "--check-only"]):
        with pytest.raises(SystemExit):
            runner.parse_args([*base, "--profile-metric", "MemoryAccess", *extra])
    args = runner.parse_args(
        [
            *base,
            "--layout-abcd",
            "--timing-mode",
            "graph",
            "--profile-metric",
            "MemoryAccess",
            "--profile-cases",
            "native_prepacked",
            "v2_prepacked",
            "--profile-replays",
            "3",
        ]
    )
    worker = runner.parse_args(runner.worker_command(args, 8192, 1)[3:])
    assert worker.layout_abcd and worker.profile_metric == "MemoryAccess" and worker.profile_replays == 3
    assert worker.profile_cases == ["native_prepacked", "v2_prepacked"]
    assert worker.tokens == [8192] and worker.seeds == [1]


def test_ratio_denominators_are_native_centered():
    ms = dict(zip(runner.LAYOUT_CASES, [2, 1, 8, 4]))
    report = {"T": 8192, "seed": 0, "layout_abcd": {}, "timings": {"graph": {"prequantized": {"median_ms": ms}}}}
    (row,) = runner.layout_comparisons(report)
    assert (row["A_over_B"], row["C_over_D"], row["D_over_B"], row["A_over_C"]) == (2, 2, 4, 0.25)
    packed = list(runner.timing_rows(report))[-1]
    assert packed["time_over_native_prepacked"] == 4 and "speedup_vs_v1" not in packed


def test_macro_is_opt_in_and_forwarded_to_worker():
    base = ["--quant-model", ".", "--output", "unused", "--timing-mode", "graph"]
    macro = ["--macro-tile", "128", "128"]
    profile = ["--profile-metric", "MemoryAccess", "--profile-cases", "v2_macro_prepacked"]
    for invalid in (macro, ["--layout-abcd", *profile], ["--layout-abcd", "--macro-tile", "256", "64"]):
        with pytest.raises(SystemExit):
            runner.parse_args([*base, *invalid])
    args = runner.parse_args([*base, "--layout-abcd", *macro, *profile])
    worker = runner.parse_args(runner.worker_command(args, 8192, 0)[3:])
    assert worker.macro_tile == [128, 128] and worker.profile_cases == ["v2_macro_prepacked"]
    ms = dict(zip(runner.LAYOUT_CASES + runner.MACRO_CASES, [2, 1, 8, 4, 2, 1]))
    report = {"T": 8192, "seed": 0, "layout_abcd": {}, "timings": {"graph": {"prequantized": {"median_ms": ms}}}}
    (row,) = runner.layout_comparisons(report)
    assert (row["A_over_E"], row["C_over_E"], row["D_over_F"], row["F_over_B"]) == (1, 4, 4, 1)


def test_macro_operand_byte_prediction():
    shape = (8192, 1, 4096, 1024)
    small = runner.tile_cost(*shape, 64, 64, 64)
    wide = runner.tile_cost(*shape, 128, 128, 64)
    assert small["input_bytes"] == 1024 * 2**20
    assert wide["x_bytes"] == wide["w_bytes"] == 256 * 2**20
    assert wide["logical_dots"] == 32768 and wide["acc_bytes_per_tile"] == 64 * 1024


def test_tile_major_is_opt_in_and_compares_against_same_macro():
    base = ["--quant-model", ".", "--output", "unused", "--layout-abcd", "--timing-mode", "graph"]
    macro = ["--macro-tile", "128", "128"]
    profile = ["--profile-metric", "PipeUtilization", "--profile-cases", runner.TILE_MAJOR_CASE]
    for extra in (["--tile-major-weight"], [*macro, *profile]):
        with pytest.raises(SystemExit):
            runner.parse_args([*base, *extra])
    args = runner.parse_args([*base, *macro, "--tile-major-weight", *profile])
    worker = runner.parse_args(runner.worker_command(args, 8192, 0)[3:])
    assert worker.tile_major_weight and worker.profile_cases == [runner.TILE_MAJOR_CASE]
    ms = dict(zip((*runner.LAYOUT_CASES, *runner.MACRO_CASES, runner.TILE_MAJOR_CASE), [8, 1, 9, 4, 6, 4, 2]))
    report = {"T": 8192, "seed": 0, "layout_abcd": {}, "timings": {"graph": {"prequantized": {"median_ms": ms}}}}
    (row,) = runner.layout_comparisons(report)
    assert (row["G_tile_major_ms"], row["F_over_G"], row["A_over_G"]) == (2, 2, 4)


def test_l0_panel_ab_cases_flags_and_ratios():
    base = ["--quant-model", ".", "--output", "unused", "--timing-mode", "graph"]
    mode = ["--l0-panel-ab"]
    profile = ["--profile-metric", "PipeUtilization", "--profile-cases", runner.TILE_MAJOR_CASE, runner.L0_PAIR_CASE]
    for invalid in (
        profile,
        [*mode, "--layout-abcd"],
        [*mode, "--num-stages", "1"],
        [*mode, "--block-shape", "64", "64", "128"],
        [*mode, "--profile-metric", "MemoryAccess", "--profile-cases", "v2"],
    ):
        with pytest.raises(SystemExit):
            runner.parse_args([*base, *invalid])
    args = runner.parse_args([*base, *mode, *profile])
    command = runner.worker_command(args, 8192, 0)
    assert "--block-shape" not in command and "--programs" not in command
    worker = runner.parse_args(command[3:])
    assert worker.l0_panel_ab and worker.layout_abcd and worker.tile_major_weight
    assert worker.profile_cases == [runner.TILE_MAJOR_CASE, runner.L0_PAIR_CASE]
    ms = dict(zip(runner.L0_AB_CASES, [4, 2, 8, 5]))
    report = {"T": 8192, "seed": 0, "layout_abcd": {}, "timings": {"graph": {"prequantized": {"median_ms": ms}}}}
    (row,) = runner.layout_comparisons(report)
    assert (row["G_over_H"], row["A_over_H"], row["H_over_B"]) == (8 / 5, 4 / 5, 5 / 2)
    shape = (8192, 1, 4096, 1024)
    g, h = (runner.tile_cost(*shape, 128, 128, bk) for bk in (128, 512))
    assert g["tiles"] == h["tiles"] and g["logical_dots"] == h["logical_dots"] * 4
    assert g["input_bytes"] == h["input_bytes"]


def test_focus_prefers_observed_pmu_only_counts_one_export(tmp_path):
    kernel = "_transpose_quant_batch_matmul_v2_kernel"
    (tmp_path / "kernel_details.csv").write_text(f"Name,Duration(us),aic_icache_miss_rate\n{kernel},12,0.1\n")
    (tmp_path / "op_summary_1.csv").write_text(
        f"Op Name,Task Duration(us),aic_read_main_memory_datas(KB)\n{kernel},12,2048\n{kernel},14,N/A\n"
    )
    focus = profiling.layout_profile_focus(tmp_path, "v2_prepacked", "MemoryAccess", 2)
    assert focus["matched_matmul_tasks"] == focus["observed_tasks"] == 2
    assert focus["sum_device_task_us_per_logical_call"] == 13
    assert focus["operators"][0]["family"] == "triton_tqbmm"
    assert focus["requested_fields_on_matmul"] == ["aic_read_main_memory_datas(KB)"]
    (counter,) = focus["operators"][0]["counter_call_means"].values()
    assert counter == {"mean": 2048, "covered_calls": 1}
    assert profiling.layout_profile_focus(tmp_path, "v2_prepacked", "L2Cache", 2)["counter_status"] == "not_observed"


def test_focus_empty_capture_fails(tmp_path):
    with pytest.raises(RuntimeError, match="No kernel_details"):
        profiling.layout_profile_focus(tmp_path, "composition", "PipeUtilization", 4)
