# SPDX-License-Identifier: Apache-2.0
"""CPU oracle and post-failure hook tests. No NPU execution is simulated."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tools.dsv4_a3_wo.tqbmm_failure import summarize_failure

BENCHMARK = Path(__file__).resolve().parents[3] / "tools/dsv4_a3_wo/benchmark_tqbmm.py"


def example():
    x = np.arange(8 * 2 * 5, dtype=np.int8).reshape(8, 2, 5) - 40
    weight = np.arange(2 * 5 * 6, dtype=np.int8).reshape(2, 5, 6) - 30
    sx = np.ones((8, 2), dtype=np.float32) * 0.125
    sw = np.ones((2, 6), dtype=np.float32) * 0.25
    native = np.einsum("tgk,gkn->tgn", x.astype(np.int64), weight.astype(np.int64)).astype(np.float32)
    native = (native * sw[None]) * sx[:, :, None]
    return dict(
        x=x,
        weight=weight,
        sx=sx,
        sw=sw,
        actual=native.copy(),
        native=native,
        block_shape=(2, 3, 4),
        reuse_m=2,
        output_cast=float,
    )


def test_tiles_accumulators_and_exact_cpu_oracle():
    case = example()
    case["actual"][4:8, 1, 3:6] += 1000
    report = summarize_failure(**case)
    assert report["mismatched_elements"] == 12
    assert report["max_abs"] == 1000
    assert {tuple(tile["group_Mtile_Ntile"]) for tile in report["top_error_tiles"]} == {(1, 2, 1), (1, 3, 1)}
    assert all(tile["errors"] == 6 for tile in report["top_error_tiles"])
    assert {point["accumulator"] for point in report["cpu_golden_points"]} == {0, 1}
    assert report["sample_native_matches_cpu"] == len(report["cpu_golden_points"])
    assert report["sample_triton_matches_cpu"] == 0
    json.dumps(report, allow_nan=False)


def test_native_mismatch_is_not_blindly_attributed_to_triton():
    case = example()
    case["native"][3, 0, 0] += 1000
    report = summarize_failure(**case)
    assert report["sample_native_matches_cpu"] == 0
    assert report["sample_triton_matches_cpu"] == 1


def test_nonfinite_outputs_remain_serializable():
    case = example()
    case["actual"][0, 0, 0] = np.nan
    report = summarize_failure(**case)
    assert report["cpu_golden_points"][0]["triton"] == "nan"
    json.dumps(report, allow_nan=False)


def test_golden_uses_exact_integer_sum_at_real_k():
    case = dict(
        x=np.full((1, 1, 4096), -128, dtype=np.int8),
        weight=np.full((1, 4096, 1), -128, dtype=np.int8),
        sx=np.ones((1, 1), dtype=np.float32),
        sw=np.ones((1, 1), dtype=np.float32),
        native=np.full((1, 1, 1), 67108864, dtype=np.float32),
        actual=np.zeros((1, 1, 1), dtype=np.float32),
        block_shape=(128, 64, 128),
        reuse_m=2,
        output_cast=float,
    )
    report = summarize_failure(**case)
    assert report["cpu_golden_points"][0]["cpu_integer_sum"] == 67108864
    assert report["sample_native_matches_cpu"] == 1


def test_matching_outputs_need_no_golden_samples():
    report = summarize_failure(**example())
    assert report["mismatched_elements"] == 0
    assert report["cpu_golden_points"] == []


def test_failure_points_identify_second_n_accumulator():
    case = example()
    case["actual"][3, 1, 4] += 100
    report = summarize_failure(**case, reuse_n=2)
    point = report["cpu_golden_points"][0]
    assert point["accumulator"] == point["N_accumulator"] == 1
    assert point["N_pair"] == 0


def comparison_fragment():
    tree = ast.parse(BENCHMARK.read_text(encoding="utf-8"))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    start = next(
        i
        for i, node in enumerate(run.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "(actual, expected)"
    )
    fragment = ast.Module(body=run.body[start : start + 3], type_ignores=[])
    return compile(ast.fix_missing_locations(fragment), str(BENCHMARK), "exec")


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("failing", [False, True])
def test_capture_is_after_comparison_and_preserves_original_assertion(monkeypatch, tmp_path, capture, failing):
    events = []
    original = AssertionError("original mismatch")

    def compare(*args, **kwargs):
        events.append("compare")
        if failing:
            raise original

    def collect(**kwargs):
        events.append("dump")
        return tmp_path

    monkeypatch.setattr("tools.dsv4_a3_wo.tqbmm_failure.dump_failure", collect)
    scope = dict(
        fused=lambda: events.append("fused"),
        reference=lambda: events.append("reference"),
        torch=SimpleNamespace(testing=SimpleNamespace(assert_close=compare)),
        args=SimpleNamespace(strict=True, failure_dir=tmp_path if capture else None),
        x=None,
        weight=None,
        sx=None,
        sw=None,
        sys=SimpleNamespace(stderr=None),
    )
    if failing:
        with pytest.raises(AssertionError) as caught:
            exec(comparison_fragment(), scope)
        assert caught.value is original
    else:
        exec(comparison_fragment(), scope)
    assert events == ["fused", "reference", "compare"] + (["dump"] if failing and capture else [])


def test_diagnostic_error_does_not_replace_original_assertion(monkeypatch, tmp_path):
    original = AssertionError("numerical failure")
    monkeypatch.setattr("tools.dsv4_a3_wo.tqbmm_failure.dump_failure", Mock(side_effect=OSError("disk full")))
    scope = dict(
        fused=lambda: None,
        reference=lambda: None,
        torch=SimpleNamespace(testing=SimpleNamespace(assert_close=Mock(side_effect=original))),
        args=SimpleNamespace(strict=True, failure_dir=tmp_path),
        x=None,
        weight=None,
        sx=None,
        sw=None,
        sys=SimpleNamespace(stderr=None),
    )
    with pytest.raises(AssertionError) as caught:
        exec(comparison_fragment(), scope)
    assert caught.value is original
