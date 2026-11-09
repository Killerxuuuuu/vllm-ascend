# SPDX-License-Identifier: Apache-2.0
"""Audit the actual Wo_a routing code without importing torch/vLLM/NPU.

These check parameter forwarding and route isolation, not device arithmetic
or performance, including for token counts beyond the measured tuning set.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, sentinel

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_v1.py"
CONFIG_NAMES = frozenset(("A3_WO_A_TRITON_BLOCK_SHAPE", "A3_WO_A_TRITON_NUM_STAGES"))


@pytest.fixture
def route():
    fused = Mock(return_value=sentinel.fused_output)
    dynamic_quant = Mock(return_value=(sentinel.quantized, MagicMock()))
    qbmm = Mock(return_value=sentinel.qbmm_output)
    stack = Mock(return_value=sentinel.stacked_output)
    namespace = {
        "torch": SimpleNamespace(Tensor=object, int8="int8", stack=stack),
        "torch_npu": SimpleNamespace(npu_dynamic_quant=dynamic_quant, npu_quant_matmul=qbmm),
        "envs": SimpleNamespace(VLLM_ASCEND_DSV4_WO_A_TRITON=True),
        "dynamic_quant_grouped_wo_a": fused,
    }
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in CONFIG_NAMES for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "_a3_int8_grouped_wo_a")
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(
        run=namespace["_a3_int8_grouped_wo_a"],
        namespace=namespace,
        fused=fused,
        dynamic_quant=dynamic_quant,
        qbmm=qbmm,
        stack=stack,
    )


def inputs(tokens=16, groups=1):
    x = MagicMock(ndim=3, shape=(tokens, groups, 4096), dtype="bfloat16")
    weight = MagicMock(ndim=3, shape=(groups, 4096, 1024), dtype="int8")
    scales = MagicMock(shape=(groups, 1024))
    scales.numel.return_value = groups * 1024
    return x, weight, scales


@pytest.mark.parametrize("tokens", [0, 1, 16, 64, 512, 1024, 8192])
@pytest.mark.parametrize("groups", [1, 2])
def test_network_route_forwards_tuned_candidate(route, tokens, groups):
    x, weight, scales = inputs(tokens, groups)
    assert route.run(x, weight, scales) is sentinel.fused_output
    route.fused.assert_called_once_with(
        x, weight, scales, block_shape=(64, 64, 128), num_stages=2, transpose_weight_tile=False
    )
    route.dynamic_quant.assert_not_called()
    route.qbmm.assert_not_called()
    route.stack.assert_not_called()


@pytest.mark.parametrize("groups", [1, 2])
def test_disabled_triton_keeps_native_composition(route, groups):
    route.namespace["envs"].VLLM_ASCEND_DSV4_WO_A_TRITON = False
    x, weight, scales = inputs(groups=groups)
    assert route.run(x, weight, scales) is sentinel.stacked_output
    route.fused.assert_not_called()
    assert route.dynamic_quant.call_count == route.qbmm.call_count == groups
    route.stack.assert_called_once_with([sentinel.qbmm_output] * groups, dim=1)


def test_missing_triton_is_not_silently_replaced(route):
    route.namespace["dynamic_quant_grouped_wo_a"] = None
    with pytest.raises(RuntimeError, match="unavailable"):
        route.run(*inputs())
    route.dynamic_quant.assert_not_called()
    route.qbmm.assert_not_called()


def test_invalid_scales_are_rejected_before_tuned_launch(route):
    x, weight, scales = inputs()
    scales.numel.return_value = 1
    with pytest.raises(ValueError, match="one scale per output channel"):
        route.run(x, weight, scales)
    route.fused.assert_not_called()


def test_zero_groups_are_rejected_before_tuned_launch(route):
    with pytest.raises(ValueError, match="at least one group"):
        route.run(*inputs(groups=0))
    route.fused.assert_not_called()
