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
    prepared = Mock(return_value=sentinel.prepared_output)
    prepare = Mock(return_value=prepared)
    namespace = {
        "torch": SimpleNamespace(Tensor=object, int8="int8", stack=stack),
        "torch_npu": SimpleNamespace(npu_dynamic_quant=dynamic_quant, npu_quant_matmul=qbmm),
        "envs": SimpleNamespace(VLLM_ASCEND_DSV4_WO_A_TRITON=True),
        "dynamic_quant_grouped_wo_a": fused,
        "PreparedDynamicWoA": prepare,
    }
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in CONFIG_NAMES for target in node.targets)
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name in ("_a3_int8_grouped_wo_a", "_a3_int8_grouped_wo_a_for_layer", "_wo_a_tensor_signature")
        )
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(
        run=namespace["_a3_int8_grouped_wo_a"],
        namespace=namespace,
        fused=fused,
        dynamic_quant=dynamic_quant,
        qbmm=qbmm,
        stack=stack,
        run_layer=namespace["_a3_int8_grouped_wo_a_for_layer"],
        prepare=prepare,
        prepared=prepared,
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


def make_owner(groups=1):
    x, weight, scale = inputs(groups=groups)
    owner = SimpleNamespace(wo_a=SimpleNamespace(weight=weight, weight_scale=scale))
    owner._get_batched_wo_a_weight = Mock(side_effect=lambda groups: owner.wo_a.weight)
    owner._get_batched_wo_a_scale = Mock(side_effect=lambda groups: owner.wo_a.weight_scale)
    return owner, x


def test_network_prepares_once_and_forwards_each_current_activation(route):
    owner, x = make_owner()
    for value in (x, sentinel.next_activation, sentinel.third_activation):
        assert route.run_layer(owner, value, 1) is sentinel.prepared_output
    route.prepare.assert_called_once_with(
        owner.wo_a.weight,
        owner.wo_a.weight_scale,
        block_shape=(64, 64, 128),
        num_stages=2,
        transpose_weight_tile=False,
    )
    assert [call.args[0] for call in route.prepared.call_args_list] == [
        x,
        sentinel.next_activation,
        sentinel.third_activation,
    ]
    owner._get_batched_wo_a_weight.assert_called_once_with(1)
    route.fused.assert_not_called()
    route.dynamic_quant.assert_not_called()


@pytest.mark.parametrize("change", ["weight", "scale", "pointer", "stride", "groups"])
def test_network_rebuilds_binding_after_static_replacement(route, change):
    owner, x = make_owner()
    route.run_layer(owner, x, 1)
    groups = 1
    if change == "weight":
        owner.wo_a.weight = inputs()[1]
    elif change == "scale":
        owner.wo_a.weight_scale = inputs()[2]
    elif change == "pointer":
        owner.wo_a.weight.data_ptr.return_value = 123
    elif change == "stride":
        owner.wo_a.weight.stride.return_value = (1, 2, 3)
    else:
        groups = 2
    route.run_layer(owner, x, groups)
    assert route.prepare.call_count == 2
    assert route.prepare.call_args.args == (owner.wo_a.weight, owner.wo_a.weight_scale)


@pytest.mark.parametrize("triton,full_switch", [(False, False), (True, True)])
def test_composition_and_cp_weight_switch_do_not_use_prepared(route, triton, full_switch):
    owner, x = make_owner()
    owner.enable_dsa_cp_with_o_proj_tp = full_switch
    route.namespace["envs"].VLLM_ASCEND_DSV4_WO_A_TRITON = triton
    route.run_layer(owner, x, 1)
    route.prepare.assert_not_called()
    assert not hasattr(owner, "_a3_wo_a_prepared")
    assert route.fused.call_count == int(triton)


def test_network_does_not_share_plans_between_layers(route):
    first, x = make_owner()
    second, _ = make_owner()
    route.run_layer(first, x, 1)
    route.run_layer(second, x, 1)
    assert route.prepare.call_count == 2


def test_network_missing_prepared_raises_without_native_fallback(route):
    route.namespace["PreparedDynamicWoA"] = None
    owner, x = make_owner()
    with pytest.raises(RuntimeError, match="unavailable"):
        route.run_layer(owner, x, 1)
    route.dynamic_quant.assert_not_called()


def test_network_plan_construction_error_is_not_hidden_or_cached(route):
    route.prepare.side_effect = ValueError("ND storage required")
    owner, x = make_owner()
    with pytest.raises(ValueError, match="ND storage"):
        route.run_layer(owner, x, 1)
    assert not hasattr(owner, "_a3_wo_a_prepared")
    route.fused.assert_not_called()


def test_actual_attention_call_sites_use_layer_preparation():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    implementation = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendDSAImpl")
    forward = next(
        node for node in implementation.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_o_proj"
    )
    calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)]
    assert (
        sum(isinstance(node.func, ast.Name) and node.func.id == "_a3_int8_grouped_wo_a_for_layer" for node in calls)
        == 2
    )
    cp = ast.parse((SOURCE.parent / "context_parallel/dsa_cp.py").read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_a3_int8_grouped_wo_a_for_layer"
        for node in ast.walk(cp)
    )
