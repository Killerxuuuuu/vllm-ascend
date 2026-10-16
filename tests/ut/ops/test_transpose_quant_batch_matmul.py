# SPDX-License-Identifier: Apache-2.0
"""Host contract tests; actual INT8 arithmetic is covered by the NPU tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.triton.transpose_quant_batch_matmul import (
    MAX_INT8_ACCUMULATION_K,
    _validate_inputs,
    transpose_quant_batch_matmul,
)
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import AscendDeviceType


def inputs(tokens=17, groups=2, inner=65, columns=33):
    return (
        torch.empty(tokens, groups, inner, dtype=torch.int8),
        torch.empty(groups, columns, inner, dtype=torch.int8).transpose(1, 2),
        torch.empty(tokens, groups, dtype=torch.float32),
        torch.empty(groups, columns, dtype=torch.float32),
    )


@pytest.mark.parametrize("tokens", [0, 1, 17])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_contract_accepts_strided_weight_and_zero_tokens(tokens, output_dtype):
    _validate_inputs(*inputs(tokens=tokens), output_dtype)


@pytest.mark.parametrize("position", [0, 1])
def test_rejects_floating_matmul_inputs(position):
    tensors = list(inputs())
    tensors[position] = tensors[position].float()
    with pytest.raises(TypeError, match="INT8"):
        _validate_inputs(*tensors, torch.bfloat16)


@pytest.mark.parametrize("position", [2, 3])
def test_rejects_non_fp32_scales(position):
    tensors = list(inputs())
    tensors[position] = tensors[position].bfloat16()
    with pytest.raises(TypeError, match="FP32"):
        _validate_inputs(*tensors, torch.bfloat16)


@pytest.mark.parametrize("position", [2, 3])
def test_rejects_scales_missing_group_dimension(position):
    tensors = list(inputs())
    tensors[position] = tensors[position][:, 0]
    with pytest.raises(ValueError, match="x_scale"):
        _validate_inputs(*tensors, torch.bfloat16)


def test_rejects_shape_mismatch():
    x, w, sx, sw = inputs()
    with pytest.raises(ValueError, match="G,K"):
        _validate_inputs(x, w[:, :-1], sx, sw, torch.bfloat16)


def test_rejects_zero_groups():
    with pytest.raises(ValueError, match="positive"):
        _validate_inputs(*inputs(groups=0), torch.bfloat16)


def test_rejects_possible_int32_overflow():
    with pytest.raises(ValueError, match="overflow"):
        _validate_inputs(*inputs(tokens=0, inner=MAX_INT8_ACCUMULATION_K + 1), torch.bfloat16)


def test_rejects_integer_output():
    with pytest.raises(TypeError, match="output dtype"):
        _validate_inputs(*inputs(), torch.int8)


def test_rejects_mixed_devices():
    x, w, sx, sw = inputs()
    with pytest.raises(ValueError, match="same device"):
        _validate_inputs(x, w.to("meta"), sx, sw, torch.bfloat16)


def test_no_silent_cpu_fallback():
    with pytest.raises(ValueError, match="Ascend NPU"):
        transpose_quant_batch_matmul(*inputs())


@pytest.mark.parametrize(
    "enabled,prefix,device_type,expected_nz_calls",
    [
        (True, "attn.wo_a", AscendDeviceType.A3, 0),
        (False, "attn.wo_a", AscendDeviceType.A3, 1),
        (True, "attn.wo_b", AscendDeviceType.A3, 1),
        (True, "attn.wo_a", AscendDeviceType.A5, 1),
    ],
)
def test_triton_wo_a_weight_stays_nd_only_when_opted_in(enabled, prefix, device_type, expected_nz_calls):
    module = "vllm_ascend.quantization.methods.w8a8_dynamic"
    layer = SimpleNamespace(
        prefix=prefix,
        weight=torch.nn.Parameter(torch.ones(8, 16, dtype=torch.int8), requires_grad=False),
        weight_scale=torch.nn.Parameter(torch.ones(8, 1), requires_grad=False),
        weight_offset=torch.nn.Parameter(torch.zeros(8, 1), requires_grad=False),
    )
    with (
        patch(f"{module}.envs.VLLM_ASCEND_DSV4_WO_A_TRITON", enabled),
        patch(f"{module}.get_ascend_device_type", return_value=device_type),
        patch(f"{module}.maybe_trans_nz", side_effect=lambda tensor: tensor) as to_nz,
    ):
        AscendW8A8DynamicLinearMethod().process_weights_after_loading(layer)
    assert to_nz.call_count == expected_nz_calls
    assert layer.weight.shape == (16, 8)
