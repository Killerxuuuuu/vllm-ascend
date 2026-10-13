# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import AscendDSAImpl, _a3_int8_grouped_wo_a
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import AscendDeviceType


def _dynamic_quant(x: torch.Tensor, dst_type: torch.dtype = torch.int8):
    assert dst_type == torch.int8
    scale = x.float().abs().amax(dim=-1).clamp_min(1e-6) / 127
    quantized = (x.float() / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _quant_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    pertoken_scale: torch.Tensor,
    output_dtype: torch.dtype,
    **kwargs,
):
    assert x.dtype == weight.dtype == torch.int8
    return (
        (x.float() @ weight.float())
        * pertoken_scale.float().reshape(-1, 1)
        * weight_scale.float().reshape(1, -1)
    ).to(output_dtype)


def test_a3_int8_wo_a_and_wo_b_use_quant_matmul(monkeypatch):
    from vllm_ascend.attention import dsa_v1

    monkeypatch.setattr(dsa_v1, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    quant_calls = []

    def record_quant_matmul(*args, **kwargs):
        quant_calls.append((args[0].shape, args[1].shape))
        return _quant_matmul(*args, **kwargs)

    monkeypatch.setattr(dsa_v1.torch_npu, "npu_dynamic_quant", _dynamic_quant)
    monkeypatch.setattr(dsa_v1.torch_npu, "npu_quant_matmul", record_quant_matmul)
    monkeypatch.setattr(dsa_v1, "oproj_tp_enable", lambda: False)

    wo_a_weight = torch.arange(-12, 12, dtype=torch.int8).reshape(2, 4, 3)
    wo_a_scale = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    wo_b_weight = torch.arange(-18, 18, dtype=torch.int8).reshape(6, 6)
    wo_b_scale = torch.full((6,), 0.25)
    wo_b_method = AscendW8A8DynamicLinearMethod()
    wo_b_layer = SimpleNamespace(weight=wo_b_weight, weight_scale=wo_b_scale)
    impl = SimpleNamespace(
        n_local_groups=2,
        support_fp8_attention=False,
        wo_a=SimpleNamespace(
            weight=wo_a_weight,
            weight_scale=wo_a_scale,
            quant_method=AscendLinearMethod(AscendW8A8DynamicLinearMethod()),
        ),
        wo_b=lambda x: wo_b_method.apply(wo_b_layer, x),
    )
    impl._wo_a_bmm = lambda x: AscendDSAImpl._wo_a_bmm(impl, x)
    activations = torch.arange(1, 17, dtype=torch.float32).reshape(2, 2, 4).to(torch.bfloat16)
    output = torch.empty(2, 6, dtype=torch.bfloat16)

    result = AscendDSAImpl._forward_o_proj(impl, activations, output)

    assert result is output
    assert quant_calls == [((2, 4), (4, 3)), ((2, 4), (4, 3)), ((2, 6), (6, 6))]
    expected_groups = []
    for group in range(2):
        quantized_x, x_scale = _dynamic_quant(activations[:, group, :])
        expected_groups.append(
            _quant_matmul(
                quantized_x,
                wo_a_weight[group],
                wo_a_scale.reshape(2, 3)[group],
                pertoken_scale=x_scale,
                output_dtype=activations.dtype,
            )
        )
    expected_wo_a = torch.stack(expected_groups, dim=1).reshape(2, 6)
    quantized_wo_b_input, wo_b_input_scale = _dynamic_quant(expected_wo_a)
    expected = _quant_matmul(
        quantized_wo_b_input,
        wo_b_weight,
        wo_b_scale,
        pertoken_scale=wo_b_input_scale,
        output_dtype=activations.dtype,
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("weight_shape", "scale_shape"),
    [((2, 5, 3), (6,)), ((2, 4, 3), (5,))],
)
def test_a3_int8_wo_a_rejects_incompatible_weight_or_scale(weight_shape, scale_shape):
    x = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    weight = torch.zeros(weight_shape, dtype=torch.int8)
    scale = torch.ones(scale_shape)

    with pytest.raises(ValueError, match="A3 INT8 wo_a"):
        _a3_int8_grouped_wo_a(x, weight, scale)


def test_bf16_wo_a_still_uses_transpose_batchmatmul():
    from vllm_ascend.attention import dsa_v1

    expected = torch.ones(2, 2, 3, dtype=torch.bfloat16)
    impl = SimpleNamespace(
        support_fp8_attention=False,
        wo_a=SimpleNamespace(weight=torch.ones(2, 4, 3, dtype=torch.bfloat16), quant_method=None),
    )
    with (
        patch.object(dsa_v1, "get_ascend_device_type", return_value=AscendDeviceType.A3),
        patch.object(dsa_v1.torch_npu, "npu_transpose_batchmatmul", return_value=expected) as bmm,
    ):
        result = AscendDSAImpl._wo_a_bmm(impl, torch.ones(2, 2, 4, dtype=torch.bfloat16))

    assert result is expected
    bmm.assert_called_once()


def test_a5_fp8_wo_a_still_uses_transpose_quant_batchmatmul():
    from vllm_ascend.attention import dsa_v1

    expected = torch.ones(2, 2, 3, dtype=torch.bfloat16)
    activations = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    impl = SimpleNamespace(
        support_fp8_attention=True,
        wo_a=SimpleNamespace(
            weight=torch.ones(2, 4, 3, dtype=torch.bfloat16),
            weight_scale=torch.zeros(2, 4, 3, dtype=torch.uint8).view(torch.float8_e8m0fnu),
            quant_method=None,
        ),
    )
    with (
        patch.object(dsa_v1, "get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch.object(
            dsa_v1.torch_npu,
            "npu_dynamic_mx_quant",
            return_value=(activations, torch.zeros(2, 2, 4, dtype=torch.uint8).view(torch.float8_e8m0fnu)),
        ) as dynamic_quant,
        patch.object(dsa_v1.torch_npu, "npu_transpose_quant_batchmatmul", return_value=expected) as bmm,
    ):
        result = AscendDSAImpl._wo_a_bmm(impl, activations)

    assert result is expected
    dynamic_quant.assert_called_once()
    bmm.assert_called_once()
