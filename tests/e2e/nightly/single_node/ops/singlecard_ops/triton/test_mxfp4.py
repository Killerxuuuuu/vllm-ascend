# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.mxfp4 import fp32_to_mxfp4, mxfp4_to_fp32
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

MXFP4_GROUP_SIZE = 32
E8M0_EXPONENT_BIAS = 127
E2M1_MAX_VALUE = 6.0


def _encode_e2m1_reference(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.abs()
    codes = torch.where(magnitude > 0.25, 1, 0)
    codes = torch.where(magnitude >= 0.75, 2, codes)
    codes = torch.where(magnitude > 1.25, 3, codes)
    codes = torch.where(magnitude >= 1.75, 4, codes)
    codes = torch.where(magnitude > 2.5, 5, codes)
    codes = torch.where(magnitude >= 3.5, 6, codes)
    codes = torch.where(magnitude > 5.0, 7, codes)
    sign = (values < 0).to(torch.int64) << 3
    return sign | codes


def _fp32_to_mxfp4_reference(
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = input_tensor.reshape(-1, MXFP4_GROUP_SIZE).to(torch.float32)
    amax = groups.abs().amax(dim=-1)
    safe_amax = amax.clamp_min(E2M1_MAX_VALUE * torch.finfo(torch.float32).tiny)
    scale_exponents = torch.ceil(torch.log2(safe_amax / E2M1_MAX_VALUE))
    scale_exponents = scale_exponents.clamp(-127, 127)
    scales = torch.exp2(scale_exponents)

    codes = _encode_e2m1_reference(groups / scales[:, None])
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scale_codes = (scale_exponents + E8M0_EXPONENT_BIAS).to(torch.uint8)

    packed_shape = (*input_tensor.shape[:-1], input_tensor.shape[-1] // 2)
    scales_shape = (
        *input_tensor.shape[:-1],
        input_tensor.shape[-1] // MXFP4_GROUP_SIZE,
    )
    return packed.reshape(packed_shape), scale_codes.reshape(scales_shape)


def _mxfp4_to_fp32_reference(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    low_codes = packed & 0xF
    high_codes = (packed >> 4) & 0xF
    codes = torch.stack((low_codes, high_codes), dim=-1).flatten(start_dim=-2)
    e2m1_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )
    values = e2m1_values[codes.to(torch.int64)]
    decoded_scales = torch.exp2(scales.to(torch.float32) - E8M0_EXPONENT_BIAS)
    return values * decoded_scales.repeat_interleave(MXFP4_GROUP_SIZE, dim=-1)


def _make_groups(shape: tuple[int, ...]) -> torch.Tensor:
    num_groups = math.prod(shape) // MXFP4_GROUP_SIZE
    generator = torch.Generator().manual_seed(0)
    normalized = torch.empty((num_groups, MXFP4_GROUP_SIZE)).uniform_(
        -5.75, 5.75, generator=generator
    )
    normalized[:, 0] = E2M1_MAX_VALUE
    scale_exponents = torch.arange(num_groups, dtype=torch.float32) % 9 - 4
    return (normalized * torch.exp2(scale_exponents[:, None])).reshape(shape)


@pytest.mark.parametrize("shape", [(1, 128), (3, 4, 128), (2, 96)])
@torch.inference_mode()
def test_mxfp4_pack_unpack_matches_reference(shape):
    init_device_properties_triton()
    input_cpu = _make_groups(shape)
    expected_packed, expected_scales = _fp32_to_mxfp4_reference(input_cpu)
    expected_output = _mxfp4_to_fp32_reference(expected_packed, expected_scales)

    packed, scales = fp32_to_mxfp4(input_cpu.to("npu"))
    output = mxfp4_to_fp32(packed, scales)

    assert packed.shape == expected_packed.shape
    assert scales.shape == expected_scales.shape
    assert output.shape == input_cpu.shape
    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.uint8
    assert output.dtype == torch.float32
    assert torch.equal(packed.cpu(), expected_packed)
    assert torch.equal(scales.cpu(), expected_scales)
    assert torch.equal(output.cpu(), expected_output)


@torch.inference_mode()
def test_mxfp4_round_ties_to_even():
    init_device_properties_triton()
    positive_values = torch.tensor(
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            6.0,
        ]
    )
    input_cpu = torch.cat((positive_values, -positive_values)).reshape(1, MXFP4_GROUP_SIZE)
    expected_packed, expected_scales = _fp32_to_mxfp4_reference(input_cpu)
    expected_packed_bytes = torch.tensor(
        [
            [
                0x00,
                0x21,
                0x22,
                0x43,
                0x44,
                0x65,
                0x66,
                0x77,
                0x80,
                0xA9,
                0xAA,
                0xCB,
                0xCC,
                0xED,
                0xEE,
                0xFF,
            ]
        ],
        dtype=torch.uint8,
    )

    packed, scales = fp32_to_mxfp4(input_cpu.to("npu"))

    assert torch.equal(expected_packed, expected_packed_bytes)
    assert torch.equal(packed.cpu(), expected_packed)
    assert torch.equal(scales.cpu(), expected_scales)
    assert scales.cpu().item() == E8M0_EXPONENT_BIAS


@torch.inference_mode()
def test_mxfp4_zero_group_uses_minimum_normal_scale():
    init_device_properties_triton()
    input_tensor = torch.zeros((1, MXFP4_GROUP_SIZE), dtype=torch.float32, device="npu")

    packed, scales = fp32_to_mxfp4(input_tensor)
    output = mxfp4_to_fp32(packed, scales)

    assert torch.count_nonzero(packed).cpu().item() == 0
    assert scales.cpu().item() == 1
    assert torch.equal(output, input_tensor)


def test_mxfp4_input_validation():
    with pytest.raises(TypeError, match="torch.float32"):
        fp32_to_mxfp4(torch.empty((1, 32), dtype=torch.float16))
    with pytest.raises(ValueError, match="divisible by 32"):
        fp32_to_mxfp4(torch.empty((1, 33), dtype=torch.float32))
    with pytest.raises(TypeError, match="packed must have dtype torch.uint8"):
        mxfp4_to_fp32(
            torch.empty((1, 16), dtype=torch.int8),
            torch.empty((1, 1), dtype=torch.uint8),
        )
    with pytest.raises(ValueError, match=r"scales.shape\[-1\] must be 1"):
        mxfp4_to_fp32(
            torch.empty((1, 16), dtype=torch.uint8),
            torch.empty((1, 2), dtype=torch.uint8),
        )
