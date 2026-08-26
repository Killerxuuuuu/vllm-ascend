#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Triton Ascend helpers for MXFP4 E2M1 conversion."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

MXFP4_GROUP_SIZE: tl.constexpr = 32
E8M0_EXPONENT_BIAS: tl.constexpr = 127
E8M0_MIN_EXPONENT: tl.constexpr = -127
E8M0_MAX_EXPONENT: tl.constexpr = 127
E2M1_MAX_VALUE: tl.constexpr = 6.0
FP32_MIN_NORMAL: tl.constexpr = 2**-126
MXFP4_UNPACK_BLOCK_SIZE: tl.constexpr = 1024
FP32_EXPONENT_SHIFT: tl.constexpr = 23
FP32_MIN_E8M0_SCALE_BITS: tl.constexpr = 1 << 22


@triton.jit
def _decode_e2m1(code):
    """Decode E2M1 bit patterns in ``code`` to floating-point values."""
    code = code.to(tl.int32)

    sign_bit = (code >> 3) & 0x1
    magnitude_code = code & 0x7
    exponent_bits = magnitude_code >> 1
    mantissa_bit = magnitude_code & 0x1

    mantissa = mantissa_bit.to(tl.float32)

    # E2M1 values with exponent bits 00 are zero and 0.5.
    subnormal_value = mantissa * 0.5

    # The remaining values are (1 + mantissa / 2) * 2 ** (exponent - 1).
    normal_value = (1.0 + mantissa * 0.5) * tl.exp2(
        exponent_bits.to(tl.float32) - 1.0
    )
    magnitude = tl.where(exponent_bits == 0, subnormal_value, normal_value)

    return tl.where(sign_bit == 0, magnitude, -magnitude)


@triton.jit
def _decode_e2m1_half_units(
    code,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    """Decode E2M1 into exact signed half units for an INT8 Cube dot.

    The returned integer is exactly twice the represented E2M1 value:
    ``[0, 1, 2, 3, 4, 6, 8, 12]`` for positive magnitude codes. Keeping the
    direct INT8 lookup avoids computed casts, bitcasts, and narrow-integer
    arithmetic before ``tl.dot`` on Ascend A5.
    """
    shape: tl.constexpr = (ROWS, COLS)
    result = tl.zeros(shape, tl.int8)

    # Use same-shaped tensor constants on both sides of every comparison and
    # select. This keeps BiSheng from lowering a tensor-scalar expression to a
    # narrow integer arithmetic op followed by an unsupported truncating cast.
    result = tl.where(
        code == tl.full(shape, 1, tl.int8),
        tl.full(shape, 1, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 2, tl.int8),
        tl.full(shape, 2, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 3, tl.int8),
        tl.full(shape, 3, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 4, tl.int8),
        tl.full(shape, 4, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 5, tl.int8),
        tl.full(shape, 6, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 6, tl.int8),
        tl.full(shape, 8, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 7, tl.int8),
        tl.full(shape, 12, tl.int8),
        result,
    )

    # Code 8 is negative zero and intentionally keeps the initial zero value.
    # Negative values are direct INT8 constants rather than unary negations, so
    # no computed narrow-integer conversion is introduced before Cube.
    result = tl.where(
        code == tl.full(shape, 9, tl.int8),
        tl.full(shape, -1, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 10, tl.int8),
        tl.full(shape, -2, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 11, tl.int8),
        tl.full(shape, -3, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 12, tl.int8),
        tl.full(shape, -4, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 13, tl.int8),
        tl.full(shape, -6, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 14, tl.int8),
        tl.full(shape, -8, tl.int8),
        result,
    )
    result = tl.where(
        code == tl.full(shape, 15, tl.int8),
        tl.full(shape, -12, tl.int8),
        result,
    )
    return result


@triton.jit
def _decode_e8m0(scale_code):
    """Decode an unsigned E8M0 scale code to a floating-point scale."""
    scale_code = scale_code.to(tl.int32)

    # For codes 1..254, the E8M0 code is exactly the FP32 exponent field.
    # Code 0 represents 2^-127, encoded as FP32 subnormal bit 22. Code 255
    # follows the existing exp2(128) behavior and decodes to infinity.
    is_min_scale = (scale_code == 0).to(tl.int32)
    scale_bits = (
        scale_code << FP32_EXPONENT_SHIFT
    ) | (is_min_scale * FP32_MIN_E8M0_SCALE_BITS)
    return scale_bits.to(tl.float32, bitcast=True)


@triton.jit
def _decode_mxfp4(code, scale_code):
    """Decode E2M1 values and apply their shared E8M0 block scale."""
    return _decode_e2m1(code) * _decode_e8m0(scale_code)


@triton.jit
def _encode_e2m1(value):
    """Encode normalized floating-point values as E2M1 bit patterns."""
    value = value.to(tl.float32)

    sign_bit = (value < 0.0).to(tl.int32)
    magnitude = tl.abs(value)

    # Select the nearest value from 0, 0.5, 1, 1.5, 2, 3, 4, and 6.
    # Alternating > and >= implements round-to-nearest, ties-to-even.
    magnitude_code = tl.where(magnitude > 0.25, 1, 0)
    magnitude_code = tl.where(magnitude >= 0.75, 2, magnitude_code)
    magnitude_code = tl.where(magnitude > 1.25, 3, magnitude_code)
    magnitude_code = tl.where(magnitude >= 1.75, 4, magnitude_code)
    magnitude_code = tl.where(magnitude > 2.5, 5, magnitude_code)
    magnitude_code = tl.where(magnitude >= 3.5, 6, magnitude_code)
    magnitude_code = tl.where(magnitude > 5.0, 7, magnitude_code)

    return (sign_bit << 3) | magnitude_code


@triton.jit
def _fp32_to_mxfp4_kernel(
    input_ptr,
    packed_ptr,
    scales_ptr,
    n_groups,
    GROUP_SIZE: tl.constexpr,
):
    """Quantize independent groups of FP32 values to MXFP4."""
    program_id = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    for group_id in range(program_id, n_groups, num_programs):
        pair_offsets = tl.arange(0, GROUP_SIZE // 2)
        group_start = group_id * GROUP_SIZE
        low_values = tl.load(input_ptr + group_start + pair_offsets * 2).to(
            tl.float32
        )
        high_values = tl.load(
            input_ptr + group_start + pair_offsets * 2 + 1
        ).to(tl.float32)

        low_amax = tl.max(tl.abs(low_values), axis=0)
        high_amax = tl.max(tl.abs(high_values), axis=0)
        amax = tl.maximum(low_amax, high_amax)

        # Keep scale calculation in the normal FP32 range. The largest E2M1
        # magnitude is 6, so ceil(log2(amax / 6)) prevents overflow.
        safe_amax = tl.maximum(amax, E2M1_MAX_VALUE * FP32_MIN_NORMAL)
        scale_exponent = tl.ceil(tl.log2(safe_amax / E2M1_MAX_VALUE))
        scale_exponent = tl.maximum(scale_exponent, E8M0_MIN_EXPONENT)
        scale_exponent = tl.minimum(scale_exponent, E8M0_MAX_EXPONENT)
        scale = tl.exp2(scale_exponent)

        low_codes = _encode_e2m1(low_values / scale)
        high_codes = _encode_e2m1(high_values / scale)
        packed_codes = low_codes | (high_codes << 4)

        packed_group_start = group_id * (GROUP_SIZE // 2)
        tl.store(
            packed_ptr + packed_group_start + pair_offsets,
            packed_codes.to(tl.uint8),
        )
        scale_code = (scale_exponent + E8M0_EXPONENT_BIAS).to(tl.uint8)
        tl.store(scales_ptr + group_id, scale_code)


@triton.jit
def _mxfp4_to_fp32_kernel(
    packed_ptr,
    scales_ptr,
    output_ptr,
    n_packed_elements,
    n_tiles,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Unpack pairs of E2M1 values and apply their E8M0 block scales."""
    program_id = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    for tile_id in range(program_id, n_tiles, num_programs):
        packed_offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        packed_mask = packed_offsets < n_packed_elements
        packed = tl.load(
            packed_ptr + packed_offsets, mask=packed_mask, other=0
        ).to(tl.int32)

        low_codes = packed & 0xF
        high_codes = (packed >> 4) & 0xF

        packed_elements_per_group = GROUP_SIZE // 2
        scale_offsets = packed_offsets // packed_elements_per_group
        scale_codes = tl.load(
            scales_ptr + scale_offsets,
            mask=packed_mask,
            other=E8M0_EXPONENT_BIAS,
        )
        low_values = _decode_mxfp4(low_codes, scale_codes)
        high_values = _decode_mxfp4(high_codes, scale_codes)

        output_offsets = packed_offsets * 2
        tl.store(output_ptr + output_offsets, low_values, mask=packed_mask)
        tl.store(output_ptr + output_offsets + 1, high_values, mask=packed_mask)


def fp32_to_mxfp4(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous last dimension to group-32 MXFP4."""
    if input_tensor.dtype != torch.float32:
        raise TypeError(
            "input_tensor must have dtype torch.float32, "
            f"but got {input_tensor.dtype}"
        )
    if input_tensor.ndim == 0:
        raise ValueError("input_tensor must have at least one dimension")
    if input_tensor.shape[-1] % MXFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"input_tensor.shape[-1] must be divisible by {MXFP4_GROUP_SIZE}, "
            f"but got {input_tensor.shape[-1]}"
        )

    input_tensor = input_tensor.contiguous()
    packed_shape = (*input_tensor.shape[:-1], input_tensor.shape[-1] // 2)
    scales_shape = (
        *input_tensor.shape[:-1],
        input_tensor.shape[-1] // MXFP4_GROUP_SIZE,
    )
    packed = torch.empty(packed_shape, dtype=torch.uint8, device=input_tensor.device)
    scales = torch.empty(scales_shape, dtype=torch.uint8, device=input_tensor.device)
    if input_tensor.numel() == 0:
        return packed, scales

    n_groups = input_tensor.numel() // MXFP4_GROUP_SIZE
    num_programs = min(n_groups, get_vectorcore_num())
    _fp32_to_mxfp4_kernel[(num_programs,)](
        input_tensor,
        packed,
        scales,
        n_groups,
        GROUP_SIZE=MXFP4_GROUP_SIZE,
    )
    return packed, scales


def mxfp4_to_fp32(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize group-32 MXFP4 data to a contiguous FP32 tensor."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must have dtype torch.uint8, but got {packed.dtype}")
    if scales.dtype != torch.uint8:
        raise TypeError(f"scales must have dtype torch.uint8, but got {scales.dtype}")
    if packed.device != scales.device:
        raise ValueError("packed and scales must be on the same device")
    if packed.ndim == 0 or scales.ndim == 0:
        raise ValueError("packed and scales must have at least one dimension")
    if packed.shape[:-1] != scales.shape[:-1]:
        raise ValueError("packed and scales must have identical leading dimensions")

    packed_elements_per_group = MXFP4_GROUP_SIZE // 2
    if packed.shape[-1] % packed_elements_per_group != 0:
        raise ValueError(
            f"packed.shape[-1] must be divisible by {packed_elements_per_group}, "
            f"but got {packed.shape[-1]}"
        )

    expected_scale_count = packed.shape[-1] // packed_elements_per_group
    if scales.shape[-1] != expected_scale_count:
        raise ValueError(
            f"scales.shape[-1] must be {expected_scale_count}, "
            f"but got {scales.shape[-1]}"
        )

    packed = packed.contiguous()
    scales = scales.contiguous()
    output_shape = (*packed.shape[:-1], packed.shape[-1] * 2)
    output = torch.empty(output_shape, dtype=torch.float32, device=packed.device)
    if packed.numel() == 0:
        return output

    n_packed_elements = packed.numel()
    n_tiles = (
        n_packed_elements + MXFP4_UNPACK_BLOCK_SIZE - 1
    ) // MXFP4_UNPACK_BLOCK_SIZE
    num_programs = min(n_tiles, get_vectorcore_num())

    _mxfp4_to_fp32_kernel[(num_programs,)](
        packed,
        scales,
        output,
        n_packed_elements,
        n_tiles,
        GROUP_SIZE=MXFP4_GROUP_SIZE,
        BLOCK_SIZE=MXFP4_UNPACK_BLOCK_SIZE,
    )
    return output
