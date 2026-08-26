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

"""Triton Ascend MXFP4 query-key matmul for the Lightning Indexer."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.mxfp4 import (
    E8M0_EXPONENT_BIAS,
    MXFP4_GROUP_SIZE,
)

INDEXER_NUM_HEADS: tl.constexpr = 64
INDEXER_HEAD_DIM: tl.constexpr = 128
INDEXER_PACKED_HEAD_DIM: tl.constexpr = INDEXER_HEAD_DIM // 2
INDEXER_NUM_SCALE_GROUPS: tl.constexpr = INDEXER_HEAD_DIM // MXFP4_GROUP_SIZE
INDEXER_QK_HEAD_BLOCK_SIZE: tl.constexpr = 32
INDEXER_QK_KEY_BLOCK_SIZE: tl.constexpr = 64
INDEXER_COMPRESS_RATIO: tl.constexpr = 4
COMBINED_E8M0_EXPONENT_BIAS: tl.constexpr = 254
FP32_MIN_NORMAL_EXPONENT: tl.constexpr = -126
FP32_MAX_NORMAL_EXPONENT: tl.constexpr = 127
UINT8_CODE_MASK: tl.constexpr = 0xFF
E2M1_HALF_UNIT_LUT_VALUES = (
    0,
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    0,
    -1,
    -2,
    -3,
    -4,
    -6,
    -8,
    -12,
)


@triton.jit
def _indexer_query_metadata_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    request_indices_ptr,
    valid_key_counts_ptr,
    num_query_tokens,
    num_requests,
    max_num_keys,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Map each packed TND query row to its request and causal key count."""
    query_token_index = tl.program_id(axis=0)
    request_offsets = tl.arange(0, BLOCK_B)
    request_mask = request_offsets < num_requests
    query_ends = tl.load(
        query_start_loc_ptr + request_offsets + 1,
        mask=request_mask,
        other=0x7FFFFFFF,
    ).to(tl.int32)
    request_index = tl.sum(
        (query_token_index >= query_ends).to(tl.int32),
        axis=0,
    )

    query_start = tl.load(
        query_start_loc_ptr + request_index,
    ).to(tl.int32)
    query_end = tl.load(
        query_start_loc_ptr + request_index + 1,
    ).to(tl.int32)
    sequence_length = tl.load(
        seq_lens_ptr + request_index,
    ).to(tl.int32)
    query_length = query_end - query_start
    local_query_index = query_token_index - query_start
    # sparse_mode=3 is right-down causal. The C8 metadata kernel computes the
    # same count by dividing the causally visible original-token range by 4.
    visible_original_tokens = (
        sequence_length - query_length + local_query_index + 1
    )
    valid_key_count = visible_original_tokens // COMPRESS_RATIO
    valid_key_count = tl.maximum(
        0,
        tl.minimum(valid_key_count, max_num_keys),
    )
    token_mask = query_token_index < num_query_tokens
    tl.store(
        request_indices_ptr + query_token_index,
        request_index,
        mask=token_mask,
    )
    tl.store(
        valid_key_counts_ptr + query_token_index,
        valid_key_count,
        mask=token_mask,
    )


@triton.jit
def _mxfp4_indexer_paged_qk_kernel(
    q_packed_ptr,
    q_scales_ptr,
    k_cache_ptr,
    scale_cache_ptr,
    block_table_ptr,
    request_indices_ptr,
    valid_key_counts_ptr,
    e2m1_lut_ptr,
    output_ptr,
    max_num_keys,
    max_num_blocks,
    cache_block_size,
    key_page_stride,
    key_token_stride,
    key_dim_stride,
    scale_page_stride,
    scale_token_stride,
    scale_dim_stride,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SCALE_BIAS: tl.constexpr,
    COMBINED_SCALE_BIAS: tl.constexpr,
    MIN_NORMAL_EXPONENT: tl.constexpr,
    MAX_NORMAL_EXPONENT: tl.constexpr,
    SCALE_CODE_MASK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute one query/head/key tile from a paged MXFP4 key cache."""
    query_token_index = tl.program_id(axis=0)
    head_tile_index = tl.program_id(axis=1)
    key_tile_index = tl.program_id(axis=2)

    head_offsets = head_tile_index * BLOCK_H + tl.arange(0, BLOCK_H)
    key_offsets = key_tile_index * BLOCK_N + tl.arange(0, BLOCK_N)
    output_key_mask = key_offsets < max_num_keys
    request_index = tl.load(
        request_indices_ptr + query_token_index,
    ).to(tl.int32)
    valid_key_count = tl.load(
        valid_key_counts_ptr + query_token_index,
    ).to(tl.int32)
    key_mask = output_key_mask & (key_offsets < valid_key_count)
    safe_key_offsets = tl.where(
        key_mask,
        key_offsets,
        key_offsets ^ key_offsets,
    )

    logical_block_offsets = safe_key_offsets // cache_block_size
    block_table_mask = key_mask & (logical_block_offsets < max_num_blocks)
    physical_block_offsets = tl.load(
        block_table_ptr
        + request_index * max_num_blocks
        + logical_block_offsets,
        mask=block_table_mask,
        other=0,
    ).to(tl.int32)
    token_offsets = safe_key_offsets % cache_block_size
    group_dim_offsets = tl.arange(0, GROUP_SIZE)

    scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
    for group_index in range(0, NUM_GROUPS):
        packed_group_offset = group_index * (GROUP_SIZE // 2)
        q_byte_offsets = (
            query_token_index * INDEXER_NUM_HEADS * INDEXER_PACKED_HEAD_DIM
            + head_offsets[:, None] * INDEXER_PACKED_HEAD_DIM
            + packed_group_offset
            + group_dim_offsets[None, :] // 2
        )
        q_packed = tl.load(q_packed_ptr + q_byte_offsets)
        q_codes = tl.where(
            (group_dim_offsets[None, :] & 1) == 0,
            q_packed & 0xF,
            (q_packed >> 4) & 0xF,
        )
        q_half_units = tl.load(e2m1_lut_ptr + q_codes)

        q_scale_offsets = (
            query_token_index * INDEXER_NUM_HEADS * NUM_GROUPS
            + head_offsets * NUM_GROUPS
            + group_index
        )
        q_scale_code_mask = tl.full(
            (BLOCK_H,),
            SCALE_CODE_MASK,
            tl.int32,
        )
        q_scale_codes = (
            tl.load(q_scales_ptr + q_scale_offsets).to(tl.int32)
            & q_scale_code_mask
        )

        k_byte_offsets = (
            physical_block_offsets[:, None] * key_page_stride
            + token_offsets[:, None] * key_token_stride
            + (
                packed_group_offset
                + group_dim_offsets[None, :] // 2
            ) * key_dim_stride
        )
        k_packed = tl.load(
            k_cache_ptr + k_byte_offsets,
            mask=key_mask[:, None],
            other=0,
        )
        k_codes = tl.where(
            (group_dim_offsets[None, :] & 1) == 0,
            k_packed & 0xF,
            (k_packed >> 4) & 0xF,
        )
        k_codes = tl.where(
            key_mask[:, None],
            k_codes,
            k_codes ^ k_codes,
        )
        k_half_units = tl.load(e2m1_lut_ptr + k_codes)

        k_scale_offsets = (
            physical_block_offsets * scale_page_stride
            + token_offsets * scale_token_stride
            + group_index * scale_dim_stride
        )
        k_scale_code_mask = tl.full(
            (BLOCK_N,),
            SCALE_CODE_MASK,
            tl.int32,
        )
        k_scale_codes = (
            tl.load(
                scale_cache_ptr + k_scale_offsets,
                mask=key_mask,
                other=SCALE_BIAS,
            ).to(tl.int32)
            & k_scale_code_mask
        )

        group_scores_quarter_units = tl.dot(
            q_half_units,
            tl.trans(k_half_units),
        )
        group_scores = group_scores_quarter_units.to(tl.float32) * 0.25

        scale_shape: tl.constexpr = (BLOCK_H, BLOCK_N)
        combined_scale_bias = tl.full(
            scale_shape,
            COMBINED_SCALE_BIAS,
            tl.int32,
        )
        min_normal_exponent = tl.full(
            scale_shape,
            MIN_NORMAL_EXPONENT,
            tl.int32,
        )
        max_normal_exponent = tl.full(
            scale_shape,
            MAX_NORMAL_EXPONENT,
            tl.int32,
        )
        combined_exponent = (
            q_scale_codes[:, None]
            + k_scale_codes[None, :]
            - combined_scale_bias
        )
        first_exponent = tl.minimum(
            tl.maximum(combined_exponent, min_normal_exponent),
            max_normal_exponent,
        )
        remaining_exponent = combined_exponent - first_exponent
        second_exponent = tl.minimum(
            tl.maximum(remaining_exponent, min_normal_exponent),
            max_normal_exponent,
        )
        third_exponent = remaining_exponent - second_exponent

        group_scores *= tl.exp2(first_exponent.to(tl.float32))
        group_scores *= tl.exp2(second_exponent.to(tl.float32))
        group_scores *= tl.exp2(third_exponent.to(tl.float32))
        scores += group_scores

    output_offsets = (
        query_token_index * INDEXER_NUM_HEADS * max_num_keys
        + head_offsets[:, None] * max_num_keys
        + key_offsets[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        scores,
        mask=output_key_mask[None, :],
    )


def create_e2m1_half_unit_lut(
    device: torch.device | str,
) -> torch.Tensor:
    """Create the 16-byte E2M1 lookup table used by the QK kernel.

    The caller should create this tensor once per device and keep it alive for
    repeated calls, for example by registering it as a non-persistent module
    buffer. Each value is exactly twice the corresponding E2M1 value.
    """
    return torch.tensor(
        E2M1_HALF_UNIT_LUT_VALUES,
        dtype=torch.int8,
        device=device,
    )


@triton.jit
def _mxfp4_indexer_qk_kernel(
    q_packed_ptr,
    q_scales_ptr,
    k_packed_ptr,
    k_scales_ptr,
    e2m1_lut_ptr,
    output_ptr,
    num_key_tokens,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SCALE_BIAS: tl.constexpr,
    COMBINED_SCALE_BIAS: tl.constexpr,
    MIN_NORMAL_EXPONENT: tl.constexpr,
    MAX_NORMAL_EXPONENT: tl.constexpr,
    SCALE_CODE_MASK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute one query-head tile against one tile of shared MXFP4 keys."""
    query_token_index = tl.program_id(axis=0)
    head_tile_index = tl.program_id(axis=1)
    key_tile_index = tl.program_id(axis=2)

    head_offsets = head_tile_index * BLOCK_H + tl.arange(0, BLOCK_H)
    key_offsets = key_tile_index * BLOCK_N + tl.arange(0, BLOCK_N)
    key_mask = key_offsets < num_key_tokens
    safe_key_offsets = tl.where(
        key_mask,
        key_offsets,
        key_offsets ^ key_offsets,
    )
    group_dim_offsets = tl.arange(0, GROUP_SIZE)

    # Accumulate the four 32-element MXFP4 scale groups independently. Cube
    # receives an exact signed half-unit representation of E2M1. Its FP32
    # result is divided by four because both operands were multiplied by two.
    # The two E8M0 block scales are then applied to each group's dot result.
    scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
    for group_index in range(0, NUM_GROUPS):
        packed_group_offset = group_index * (GROUP_SIZE // 2)

        q_byte_offsets = (
            query_token_index * INDEXER_NUM_HEADS * INDEXER_PACKED_HEAD_DIM
            + head_offsets[:, None] * INDEXER_PACKED_HEAD_DIM
            + packed_group_offset
            + group_dim_offsets[None, :] // 2
        )
        q_packed = tl.load(q_packed_ptr + q_byte_offsets)
        q_codes = tl.where(
            (group_dim_offsets[None, :] & 1) == 0,
            q_packed & 0xF,
            (q_packed >> 4) & 0xF,
        )
        q_scale_offsets = (
            query_token_index * INDEXER_NUM_HEADS * NUM_GROUPS
            + head_offsets * NUM_GROUPS
            + group_index
        )
        # The wrapper presents uint8 E8M0 storage as int8 because A5 may
        # sign-extend a direct uint8-to-int32 conversion. Masking the widened
        # signed byte recovers its exact unsigned code, e.g. -2 & 255 == 254.
        q_scale_code_mask = tl.full(
            (BLOCK_H,),
            SCALE_CODE_MASK,
            tl.int32,
        )
        q_scale_codes = (
            tl.load(q_scales_ptr + q_scale_offsets).to(tl.int32)
            & q_scale_code_mask
        )
        # The A5 backend cannot infer a valid Cube layout when a vsel result is
        # consumed directly by tl.dot. Loading the exact signed half-unit value
        # from a 16-byte device LUT creates a normal load-backed Cube operand.
        q_half_units = tl.load(e2m1_lut_ptr + q_codes)

        k_byte_offsets = (
            safe_key_offsets[:, None] * INDEXER_PACKED_HEAD_DIM
            + packed_group_offset
            + group_dim_offsets[None, :] // 2
        )
        k_packed = tl.load(
            k_packed_ptr + k_byte_offsets,
            mask=key_mask[:, None],
            other=0,
        )
        k_codes = tl.where(
            (group_dim_offsets[None, :] & 1) == 0,
            k_packed & 0xF,
            (k_packed >> 4) & 0xF,
        )
        # Ascend masked loads are not sufficient by themselves for values
        # consumed by Cube, so explicitly zero every invalid key lane.
        k_codes = tl.where(
            key_mask[:, None],
            k_codes,
            k_codes ^ k_codes,
        )
        k_scale_offsets = (
            safe_key_offsets * NUM_GROUPS + group_index
        )
        k_scale_code_mask = tl.full(
            (BLOCK_N,),
            SCALE_CODE_MASK,
            tl.int32,
        )
        k_scale_codes = (
            tl.load(
                k_scales_ptr + k_scale_offsets,
                mask=key_mask,
                other=SCALE_BIAS,
            ).to(tl.int32)
            & k_scale_code_mask
        )
        k_half_units = tl.load(e2m1_lut_ptr + k_codes)

        # INT8 Cube operands produce an INT32 dot result. This cast is after
        # tl.dot, so it consumes a real Cube output allocation rather than
        # creating the unsupported computed-cast input that triggered the A5
        # root-allocation failure. The largest group sum is only 4608 and is
        # therefore represented exactly in both INT32 and FP32.
        group_scores_quarter_units = tl.dot(
            q_half_units,
            tl.trans(k_half_units),
        )
        group_scores = group_scores_quarter_units.to(tl.float32) * 0.25

        # E8M0 multiplication is exponent addition:
        #   2 ** (q_code - 127) * 2 ** (k_code - 127)
        #     = 2 ** (q_code + k_code - 254).
        # Apply that combined exponent in finite normal-FP32 chunks. This
        # avoids both the unsupported integer shift/bitcast lowering and the
        # 0 * inf / intermediate-subnormal problems of materializing the two
        # scales independently.
        scale_shape: tl.constexpr = (BLOCK_H, BLOCK_N)
        combined_scale_bias = tl.full(
            scale_shape,
            COMBINED_SCALE_BIAS,
            tl.int32,
        )
        min_normal_exponent = tl.full(
            scale_shape,
            MIN_NORMAL_EXPONENT,
            tl.int32,
        )
        max_normal_exponent = tl.full(
            scale_shape,
            MAX_NORMAL_EXPONENT,
            tl.int32,
        )
        combined_exponent = (
            q_scale_codes[:, None]
            + k_scale_codes[None, :]
            - combined_scale_bias
        )
        first_exponent = tl.minimum(
            tl.maximum(combined_exponent, min_normal_exponent),
            max_normal_exponent,
        )
        remaining_exponent = combined_exponent - first_exponent
        second_exponent = tl.minimum(
            tl.maximum(remaining_exponent, min_normal_exponent),
            max_normal_exponent,
        )
        third_exponent = remaining_exponent - second_exponent

        group_scores *= tl.exp2(first_exponent.to(tl.float32))
        group_scores *= tl.exp2(second_exponent.to(tl.float32))
        group_scores *= tl.exp2(third_exponent.to(tl.float32))
        scores += group_scores

    output_offsets = (
        query_token_index * INDEXER_NUM_HEADS * num_key_tokens
        + head_offsets[:, None] * num_key_tokens
        + safe_key_offsets[None, :]
    )
    tl.store(output_ptr + output_offsets, scores, mask=key_mask[None, :])


def mxfp4_indexer_qk_matmul(
    q_packed: torch.Tensor,
    q_scales: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    e2m1_lut: torch.Tensor,
) -> torch.Tensor:
    """Compute raw per-head Lightning Indexer scores from MXFP4 Q and K.

    Args:
        q_packed: Packed query with shape ``[T, 64, 64]`` and dtype uint8.
        q_scales: Query E8M0 scales with shape ``[T, 64, 4]``.
        k_packed: Packed shared key with shape ``[S, 64]`` or
            ``[S, 1, 64]`` and dtype uint8.
        k_scales: Key E8M0 scales with shape ``[S, 4]`` or
            ``[S, 1, 4]``.
        e2m1_lut: A contiguous INT8 tensor with 16 entries, created once with
            :func:`create_e2m1_half_unit_lut` on the input device.

    Returns:
        A float32 tensor with shape ``[T, 64, S]``.
    """
    uint8_tensors = {
        "q_packed": q_packed,
        "q_scales": q_scales,
        "k_packed": k_packed,
        "k_scales": k_scales,
    }
    for name, tensor in uint8_tensors.items():
        if tensor.dtype != torch.uint8:
            raise TypeError(
                f"{name} must have dtype torch.uint8, but got {tensor.dtype}"
            )
    if e2m1_lut.dtype != torch.int8:
        raise TypeError(
            "e2m1_lut must have dtype torch.int8, "
            f"but got {e2m1_lut.dtype}"
        )
    if tuple(e2m1_lut.shape) != (len(E2M1_HALF_UNIT_LUT_VALUES),):
        raise ValueError(
            "e2m1_lut must have shape "
            f"({len(E2M1_HALF_UNIT_LUT_VALUES)},), "
            f"but got {tuple(e2m1_lut.shape)}"
        )

    tensors = (*uint8_tensors.values(), e2m1_lut)
    input_devices = {tensor.device for tensor in tensors}
    if len(input_devices) != 1:
        raise ValueError(
            "q_packed, q_scales, k_packed, k_scales, and e2m1_lut "
            "must be on the same device"
        )

    expected_q_tail = (INDEXER_NUM_HEADS, INDEXER_PACKED_HEAD_DIM)
    if q_packed.ndim != 3 or tuple(q_packed.shape[1:]) != expected_q_tail:
        raise ValueError(
            "q_packed must have shape "
            f"[T, {INDEXER_NUM_HEADS}, {INDEXER_PACKED_HEAD_DIM}], "
            f"but got {tuple(q_packed.shape)}"
        )

    num_query_tokens = q_packed.shape[0]
    expected_q_scales_shape = (
        num_query_tokens,
        INDEXER_NUM_HEADS,
        INDEXER_NUM_SCALE_GROUPS,
    )
    if tuple(q_scales.shape) != expected_q_scales_shape:
        raise ValueError(
            f"q_scales must have shape {expected_q_scales_shape}, "
            f"but got {tuple(q_scales.shape)}"
        )

    if k_packed.ndim == 3:
        if k_packed.shape[1] != 1:
            raise ValueError(
                "k_packed must contain exactly one shared key head, "
                f"but got shape {tuple(k_packed.shape)}"
            )
        k_packed = k_packed[:, 0, :]
    elif k_packed.ndim != 2:
        raise ValueError(
            "k_packed must have shape [S, 64] or [S, 1, 64], "
            f"but got {tuple(k_packed.shape)}"
        )

    if k_scales.ndim == 3:
        if k_scales.shape[1] != 1:
            raise ValueError(
                "k_scales must contain exactly one shared key head, "
                f"but got shape {tuple(k_scales.shape)}"
            )
        k_scales = k_scales[:, 0, :]
    elif k_scales.ndim != 2:
        raise ValueError(
            "k_scales must have shape [S, 4] or [S, 1, 4], "
            f"but got {tuple(k_scales.shape)}"
        )

    if k_packed.shape[1] != INDEXER_PACKED_HEAD_DIM:
        raise ValueError(
            f"k_packed.shape[-1] must be {INDEXER_PACKED_HEAD_DIM}, "
            f"but got {k_packed.shape[1]}"
        )

    num_key_tokens = k_packed.shape[0]
    expected_k_scales_shape = (num_key_tokens, INDEXER_NUM_SCALE_GROUPS)
    if tuple(k_scales.shape) != expected_k_scales_shape:
        raise ValueError(
            f"k_scales must have shape {expected_k_scales_shape}, "
            f"but got {tuple(k_scales.shape)}"
        )

    q_packed = q_packed.contiguous()
    q_scales = q_scales.contiguous()
    k_packed = k_packed.contiguous()
    k_scales = k_scales.contiguous()
    e2m1_lut = e2m1_lut.contiguous()

    # Keep the public/cache format as packed uint8, but present the same bytes
    # to Triton as signed int8. This is a metadata-only view and avoids the A5
    # uint8-to-uint16 conversion generated by the two-dimensional nibble path.
    # Arithmetic right shift is safe here because the final ``& 0xF`` recovers
    # the original high nibble for both positive and negative int8 bytes.
    q_packed_int8 = q_packed.view(torch.int8)
    q_scales_int8 = q_scales.view(torch.int8)
    k_packed_int8 = k_packed.view(torch.int8)
    k_scales_int8 = k_scales.view(torch.int8)

    output = torch.empty(
        (num_query_tokens, INDEXER_NUM_HEADS, num_key_tokens),
        dtype=torch.float32,
        device=q_packed.device,
    )
    if num_query_tokens == 0 or num_key_tokens == 0:
        return output

    grid = (
        num_query_tokens,
        triton.cdiv(INDEXER_NUM_HEADS, INDEXER_QK_HEAD_BLOCK_SIZE),
        triton.cdiv(num_key_tokens, INDEXER_QK_KEY_BLOCK_SIZE),
    )
    _mxfp4_indexer_qk_kernel[grid](
        q_packed_int8,
        q_scales_int8,
        k_packed_int8,
        k_scales_int8,
        e2m1_lut,
        output,
        num_key_tokens,
        GROUP_SIZE=MXFP4_GROUP_SIZE,
        NUM_GROUPS=INDEXER_NUM_SCALE_GROUPS,
        SCALE_BIAS=E8M0_EXPONENT_BIAS,
        COMBINED_SCALE_BIAS=COMBINED_E8M0_EXPONENT_BIAS,
        MIN_NORMAL_EXPONENT=FP32_MIN_NORMAL_EXPONENT,
        MAX_NORMAL_EXPONENT=FP32_MAX_NORMAL_EXPONENT,
        SCALE_CODE_MASK=UINT8_CODE_MASK,
        BLOCK_H=INDEXER_QK_HEAD_BLOCK_SIZE,
        BLOCK_N=INDEXER_QK_KEY_BLOCK_SIZE,
    )
    return output


def mxfp4_indexer_paged_qk_matmul(
    q_packed: torch.Tensor,
    q_scales: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    e2m1_lut: torch.Tensor,
    compress_ratio: int = INDEXER_COMPRESS_RATIO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score TND MXFP4 queries against a paged MXFP4 Indexer cache.

    ``seq_lens`` remains expressed in original-token units, matching the
    existing Lightning Indexer interface. The returned valid counts and top-k
    index domain are compressed-key positions.

    Returns:
        ``(qk_scores, valid_key_counts)`` where scores have shape
        ``[T, 64, max(1, max_seq_len // compress_ratio)]`` and valid counts
        have shape ``[T]`` with dtype int32.
    """
    if compress_ratio != INDEXER_COMPRESS_RATIO:
        raise ValueError(
            f"compress_ratio must be {INDEXER_COMPRESS_RATIO}, "
            f"but got {compress_ratio}"
        )
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
        raise TypeError("max_seq_len must be an int")
    if max_seq_len < 0:
        raise ValueError("max_seq_len must be non-negative")

    uint8_tensors = {
        "q_packed": q_packed,
        "q_scales": q_scales,
        "k_cache": k_cache,
        "scale_cache": scale_cache,
    }
    for name, tensor in uint8_tensors.items():
        if tensor.dtype != torch.uint8:
            raise TypeError(
                f"{name} must have dtype torch.uint8, but got {tensor.dtype}"
            )
    int32_tensors = {
        "block_table": block_table,
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
    }
    for name, tensor in int32_tensors.items():
        if tensor.dtype != torch.int32:
            raise TypeError(
                f"{name} must have dtype torch.int32, but got {tensor.dtype}"
            )
    if e2m1_lut.dtype != torch.int8:
        raise TypeError("e2m1_lut must have dtype torch.int8")

    num_query_tokens = q_packed.shape[0]
    if tuple(q_packed.shape) != (
        num_query_tokens,
        INDEXER_NUM_HEADS,
        INDEXER_PACKED_HEAD_DIM,
    ):
        raise ValueError(
            "q_packed must have shape [T, 64, 64], "
            f"but got {tuple(q_packed.shape)}"
        )
    if tuple(q_scales.shape) != (
        num_query_tokens,
        INDEXER_NUM_HEADS,
        INDEXER_NUM_SCALE_GROUPS,
    ):
        raise ValueError(
            "q_scales must have shape [T, 64, 4], "
            f"but got {tuple(q_scales.shape)}"
        )
    if k_cache.ndim != 4 or tuple(k_cache.shape[2:]) != (
        1,
        INDEXER_PACKED_HEAD_DIM,
    ):
        raise ValueError(
            "k_cache must have shape [P, B, 1, 64], "
            f"but got {tuple(k_cache.shape)}"
        )
    if scale_cache.ndim != 4 or tuple(scale_cache.shape[2:]) != (
        1,
        INDEXER_NUM_SCALE_GROUPS,
    ):
        raise ValueError(
            "scale_cache must have shape [P, B, 1, 4], "
            f"but got {tuple(scale_cache.shape)}"
        )
    if k_cache.shape[:2] != scale_cache.shape[:2]:
        raise ValueError("k_cache and scale_cache page geometry must match")
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [num_requests, max_blocks]")
    num_requests = block_table.shape[0]
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] != num_requests + 1:
        raise ValueError(
            "query_start_loc must have shape [num_requests + 1]"
        )
    if tuple(seq_lens.shape) != (num_requests,):
        raise ValueError("seq_lens must have shape [num_requests]")
    if tuple(e2m1_lut.shape) != (len(E2M1_HALF_UNIT_LUT_VALUES),):
        raise ValueError(
            f"e2m1_lut must have shape ({len(E2M1_HALF_UNIT_LUT_VALUES)},)"
        )
    if num_requests == 0 and num_query_tokens != 0:
        raise ValueError("non-empty queries require at least one request")

    tensors = (
        *uint8_tensors.values(),
        *int32_tensors.values(),
        e2m1_lut,
    )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all paged QK tensors must share a device")

    max_num_keys = max(1, max_seq_len // compress_ratio)
    output = torch.empty(
        (num_query_tokens, INDEXER_NUM_HEADS, max_num_keys),
        dtype=torch.float32,
        device=q_packed.device,
    )
    valid_key_counts = torch.empty(
        (num_query_tokens,),
        dtype=torch.int32,
        device=q_packed.device,
    )
    if num_query_tokens == 0:
        return output, valid_key_counts

    q_packed_int8 = q_packed.contiguous().view(torch.int8)
    q_scales_int8 = q_scales.contiguous().view(torch.int8)
    k_cache_int8 = k_cache.view(torch.int8)
    scale_cache_int8 = scale_cache.view(torch.int8)
    block_table = block_table.contiguous()
    query_start_loc = query_start_loc.contiguous()
    seq_lens = seq_lens.contiguous()
    e2m1_lut = e2m1_lut.contiguous()

    request_indices = torch.empty_like(valid_key_counts)
    block_b = triton.next_power_of_2(max(1, num_requests))
    _indexer_query_metadata_kernel[(num_query_tokens,)](
        query_start_loc,
        seq_lens,
        request_indices,
        valid_key_counts,
        num_query_tokens,
        num_requests,
        max_num_keys,
        COMPRESS_RATIO=INDEXER_COMPRESS_RATIO,
        BLOCK_B=block_b,
    )

    grid = (
        num_query_tokens,
        triton.cdiv(INDEXER_NUM_HEADS, INDEXER_QK_HEAD_BLOCK_SIZE),
        triton.cdiv(max_num_keys, INDEXER_QK_KEY_BLOCK_SIZE),
    )
    _mxfp4_indexer_paged_qk_kernel[grid](
        q_packed_int8,
        q_scales_int8,
        k_cache_int8,
        scale_cache_int8,
        block_table,
        request_indices,
        valid_key_counts,
        e2m1_lut,
        output,
        max_num_keys,
        block_table.shape[1],
        k_cache.shape[1],
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        scale_cache.stride(0),
        scale_cache.stride(1),
        scale_cache.stride(3),
        GROUP_SIZE=MXFP4_GROUP_SIZE,
        NUM_GROUPS=INDEXER_NUM_SCALE_GROUPS,
        SCALE_BIAS=E8M0_EXPONENT_BIAS,
        COMBINED_SCALE_BIAS=COMBINED_E8M0_EXPONENT_BIAS,
        MIN_NORMAL_EXPONENT=FP32_MIN_NORMAL_EXPONENT,
        MAX_NORMAL_EXPONENT=FP32_MAX_NORMAL_EXPONENT,
        SCALE_CODE_MASK=UINT8_CODE_MASK,
        BLOCK_H=INDEXER_QK_HEAD_BLOCK_SIZE,
        BLOCK_N=INDEXER_QK_KEY_BLOCK_SIZE,
    )
    return output, valid_key_counts
