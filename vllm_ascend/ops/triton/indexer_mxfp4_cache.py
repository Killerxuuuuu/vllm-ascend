# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Paged-cache scatter helpers for the MXFP4 DSV4 Indexer."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.indexer_mxfp4_qk import (
    INDEXER_NUM_SCALE_GROUPS,
    INDEXER_PACKED_HEAD_DIM,
)


@triton.jit
def _scatter_mxfp4_indexer_cache_kernel(
    packed_ptr,
    scales_ptr,
    key_cache_ptr,
    scale_cache_ptr,
    slot_mapping_ptr,
    num_tokens,
    cache_block_size,
    key_page_stride,
    key_token_stride,
    key_dim_stride,
    scale_page_stride,
    scale_token_stride,
    scale_dim_stride,
    PACKED_DIM: tl.constexpr,
    NUM_SCALE_GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_index = tl.program_id(axis=0)
    dim_offsets = tl.arange(0, BLOCK_D)
    token_mask = token_index < num_tokens
    physical_slot = tl.load(
        slot_mapping_ptr + token_index,
        mask=token_mask,
        other=-1,
    ).to(tl.int32)
    write_mask = token_mask & (physical_slot >= 0)

    safe_slot = tl.where(
        write_mask,
        physical_slot,
        physical_slot ^ physical_slot,
    )
    page_index = safe_slot // cache_block_size
    token_offset = safe_slot % cache_block_size

    packed = tl.load(
        packed_ptr + token_index * PACKED_DIM + dim_offsets,
        mask=write_mask & (dim_offsets < PACKED_DIM),
        other=0,
    )
    key_cache_offsets = (
        page_index * key_page_stride
        + token_offset * key_token_stride
        + dim_offsets * key_dim_stride
    )
    tl.store(
        key_cache_ptr + key_cache_offsets,
        packed,
        mask=write_mask & (dim_offsets < PACKED_DIM),
    )

    scales = tl.load(
        scales_ptr + token_index * NUM_SCALE_GROUPS + dim_offsets,
        mask=write_mask & (dim_offsets < NUM_SCALE_GROUPS),
        other=0,
    )
    scale_cache_offsets = (
        page_index * scale_page_stride
        + token_offset * scale_token_stride
        + dim_offsets * scale_dim_stride
    )
    tl.store(
        scale_cache_ptr + scale_cache_offsets,
        scales,
        mask=write_mask & (dim_offsets < NUM_SCALE_GROUPS),
    )


def scatter_mxfp4_indexer_cache(
    packed: torch.Tensor,
    scales: torch.Tensor,
    key_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter contiguous MXFP4 keys and E8M0 scales into a paged cache.

    ``slot_mapping`` contains flattened physical cache slots. Negative slots
    are padding and are ignored. The cache may contain per-page padding, so
    the kernel uses the real tensor strides instead of flattening the views.
    """
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("packed and scales must have dtype torch.uint8")
    if key_cache.dtype != torch.uint8 or scale_cache.dtype != torch.uint8:
        raise TypeError("MXFP4 indexer caches must have dtype torch.uint8")
    if slot_mapping.dtype != torch.int32:
        raise TypeError("slot_mapping must have dtype torch.int32")
    if packed.ndim not in (2, 3) or packed.shape[-1] != INDEXER_PACKED_HEAD_DIM:
        raise ValueError(
            "packed must have shape [T, 64] or [T, 1, 64], "
            f"but got {tuple(packed.shape)}"
        )
    if packed.ndim == 3 and packed.shape[1] != 1:
        raise ValueError("packed must contain exactly one shared key head")
    expected_scale_shape = (*packed.shape[:-1], INDEXER_NUM_SCALE_GROUPS)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"scales must have shape {expected_scale_shape}, "
            f"but got {tuple(scales.shape)}"
        )
    if key_cache.ndim != 4 or tuple(key_cache.shape[2:]) != (
        1,
        INDEXER_PACKED_HEAD_DIM,
    ):
        raise ValueError(
            "key_cache must have shape [P, B, 1, 64], "
            f"but got {tuple(key_cache.shape)}"
        )
    if scale_cache.ndim != 4 or tuple(scale_cache.shape[2:]) != (
        1,
        INDEXER_NUM_SCALE_GROUPS,
    ):
        raise ValueError(
            "scale_cache must have shape [P, B, 1, 4], "
            f"but got {tuple(scale_cache.shape)}"
        )
    if key_cache.shape[:2] != scale_cache.shape[:2]:
        raise ValueError("key_cache and scale_cache page geometry must match")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != packed.shape[0]:
        raise ValueError(
            "slot_mapping must have shape [T] matching packed.shape[0]"
        )
    tensors = (packed, scales, key_cache, scale_cache, slot_mapping)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all MXFP4 cache scatter tensors must share a device")

    num_tokens = packed.shape[0]
    if num_tokens == 0:
        return
    packed = packed.contiguous().view(num_tokens, INDEXER_PACKED_HEAD_DIM)
    scales = scales.contiguous().view(num_tokens, INDEXER_NUM_SCALE_GROUPS)
    slot_mapping = slot_mapping.contiguous()
    _scatter_mxfp4_indexer_cache_kernel[(num_tokens,)](
        packed,
        scales,
        key_cache,
        scale_cache,
        slot_mapping,
        num_tokens,
        key_cache.shape[1],
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(3),
        scale_cache.stride(0),
        scale_cache.stride(1),
        scale_cache.stride(3),
        PACKED_DIM=INDEXER_PACKED_HEAD_DIM,
        NUM_SCALE_GROUPS=INDEXER_NUM_SCALE_GROUPS,
        BLOCK_D=INDEXER_PACKED_HEAD_DIM,
    )
