# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Block-level top-k selection for the C4 Lightning Indexer path."""

import math

import torch
from vllm.triton_utils import tl, triton


INDEXER_TOPK_CHUNK_SIZE = 1024
INDEXER_MAX_TOP_K = 512
INDEXER_NUM_HEADS = 64
INDEXER_HEAD_REDUCE_BLOCK_SIZE = 256


@triton.jit
def _indexer_head_reduce_kernel(
    qk_scores_ptr,
    weights_ptr,
    block_scores_ptr,
    num_blocks,
    NUM_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply ReLU and reduce the weighted scores of all indexer heads."""
    row_index = tl.program_id(axis=0)
    block_tile_index = tl.program_id(axis=1)

    head_offsets = tl.arange(0, NUM_HEADS)
    block_offsets = (
        block_tile_index * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )
    block_mask = block_offsets < num_blocks

    score_offsets = (
        row_index * NUM_HEADS * num_blocks
        + head_offsets[:, None] * num_blocks
        + block_offsets[None, :]
    )
    qk_scores = tl.load(
        qk_scores_ptr + score_offsets,
        mask=block_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(
        weights_ptr + row_index * NUM_HEADS + head_offsets,
    ).to(tl.float32)

    activated_scores = tl.maximum(qk_scores, 0.0)
    block_scores = tl.sum(
        activated_scores * weights[:, None],
        axis=0,
    )
    tl.store(
        block_scores_ptr + row_index * num_blocks + block_offsets,
        block_scores,
        mask=block_mask,
    )


@triton.jit
def _indexer_topk_chunk_kernel(
    input_values_ptr,
    input_indices_ptr,
    valid_counts_ptr,
    output_values_ptr,
    output_indices_ptr,
    num_input_values,
    top_k,
    output_row_stride,
    HAS_INPUT_INDICES: tl.constexpr,
    HAS_VALID_COUNTS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Reduce one input chunk to at most ``top_k`` candidates."""
    row_index = tl.program_id(axis=0)
    chunk_index = tl.program_id(axis=1)

    lane_offsets = tl.arange(0, CHUNK_SIZE)
    chunk_start = chunk_index * CHUNK_SIZE
    input_offsets = chunk_start + lane_offsets

    row_valid_count = num_input_values
    if HAS_VALID_COUNTS:
        row_valid_count = tl.load(valid_counts_ptr + row_index)
        row_valid_count = tl.maximum(0, tl.minimum(
            row_valid_count,
            num_input_values,
        ))

    input_mask = (
        (input_offsets < num_input_values)
        & (input_offsets < row_valid_count)
    )
    values = tl.load(
        input_values_ptr + row_index * num_input_values + input_offsets,
        mask=input_mask,
        other=-float("inf"),
    ).to(tl.float32)

    if HAS_INPUT_INDICES:
        indices = tl.load(
            input_indices_ptr + row_index * num_input_values + input_offsets,
            mask=input_mask,
            other=-1,
        ).to(tl.int32)
    else:
        indices = input_offsets.to(tl.int32)

    local_valid_count = tl.maximum(
        0,
        tl.minimum(row_valid_count - chunk_start, CHUNK_SIZE),
    )
    local_output_count = tl.minimum(local_valid_count, top_k)
    output_base = (
        row_index * output_row_stride
        + chunk_index * top_k
    )

    # ``top_k`` is a runtime scalar, so tl.range lowers this to a device loop
    # instead of unrolling 512 copies into the compiled kernel.
    for rank in tl.range(0, top_k):
        selected_lane = tl.argmax(values, axis=0)
        selected_value = tl.max(values, axis=0)
        selected_index = tl.sum(
            tl.where(lane_offsets == selected_lane, indices, 0),
            axis=0,
        )

        rank_is_valid = rank < local_output_count
        tl.store(
            output_values_ptr + output_base + rank,
            tl.where(rank_is_valid, selected_value, -float("inf")),
        )
        tl.store(
            output_indices_ptr + output_base + rank,
            tl.where(rank_is_valid, selected_index, -1),
        )

        values = tl.where(
            lane_offsets == selected_lane,
            -float("inf"),
            values,
        )


def _allocate_topk_outputs(
    scores: torch.Tensor,
    output_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty(
        output_shape,
        dtype=torch.float32,
        device=scores.device,
    )
    indices = torch.empty(
        output_shape,
        dtype=torch.int32,
        device=scores.device,
    )
    return values, indices


def indexer_c4_head_reduce(
    qk_scores: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Reduce C4 QK scores from 64 heads to one score per block.

    This implements the production Lightning Indexer score formula before
    top-k selection::

        block_score = sum_h(weights[h] * relu(qk_score[h]))

    Args:
        qk_scores: FP32 tensor with shape ``[..., 64, num_blocks]``, normally
            produced by :func:`mxfp4_indexer_qk_matmul`.
        weights: FP32 tensor with shape ``[..., 64]``. The caller is
            responsible for applying the model's indexer scale beforehand.

    Returns:
        FP32 tensor with shape ``[..., num_blocks]``.
    """
    if qk_scores.dtype != torch.float32:
        raise TypeError(
            "qk_scores must have dtype torch.float32, "
            f"but got {qk_scores.dtype}"
        )
    if weights.dtype != torch.float32:
        raise TypeError(
            "weights must have dtype torch.float32, "
            f"but got {weights.dtype}"
        )
    if qk_scores.ndim < 2:
        raise ValueError("qk_scores must have at least two dimensions")
    if qk_scores.shape[-2] != INDEXER_NUM_HEADS:
        raise ValueError(
            f"qk_scores.shape[-2] must be {INDEXER_NUM_HEADS}, "
            f"but got {qk_scores.shape[-2]}"
        )

    expected_weights_shape = tuple(qk_scores.shape[:-1])
    if tuple(weights.shape) != expected_weights_shape:
        raise ValueError(
            f"weights must have shape {expected_weights_shape}, "
            f"but got {tuple(weights.shape)}"
        )
    if weights.device != qk_scores.device:
        raise ValueError("qk_scores and weights must be on the same device")

    leading_shape = tuple(qk_scores.shape[:-2])
    num_rows = math.prod(leading_shape) if leading_shape else 1
    num_blocks = qk_scores.shape[-1]
    output = torch.empty(
        (*leading_shape, num_blocks),
        dtype=torch.float32,
        device=qk_scores.device,
    )
    if num_rows == 0 or num_blocks == 0:
        return output

    flat_qk_scores = qk_scores.contiguous().view(
        num_rows,
        INDEXER_NUM_HEADS,
        num_blocks,
    )
    flat_weights = weights.contiguous().view(
        num_rows,
        INDEXER_NUM_HEADS,
    )
    grid = (
        num_rows,
        triton.cdiv(num_blocks, INDEXER_HEAD_REDUCE_BLOCK_SIZE),
    )
    _indexer_head_reduce_kernel[grid](
        flat_qk_scores,
        flat_weights,
        output,
        num_blocks,
        NUM_HEADS=INDEXER_NUM_HEADS,
        BLOCK_N=INDEXER_HEAD_REDUCE_BLOCK_SIZE,
    )
    return output


def indexer_block_softmax_topk(
    block_softmax_scores: torch.Tensor,
    top_k: int,
    valid_block_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k blocks from C4 Lightning Indexer softmax scores.

    The input is assumed to contain block-level softmax scores. Softmax is
    monotonic, so the same kernel can also be used for logits when only the
    selected indices are needed.

    Long rows are processed hierarchically. Each Triton program reduces one
    1024-element chunk to ``top_k`` candidates, and subsequent passes merge
    those candidates until a single sorted top-k list remains.

    Args:
        block_softmax_scores: Contiguous or strided FP32 tensor with shape
            ``[..., num_blocks]``.
        top_k: Number of blocks to return. The C4 DSV4 path supports at most
            512, matching ``index_topk``.
        valid_block_counts: Optional INT32 tensor with shape
            ``block_softmax_scores.shape[:-1]``. Values are clamped to
            ``[0, num_blocks]`` on device. Rows shorter than ``top_k`` are
            padded with value ``-inf`` and index ``-1``.

    Returns:
        ``(topk_values, topk_indices)`` with shape ``[..., top_k]`` and
        dtypes FP32 and INT32 respectively. Values are sorted descending.
    """
    if block_softmax_scores.dtype != torch.float32:
        raise TypeError(
            "block_softmax_scores must have dtype torch.float32, "
            f"but got {block_softmax_scores.dtype}"
        )
    if block_softmax_scores.ndim == 0:
        raise ValueError("block_softmax_scores must have at least one dimension")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an int, but got {type(top_k).__name__}")

    num_blocks = block_softmax_scores.shape[-1]
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, but got {top_k}")
    if top_k > INDEXER_MAX_TOP_K:
        raise ValueError(
            f"top_k must not exceed {INDEXER_MAX_TOP_K}, but got {top_k}"
        )

    leading_shape = tuple(block_softmax_scores.shape[:-1])
    num_rows = math.prod(leading_shape) if leading_shape else 1
    output_shape = (*leading_shape, top_k)
    topk_values, topk_indices = _allocate_topk_outputs(
        block_softmax_scores,
        output_shape,
    )
    if num_rows == 0 or top_k == 0:
        return topk_values, topk_indices
    if num_blocks == 0:
        topk_values.fill_(-float("inf"))
        topk_indices.fill_(-1)
        return topk_values, topk_indices

    flat_scores = block_softmax_scores.contiguous().view(num_rows, num_blocks)
    flat_valid_counts = flat_scores
    has_valid_counts = valid_block_counts is not None
    if valid_block_counts is not None:
        if valid_block_counts.dtype != torch.int32:
            raise TypeError(
                "valid_block_counts must have dtype torch.int32, "
                f"but got {valid_block_counts.dtype}"
            )
        if tuple(valid_block_counts.shape) != leading_shape:
            raise ValueError(
                f"valid_block_counts must have shape {leading_shape}, "
                f"but got {tuple(valid_block_counts.shape)}"
            )
        if valid_block_counts.device != block_softmax_scores.device:
            raise ValueError(
                "valid_block_counts and block_softmax_scores must be on "
                "the same device"
            )
        flat_valid_counts = valid_block_counts.contiguous().view(-1)

    current_values = flat_scores
    current_indices = flat_scores
    current_length = num_blocks
    has_input_indices = False
    use_valid_counts = has_valid_counts

    while current_length > INDEXER_TOPK_CHUNK_SIZE:
        num_chunks = triton.cdiv(
            current_length,
            INDEXER_TOPK_CHUNK_SIZE,
        )
        output_length = num_chunks * top_k
        next_values = torch.empty(
            (num_rows, output_length),
            dtype=torch.float32,
            device=block_softmax_scores.device,
        )
        next_indices = torch.empty(
            (num_rows, output_length),
            dtype=torch.int32,
            device=block_softmax_scores.device,
        )
        _indexer_topk_chunk_kernel[(num_rows, num_chunks)](
            current_values,
            current_indices,
            flat_valid_counts,
            next_values,
            next_indices,
            current_length,
            top_k,
            output_length,
            HAS_INPUT_INDICES=has_input_indices,
            HAS_VALID_COUNTS=use_valid_counts,
            CHUNK_SIZE=INDEXER_TOPK_CHUNK_SIZE,
        )
        current_values = next_values
        current_indices = next_indices
        current_length = output_length
        has_input_indices = True
        # Intermediate rows contain fixed-width candidate groups. Invalid
        # slots are represented by (-inf, -1), so no separate count is needed.
        use_valid_counts = False

    flat_topk_values = topk_values.view(num_rows, top_k)
    flat_topk_indices = topk_indices.view(num_rows, top_k)
    _indexer_topk_chunk_kernel[(num_rows, 1)](
        current_values,
        current_indices,
        flat_valid_counts,
        flat_topk_values,
        flat_topk_indices,
        current_length,
        top_k,
        top_k,
        HAS_INPUT_INDICES=has_input_indices,
        HAS_VALID_COUNTS=use_valid_counts,
        CHUNK_SIZE=INDEXER_TOPK_CHUNK_SIZE,
    )
    return topk_values, topk_indices


def indexer_c4_score_topk(
    qk_scores: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    valid_block_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce C4 per-head scores and select the top-k blocks.

    DSA currently consumes only the selected block indices. A softmax over the
    final scalar block scores is therefore intentionally omitted because it is
    monotonic and cannot change those indices.
    """
    block_scores = indexer_c4_head_reduce(qk_scores, weights)
    return indexer_block_softmax_topk(
        block_scores,
        top_k,
        valid_block_counts,
    )
