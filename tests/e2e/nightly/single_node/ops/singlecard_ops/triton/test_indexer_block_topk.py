# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.indexer_block_topk import (
    INDEXER_MAX_TOP_K,
    indexer_block_softmax_topk,
    indexer_c4_head_reduce,
    indexer_c4_score_topk,
)
from vllm_ascend.ops.triton.indexer_mxfp4_qk import (
    create_e2m1_half_unit_lut,
    mxfp4_indexer_qk_matmul,
)
from vllm_ascend.ops.triton.mxfp4 import fp32_to_mxfp4
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


INDEXER_NUM_HEADS = 64
INDEXER_HEAD_DIM = 128


@pytest.mark.parametrize(
    "num_rows,num_blocks",
    [
        (1, 17),
        (2, 257),
        (2, 1025),
    ],
)
@torch.inference_mode()
def test_indexer_c4_head_reduce_matches_reference(num_rows, num_blocks):
    init_device_properties_triton()
    generator = torch.Generator().manual_seed(num_rows * 10000 + num_blocks)
    qk_scores = torch.randn(
        (num_rows, INDEXER_NUM_HEADS, num_blocks),
        generator=generator,
        dtype=torch.float32,
    )
    weights = torch.randn(
        (num_rows, INDEXER_NUM_HEADS),
        generator=generator,
        dtype=torch.float32,
    )
    expected = (
        torch.relu(qk_scores)
        * weights[:, :, None]
    ).sum(dim=1)

    actual = indexer_c4_head_reduce(
        qk_scores.to("npu"),
        weights.to("npu"),
    )

    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


@torch.inference_mode()
def test_indexer_c4_score_topk_matches_reference():
    init_device_properties_triton()
    num_rows = 2
    num_blocks = 2049
    top_k = 32
    qk_scores = torch.zeros(
        (num_rows, INDEXER_NUM_HEADS, num_blocks),
        dtype=torch.float32,
    )
    qk_scores[:, 0, :] = torch.arange(num_blocks, dtype=torch.float32)
    weights = torch.zeros((num_rows, INDEXER_NUM_HEADS), dtype=torch.float32)
    weights[:, 0] = 1.0
    expected_scores = qk_scores[:, 0, :]
    expected_values, expected_indices = torch.topk(
        expected_scores,
        top_k,
        dim=-1,
    )

    actual_values, actual_indices = indexer_c4_score_topk(
        qk_scores.to("npu"),
        weights.to("npu"),
        top_k,
    )

    torch.testing.assert_close(actual_values.cpu(), expected_values)
    torch.testing.assert_close(
        actual_indices.cpu(),
        expected_indices.to(torch.int32),
    )


@torch.inference_mode()
def test_mxfp4_qk_head_reduce_topk_pipeline():
    """Exercise the complete O2 -> head reduction -> O3 C4 data path."""
    init_device_properties_triton()
    num_query_tokens = 1
    num_key_tokens = 33
    top_k = 8
    generator = torch.Generator().manual_seed(20260825)
    query = torch.randn(
        (num_query_tokens, INDEXER_NUM_HEADS, INDEXER_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    )
    key = torch.randn(
        (num_key_tokens, 1, INDEXER_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    )
    # Keep the integration top-k ordering insensitive to different FP32
    # reduction trees. The separate head-reduction test covers all 64 heads.
    weights = torch.zeros(
        (num_query_tokens, INDEXER_NUM_HEADS),
        dtype=torch.float32,
    )
    weights[:, 0] = 1.0

    q_packed, q_scales = fp32_to_mxfp4(query.to("npu"))
    k_packed, k_scales = fp32_to_mxfp4(key.to("npu"))
    e2m1_lut = create_e2m1_half_unit_lut("npu")
    qk_scores = mxfp4_indexer_qk_matmul(
        q_packed,
        q_scales,
        k_packed,
        k_scales,
        e2m1_lut,
    )
    expected_scores = (
        torch.relu(qk_scores.cpu())
        * weights[:, :, None]
    ).sum(dim=1)
    expected_values, expected_indices = torch.topk(
        expected_scores,
        top_k,
        dim=-1,
    )

    actual_values, actual_indices = indexer_c4_score_topk(
        qk_scores,
        weights.to("npu"),
        top_k,
    )

    torch.testing.assert_close(
        actual_values.cpu(),
        expected_values,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        actual_indices.cpu(),
        expected_indices.to(torch.int32),
    )


@pytest.mark.parametrize(
    "shape,top_k",
    [
        ((1, 17), 1),
        ((2, 64), 8),
        ((2, 1025), 32),
        ((1, 16385), 64),
        ((1, 1025), INDEXER_MAX_TOP_K),
    ],
)
@torch.inference_mode()
def test_indexer_block_softmax_topk_matches_reference(shape, top_k):
    init_device_properties_triton()
    generator = torch.Generator().manual_seed(shape[-1] + top_k)
    logits = torch.randn(shape, generator=generator, dtype=torch.float32)
    # Add a strictly increasing perturbation to avoid unspecified tie order.
    logits += torch.arange(shape[-1], dtype=torch.float32) * 1e-7
    scores = torch.softmax(logits, dim=-1)
    expected_values, expected_indices = torch.topk(
        scores,
        top_k,
        dim=-1,
        largest=True,
        sorted=True,
    )

    actual_values, actual_indices = indexer_block_softmax_topk(
        scores.to("npu"),
        top_k,
    )

    torch.testing.assert_close(
        actual_values.cpu(),
        expected_values,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual_indices.cpu(),
        expected_indices.to(torch.int32),
        rtol=0.0,
        atol=0.0,
    )


@torch.inference_mode()
def test_indexer_block_softmax_topk_respects_valid_counts():
    init_device_properties_triton()
    scores = torch.tensor(
        [
            [0.05, 0.25, 0.10, 0.30, 0.20, 0.10],
            [0.40, 0.10, 0.30, 0.20, 0.00, 0.00],
            [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
        device="npu",
    )
    valid_counts = torch.tensor([6, 4, 1], dtype=torch.int32, device="npu")

    values, indices = indexer_block_softmax_topk(
        scores,
        top_k=4,
        valid_block_counts=valid_counts,
    )

    expected_values = torch.tensor(
        [
            [0.30, 0.25, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
            [1.00, -float("inf"), -float("inf"), -float("inf")],
        ],
        dtype=torch.float32,
    )
    expected_indices = torch.tensor(
        [
            [3, 1, 4, 2],
            [0, 2, 3, 1],
            [0, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(values.cpu(), expected_values)
    torch.testing.assert_close(indices.cpu(), expected_indices)


@torch.inference_mode()
def test_indexer_block_softmax_topk_pads_when_k_exceeds_row_length():
    init_device_properties_triton()
    scores = torch.tensor(
        [[0.25, 0.75]],
        dtype=torch.float32,
        device="npu",
    )
    values, indices = indexer_block_softmax_topk(scores, top_k=4)

    expected_values = torch.tensor(
        [[0.75, 0.25, -float("inf"), -float("inf")]],
        dtype=torch.float32,
    )
    expected_indices = torch.tensor(
        [[1, 0, -1, -1]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(values.cpu(), expected_values)
    torch.testing.assert_close(indices.cpu(), expected_indices)


@torch.inference_mode()
def test_indexer_block_softmax_topk_empty_rows_and_zero_k():
    empty_scores = torch.empty((0, 16), dtype=torch.float32, device="npu")
    values, indices = indexer_block_softmax_topk(empty_scores, 4)
    assert values.shape == (0, 4)
    assert indices.shape == (0, 4)

    scores = torch.empty((2, 16), dtype=torch.float32, device="npu")
    values, indices = indexer_block_softmax_topk(scores, 0)
    assert values.shape == (2, 0)
    assert indices.shape == (2, 0)


def test_indexer_block_softmax_topk_input_validation():
    scores = torch.zeros((2, 16), dtype=torch.float32)

    with pytest.raises(TypeError, match="torch.float32"):
        indexer_block_softmax_topk(scores.to(torch.bfloat16), 4)
    with pytest.raises(ValueError, match="at least one dimension"):
        indexer_block_softmax_topk(torch.tensor(1.0), 1)
    with pytest.raises(TypeError, match="top_k must be an int"):
        indexer_block_softmax_topk(scores, 4.0)
    with pytest.raises(ValueError, match="must not exceed"):
        indexer_block_softmax_topk(scores, INDEXER_MAX_TOP_K + 1)

    valid_counts = torch.full((2,), 16, dtype=torch.int64)
    with pytest.raises(TypeError, match="valid_block_counts"):
        indexer_block_softmax_topk(scores, 4, valid_counts)
    with pytest.raises(ValueError, match="must have shape"):
        indexer_block_softmax_topk(
            scores,
            4,
            torch.full((1,), 16, dtype=torch.int32),
        )


def test_indexer_c4_head_reduce_input_validation():
    qk_scores = torch.zeros((2, INDEXER_NUM_HEADS, 16), dtype=torch.float32)
    weights = torch.zeros((2, INDEXER_NUM_HEADS), dtype=torch.float32)

    with pytest.raises(TypeError, match="qk_scores"):
        indexer_c4_head_reduce(qk_scores.to(torch.bfloat16), weights)
    with pytest.raises(TypeError, match="weights"):
        indexer_c4_head_reduce(qk_scores, weights.to(torch.bfloat16))
    with pytest.raises(ValueError, match=r"shape\[-2\]"):
        indexer_c4_head_reduce(
            torch.zeros((2, INDEXER_NUM_HEADS - 1, 16)),
            weights[:, :-1],
        )
    with pytest.raises(ValueError, match="weights must have shape"):
        indexer_c4_head_reduce(qk_scores, weights[:1])
