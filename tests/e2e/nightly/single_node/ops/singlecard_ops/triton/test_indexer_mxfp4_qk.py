# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.indexer_mxfp4_cache import (
    scatter_mxfp4_indexer_cache,
)
from vllm_ascend.ops.triton.indexer_block_topk import indexer_c4_score_topk
from vllm_ascend.ops.triton.indexer_mxfp4_qk import (
    create_e2m1_half_unit_lut,
    mxfp4_indexer_paged_qk_matmul,
    mxfp4_indexer_qk_matmul as _mxfp4_indexer_qk_matmul,
)
from vllm_ascend.ops.triton.mxfp4 import (
    _decode_e2m1_half_units,
    fp32_to_mxfp4,
    mxfp4_to_fp32,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

INDEXER_NUM_HEADS = 64
INDEXER_HEAD_DIM = 128
INDEXER_PACKED_HEAD_DIM = INDEXER_HEAD_DIM // 2
INDEXER_NUM_SCALE_GROUPS = 4
MXFP4_GROUP_SIZE = 32
E8M0_EXPONENT_BIAS = 127
E2M1_NUM_CODES = 16
DIRECT_DOT_BLOCK_M = 32
DIRECT_DOT_BLOCK_N = 64
DIRECT_DOT_BLOCK_K = 32
DIRECT_DOT_PACKED_K = DIRECT_DOT_BLOCK_K // 2


def _run_mxfp4_indexer_qk_matmul(
    q_packed: torch.Tensor,
    q_scales: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
) -> torch.Tensor:
    # Production integration should register this 16-byte constant once. Tests
    # create it per call so every case remains independent.
    e2m1_lut = create_e2m1_half_unit_lut(q_packed.device)
    return _mxfp4_indexer_qk_matmul(
        q_packed,
        q_scales,
        k_packed,
        k_scales,
        e2m1_lut,
    )


@triton.jit
def _decode_e2m1_half_units_vector_kernel(
    packed_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Decode one copy of all E2M1 codes without passing through Cube."""
    offsets = tl.arange(0, BLOCK_SIZE)[None, :]
    packed = tl.load(packed_ptr + offsets // 2)
    codes = tl.where(
        (offsets & 1) == 0,
        packed & 0xF,
        (packed >> 4) & 0xF,
    )
    decoded = _decode_e2m1_half_units(codes, 1, BLOCK_SIZE)
    tl.store(output_ptr + offsets, decoded)


@torch.inference_mode()
def test_decode_e2m1_half_units_vector_path():
    """Separate E2M1 decoding correctness from the Vector-to-Cube handoff."""
    init_device_properties_triton()
    packed = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        dtype=torch.uint8,
        device="npu",
    )
    actual = torch.empty(E2M1_NUM_CODES, dtype=torch.int8, device="npu")

    _decode_e2m1_half_units_vector_kernel[(1,)](
        packed.view(torch.int8),
        actual,
        BLOCK_SIZE=E2M1_NUM_CODES,
    )

    expected = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        dtype=torch.int8,
    )
    actual_cpu = actual.cpu()
    assert torch.equal(actual_cpu, expected), (
        f"decoded half units: {actual_cpu.tolist()}, "
        f"expected: {expected.tolist()}"
    )


@triton.jit
def _direct_int8_dot_kernel(
    query_ptr,
    key_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Run INT8 Cube with operands loaded directly from device memory."""
    row_offsets = tl.arange(0, BLOCK_M)
    column_offsets = tl.arange(0, BLOCK_N)
    reduction_offsets = tl.arange(0, BLOCK_K)

    query = tl.load(
        query_ptr
        + row_offsets[:, None] * BLOCK_K
        + reduction_offsets[None, :]
    )
    key = tl.load(
        key_ptr
        + column_offsets[:, None] * BLOCK_K
        + reduction_offsets[None, :]
    )
    result = tl.dot(query, tl.trans(key))
    output_offsets = (
        row_offsets[:, None] * BLOCK_N + column_offsets[None, :]
    )
    tl.store(output_ptr + output_offsets, result)


@torch.inference_mode()
def test_direct_int8_dot_from_device_memory():
    """Check INT8 tl.dot independently from packed decoding and VSel."""
    init_device_properties_triton()
    half_units = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        dtype=torch.int8,
    ).repeat(2)
    query = half_units[None, :].expand(
        DIRECT_DOT_BLOCK_M,
        DIRECT_DOT_BLOCK_K,
    ).contiguous().to("npu")
    key = half_units[None, :].expand(
        DIRECT_DOT_BLOCK_N,
        DIRECT_DOT_BLOCK_K,
    ).contiguous().to("npu")
    actual = torch.empty(
        (DIRECT_DOT_BLOCK_M, DIRECT_DOT_BLOCK_N),
        dtype=torch.int32,
        device="npu",
    )

    _direct_int8_dot_kernel[(1,)](
        query,
        key,
        actual,
        BLOCK_M=DIRECT_DOT_BLOCK_M,
        BLOCK_N=DIRECT_DOT_BLOCK_N,
        BLOCK_K=DIRECT_DOT_BLOCK_K,
    )

    actual_cpu = actual.cpu()
    expected_value = 1096
    assert torch.all(actual_cpu == expected_value), (
        f"direct INT8 dot unique values: {torch.unique(actual_cpu).tolist()}, "
        f"expected only: {expected_value}"
    )


@triton.jit
def _lookup_decode_int8_dot_kernel(
    query_packed_ptr,
    key_packed_ptr,
    e2m1_lut_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PACKED_K: tl.constexpr,
):
    """Decode packed nibbles through a device LUT before INT8 Cube."""
    row_offsets = tl.arange(0, BLOCK_M)
    column_offsets = tl.arange(0, BLOCK_N)
    reduction_offsets = tl.arange(0, BLOCK_K)

    query_packed = tl.load(
        query_packed_ptr
        + row_offsets[:, None] * PACKED_K
        + reduction_offsets[None, :] // 2
    )
    query_codes = tl.where(
        (reduction_offsets[None, :] & 1) == 0,
        query_packed & 0xF,
        (query_packed >> 4) & 0xF,
    )
    query = tl.load(e2m1_lut_ptr + query_codes)

    key_packed = tl.load(
        key_packed_ptr
        + column_offsets[:, None] * PACKED_K
        + reduction_offsets[None, :] // 2
    )
    key_codes = tl.where(
        (reduction_offsets[None, :] & 1) == 0,
        key_packed & 0xF,
        (key_packed >> 4) & 0xF,
    )
    key = tl.load(e2m1_lut_ptr + key_codes)

    result = tl.dot(query, tl.trans(key))
    output_offsets = (
        row_offsets[:, None] * BLOCK_N + column_offsets[None, :]
    )
    tl.store(output_ptr + output_offsets, result)


@torch.inference_mode()
def test_lookup_decode_result_can_feed_int8_dot():
    """Check that a GM lookup load is a legal Vector-to-Cube handoff."""
    init_device_properties_triton()
    packed_codes = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        dtype=torch.uint8,
    ).repeat(2)
    query_packed = packed_codes[None, :].expand(
        DIRECT_DOT_BLOCK_M,
        DIRECT_DOT_PACKED_K,
    ).contiguous().to("npu")
    key_packed = packed_codes[None, :].expand(
        DIRECT_DOT_BLOCK_N,
        DIRECT_DOT_PACKED_K,
    ).contiguous().to("npu")
    e2m1_lut = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        dtype=torch.int8,
        device="npu",
    )
    actual = torch.empty(
        (DIRECT_DOT_BLOCK_M, DIRECT_DOT_BLOCK_N),
        dtype=torch.int32,
        device="npu",
    )

    _lookup_decode_int8_dot_kernel[(1,)](
        query_packed.view(torch.int8),
        key_packed.view(torch.int8),
        e2m1_lut,
        actual,
        BLOCK_M=DIRECT_DOT_BLOCK_M,
        BLOCK_N=DIRECT_DOT_BLOCK_N,
        BLOCK_K=DIRECT_DOT_BLOCK_K,
        PACKED_K=DIRECT_DOT_PACKED_K,
    )

    actual_cpu = actual.cpu()
    expected_value = 1096
    assert torch.all(actual_cpu == expected_value), (
        f"lookup INT8 dot unique values: {torch.unique(actual_cpu).tolist()}, "
        f"expected only: {expected_value}"
    )


def _make_indexer_inputs(
    num_query_tokens: int,
    num_key_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(
        num_query_tokens * 1000 + num_key_tokens
    )
    query_groups = torch.empty(
        (
            num_query_tokens,
            INDEXER_NUM_HEADS,
            INDEXER_NUM_SCALE_GROUPS,
            MXFP4_GROUP_SIZE,
        ),
        dtype=torch.float32,
    ).uniform_(-3.0, 3.0, generator=generator)
    key_groups = torch.empty(
        (
            num_key_tokens,
            1,
            INDEXER_NUM_SCALE_GROUPS,
            MXFP4_GROUP_SIZE,
        ),
        dtype=torch.float32,
    ).uniform_(-3.0, 3.0, generator=generator)
    query_groups[..., 0] = 6.0
    key_groups[..., 0] = 6.0
    query_group_scales = torch.tensor([0.25, 0.5, 1.0, 2.0])
    key_group_scales = torch.tensor([2.0, 1.0, 0.5, 0.25])
    query_groups *= query_group_scales[None, None, :, None]
    key_groups *= key_group_scales[None, None, :, None]
    query = query_groups.reshape(
        num_query_tokens,
        INDEXER_NUM_HEADS,
        INDEXER_HEAD_DIM,
    )
    key = key_groups.reshape(num_key_tokens, 1, INDEXER_HEAD_DIM)
    return query, key


def _qk_reference_from_quantized_inputs(
    q_packed: torch.Tensor,
    q_scales: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
) -> torch.Tensor:
    # O2 multiplies the values represented by MXFP4, not the original FP32
    # inputs. E2M1 values are represented exactly as signed half units in the
    # Cube inputs, so this FP32 groupwise reference has identical values.
    query = mxfp4_to_fp32(q_packed, q_scales).cpu().reshape(
        q_packed.shape[0],
        INDEXER_NUM_HEADS,
        INDEXER_NUM_SCALE_GROUPS,
        MXFP4_GROUP_SIZE,
    )
    key = mxfp4_to_fp32(k_packed, k_scales).cpu()[:, 0, :].reshape(
        k_packed.shape[0],
        INDEXER_NUM_SCALE_GROUPS,
        MXFP4_GROUP_SIZE,
    )
    group_scores = torch.einsum("thgd,sgd->thsg", query, key)
    return group_scores.sum(dim=-1)


@pytest.mark.parametrize(
    "num_query_tokens,num_key_tokens,keep_key_head_dim",
    [
        (1, 17, False),
        (2, 64, True),
        (2, 65, False),
        (2, 129, False),
        (1, 257, True),
    ],
)
@torch.inference_mode()
def test_mxfp4_indexer_qk_matches_reference(
    num_query_tokens,
    num_key_tokens,
    keep_key_head_dim,
):
    init_device_properties_triton()
    query_cpu, key_cpu = _make_indexer_inputs(
        num_query_tokens,
        num_key_tokens,
    )
    q_packed, q_scales = fp32_to_mxfp4(query_cpu.to("npu"))
    k_packed, k_scales = fp32_to_mxfp4(key_cpu.to("npu"))

    expected = _qk_reference_from_quantized_inputs(
        q_packed,
        q_scales,
        k_packed,
        k_scales,
    )
    if not keep_key_head_dim:
        k_packed = k_packed[:, 0, :]
        k_scales = k_scales[:, 0, :]

    actual = _run_mxfp4_indexer_qk_matmul(
        q_packed,
        q_scales,
        k_packed,
        k_scales,
    )

    assert actual.shape == (
        num_query_tokens,
        INDEXER_NUM_HEADS,
        num_key_tokens,
    )
    assert actual.dtype == torch.float32
    assert actual.device == q_packed.device
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=1e-3,
        atol=1e-3,
    )


@torch.inference_mode()
def test_mxfp4_indexer_paged_cache_matches_dense_causal_reference():
    """Cover MXFP4 scatter, paged reads, batching, and causal counts."""
    init_device_properties_triton()
    cache_block_size = 4
    num_pages = 4
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([24, 16], dtype=torch.int32)
    expected_valid_counts = torch.tensor([5, 6, 4], dtype=torch.int32)
    block_table = torch.tensor(
        [[2, 0], [3, 1]],
        dtype=torch.int32,
    )

    generator = torch.Generator().manual_seed(20260826)
    query = torch.randn(
        (3, INDEXER_NUM_HEADS, INDEXER_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    )
    request_keys = [
        torch.randn(
            (6, 1, INDEXER_HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
        ),
        torch.randn(
            (4, 1, INDEXER_HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
        ),
    ]
    q_packed, q_scales = fp32_to_mxfp4(query.to("npu"))
    packed_requests = []
    scale_requests = []
    for keys in request_keys:
        packed, scales = fp32_to_mxfp4(keys.to("npu"))
        packed_requests.append(packed)
        scale_requests.append(scales)

    # Match model_runner_v1._adjust_kv_layout: every physical page stores
    # packed keys first and its E8M0 scales second. The two tensors are
    # strided views into the same allocation, so neither cache is contiguous
    # across page boundaries.
    bytes_per_token = (
        INDEXER_PACKED_HEAD_DIM + INDEXER_NUM_SCALE_GROUPS
    )
    bytes_per_page = cache_block_size * bytes_per_token
    raw_cache = torch.zeros(
        num_pages * bytes_per_page,
        dtype=torch.uint8,
        device="npu",
    )
    key_cache = torch.as_strided(
        raw_cache,
        size=(
            num_pages,
            cache_block_size,
            1,
            INDEXER_PACKED_HEAD_DIM,
        ),
        stride=(
            bytes_per_page,
            INDEXER_PACKED_HEAD_DIM,
            INDEXER_PACKED_HEAD_DIM,
            1,
        ),
    )
    scale_cache = torch.as_strided(
        raw_cache,
        size=(
            num_pages,
            cache_block_size,
            1,
            INDEXER_NUM_SCALE_GROUPS,
        ),
        stride=(
            bytes_per_page,
            INDEXER_NUM_SCALE_GROUPS,
            INDEXER_NUM_SCALE_GROUPS,
            1,
        ),
        storage_offset=cache_block_size * INDEXER_PACKED_HEAD_DIM,
    )
    slot_mapping = torch.tensor(
        [8, 9, 10, 11, 0, 1, 12, 13, 14, 15],
        dtype=torch.int32,
        device="npu",
    )
    scatter_mxfp4_indexer_cache(
        torch.cat(packed_requests, dim=0),
        torch.cat(scale_requests, dim=0),
        key_cache,
        scale_cache,
        slot_mapping,
    )

    e2m1_lut = create_e2m1_half_unit_lut("npu")
    actual, actual_valid_counts = mxfp4_indexer_paged_qk_matmul(
        q_packed,
        q_scales,
        key_cache,
        scale_cache,
        block_table.to("npu"),
        query_start_loc.to("npu"),
        seq_lens.to("npu"),
        max_seq_len=24,
        e2m1_lut=e2m1_lut,
    )

    expected = torch.zeros(
        (3, INDEXER_NUM_HEADS, 6),
        dtype=torch.float32,
    )
    query_requests = (0, 0, 1)
    for query_index, request_index in enumerate(query_requests):
        valid_count = int(expected_valid_counts[query_index])
        dense_scores = _run_mxfp4_indexer_qk_matmul(
            q_packed[query_index : query_index + 1],
            q_scales[query_index : query_index + 1],
            packed_requests[request_index][:valid_count],
            scale_requests[request_index][:valid_count],
        )
        expected[query_index, :, :valid_count] = dense_scores.cpu()[0]

    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        actual_valid_counts.cpu(),
        expected_valid_counts,
    )


@torch.inference_mode()
def test_a5_device_operator_quantizes_and_scatters_mxfp4():
    """Exercise the framework-facing query quantization and cache write API."""
    # Keep this framework-only dependency local so paged-kernel tests can run
    # in minimal Triton development images.
    from vllm_ascend.device.device_op import DeviceOperator

    init_device_properties_triton()
    generator = torch.Generator().manual_seed(20260827)
    query = torch.randn(
        (2, INDEXER_NUM_HEADS, INDEXER_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
        device="npu",
    )
    key = torch.randn(
        (2, 1, INDEXER_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
        device="npu",
    )
    key_cache = torch.zeros(
        (2, 2, 1, INDEXER_PACKED_HEAD_DIM),
        dtype=torch.uint8,
        device="npu",
    )
    scale_cache = torch.zeros(
        (2, 2, 1, INDEXER_NUM_SCALE_GROUPS),
        dtype=torch.uint8,
        device="npu",
    )
    slot_mapping = torch.tensor(
        [2, 1],
        dtype=torch.int32,
        device="npu",
    )

    q_packed, q_scales, k_packed, k_scales = (
        DeviceOperator.indexer_quant_scatter(
            query,
            key,
            key_cache,
            scale_cache,
            None,
            slot_mapping,
            use_mxfp4=True,
        )
    )
    expected_q_packed, expected_q_scales = fp32_to_mxfp4(query)
    expected_k_packed, expected_k_scales = fp32_to_mxfp4(key)

    torch.testing.assert_close(q_packed.cpu(), expected_q_packed.cpu())
    torch.testing.assert_close(q_scales.cpu(), expected_q_scales.cpu())
    torch.testing.assert_close(k_packed.cpu(), expected_k_packed.cpu())
    torch.testing.assert_close(k_scales.cpu(), expected_k_scales.cpu())
    # slot 2 is page 1 / offset 0; slot 1 is page 0 / offset 1.
    torch.testing.assert_close(
        key_cache[1, 0].cpu(),
        expected_k_packed[0].cpu(),
    )
    torch.testing.assert_close(
        key_cache[0, 1].cpu(),
        expected_k_packed[1].cpu(),
    )
    torch.testing.assert_close(
        scale_cache[1, 0].cpu(),
        expected_k_scales[0].cpu(),
    )
    torch.testing.assert_close(
        scale_cache[0, 1].cpu(),
        expected_k_scales[1].cpu(),
    )


@torch.inference_mode()
def test_a5_indexer_ops_runs_mxfp4_cache_to_topk_path():
    """Exercise the O4-O6 framework path with real Triton kernels."""
    from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerOps

    init_device_properties_triton()
    generator = torch.Generator().manual_seed(20260828)
    query = torch.randn(
        (2, INDEXER_NUM_HEADS, INDEXER_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
        device="npu",
    )
    key = torch.randn(
        (4, 1, INDEXER_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
        device="npu",
    )
    weights = torch.rand(
        (2, INDEXER_NUM_HEADS),
        generator=generator,
        dtype=torch.float32,
        device="npu",
    )
    key_cache = torch.zeros(
        (1, 4, 1, INDEXER_PACKED_HEAD_DIM),
        dtype=torch.uint8,
        device="npu",
    )
    scale_cache = torch.zeros(
        (1, 4, 1, INDEXER_NUM_SCALE_GROUPS),
        dtype=torch.uint8,
        device="npu",
    )
    slot_mapping = torch.arange(4, dtype=torch.int32, device="npu")
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([16], dtype=torch.int32, device="npu")
    block_table = torch.tensor([[0]], dtype=torch.int32, device="npu")

    indexer_ops = AscendIndexerOps(index_topk=3, use_mxfp4=True)
    q_packed, q_scales = indexer_ops.quantize_query(query)
    k_packed, key_scale = indexer_ops.quantize_key_and_update_cache(
        key,
        key_cache,
        scale_cache,
        None,
        slot_mapping,
    )
    assert key_scale is None
    actual = indexer_ops.select_topk(
        q_packed,
        weights,
        q_scales,
        key_cache,
        scale_cache,
        SimpleNamespace(
            block_table=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            max_seq_len=16,
        ),
    )

    dense_scores = _run_mxfp4_indexer_qk_matmul(
        q_packed,
        q_scales,
        k_packed,
        scale_cache.view(4, 1, INDEXER_NUM_SCALE_GROUPS),
    )
    valid_key_counts = torch.tensor([3, 4], dtype=torch.int32, device="npu")
    _, expected_indices = indexer_c4_score_topk(
        dense_scores,
        weights,
        3,
        valid_key_counts,
    )
    torch.testing.assert_close(
        actual.cpu(),
        expected_indices.unsqueeze(1).cpu(),
    )


@torch.inference_mode()
def test_mxfp4_indexer_qk_opposing_scale_exponents():
    """Large opposing E8M0 exponents must cancel without overflow."""
    init_device_properties_triton()
    num_key_tokens = 3

    # E2M1 code 0b0010 represents +1.0, so 0x22 packs two +1.0s.
    q_packed = torch.full(
        (1, INDEXER_NUM_HEADS, INDEXER_PACKED_HEAD_DIM),
        0x22,
        dtype=torch.uint8,
        device="npu",
    )
    k_packed = torch.full(
        (num_key_tokens, 1, INDEXER_PACKED_HEAD_DIM),
        0x22,
        dtype=torch.uint8,
        device="npu",
    )

    q_exponents = torch.tensor([126, -126, 100, -100], dtype=torch.int32)
    k_base_exponents = -q_exponents
    key_exponent_offsets = torch.tensor([-1, 0, 1], dtype=torch.int32)
    k_exponents = (
        k_base_exponents[None, :]
        + key_exponent_offsets[:, None]
    )
    q_scales = (
        q_exponents + E8M0_EXPONENT_BIAS
    ).to(torch.uint8)[None, None, :].expand(
        1,
        INDEXER_NUM_HEADS,
        INDEXER_NUM_SCALE_GROUPS,
    ).contiguous().to("npu")
    k_scales = (
        k_exponents + E8M0_EXPONENT_BIAS
    ).to(torch.uint8)[:, None, :].contiguous().to("npu")

    actual = _run_mxfp4_indexer_qk_matmul(
        q_packed,
        q_scales,
        k_packed,
        k_scales,
    )

    # Each group contributes 32 * 2**offset and there are four groups.
    expected_per_key = 128.0 * torch.exp2(
        key_exponent_offsets.to(torch.float32)
    )
    expected = expected_per_key[None, None, :].expand(
        1,
        INDEXER_NUM_HEADS,
        num_key_tokens,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_mxfp4_indexer_qk_avoids_intermediate_subnormal():
    """Opposing endpoint scales must not flush a finite result to zero."""
    init_device_properties_triton()

    # Each 32-element group contains one 0.5 and 31 zeros, giving a raw
    # query-key group dot of 0.25.
    packed_head = torch.zeros(INDEXER_PACKED_HEAD_DIM, dtype=torch.uint8)
    packed_head[:: MXFP4_GROUP_SIZE // 2] = 0x01
    q_packed = packed_head[None, None, :].expand(
        1,
        INDEXER_NUM_HEADS,
        INDEXER_PACKED_HEAD_DIM,
    ).contiguous().to("npu")
    k_packed = packed_head[None, None, :].to("npu")

    q_exponents = torch.tensor([127, -127, 127, -127])
    k_exponents = -q_exponents
    q_scales = (
        q_exponents + E8M0_EXPONENT_BIAS
    ).to(torch.uint8)[None, None, :].expand(
        1,
        INDEXER_NUM_HEADS,
        INDEXER_NUM_SCALE_GROUPS,
    ).contiguous().to("npu")
    k_scales = (
        k_exponents + E8M0_EXPONENT_BIAS
    ).to(torch.uint8)[None, None, :].to("npu")

    actual = _run_mxfp4_indexer_qk_matmul(
        q_packed,
        q_scales,
        k_packed,
        k_scales,
    )

    # Four groups each contribute 0.25 after the opposing scales cancel.
    expected = torch.ones((1, INDEXER_NUM_HEADS, 1))
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_mxfp4_indexer_qk_all_e2m1_codes():
    """The exact half-unit decode must preserve all 16 E2M1 codes."""
    init_device_properties_triton()

    # Low nibbles are even codes and high nibbles are the following odd code.
    all_codes_once = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        dtype=torch.uint8,
    )
    packed_head = all_codes_once.repeat(
        INDEXER_PACKED_HEAD_DIM // all_codes_once.numel()
    )
    q_packed = packed_head[None, None, :].expand(
        1,
        INDEXER_NUM_HEADS,
        INDEXER_PACKED_HEAD_DIM,
    ).contiguous().to("npu")
    k_packed = packed_head[None, None, :].to("npu")
    q_scales = torch.full(
        (1, INDEXER_NUM_HEADS, INDEXER_NUM_SCALE_GROUPS),
        E8M0_EXPONENT_BIAS,
        dtype=torch.uint8,
        device="npu",
    )
    k_scales = torch.full(
        (1, 1, INDEXER_NUM_SCALE_GROUPS),
        E8M0_EXPONENT_BIAS,
        dtype=torch.uint8,
        device="npu",
    )

    actual = _run_mxfp4_indexer_qk_matmul(
        q_packed,
        q_scales,
        k_packed,
        k_scales,
    )

    # Sum of squared E2M1 values 0..15 is 137; the pattern repeats 8 times.
    expected = torch.full((1, INDEXER_NUM_HEADS, 1), 1096.0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_mxfp4_indexer_qk_zero_inputs():
    init_device_properties_triton()
    query = torch.zeros(
        (1, INDEXER_NUM_HEADS, INDEXER_HEAD_DIM),
        dtype=torch.float32,
        device="npu",
    )
    key = torch.zeros(
        (129, 1, INDEXER_HEAD_DIM),
        dtype=torch.float32,
        device="npu",
    )
    q_packed, q_scales = fp32_to_mxfp4(query)
    k_packed, k_scales = fp32_to_mxfp4(key)

    output = _run_mxfp4_indexer_qk_matmul(
        q_packed,
        q_scales,
        k_packed,
        k_scales,
    )

    assert torch.count_nonzero(output).cpu().item() == 0


@pytest.mark.parametrize(
    "num_query_tokens,num_key_tokens",
    [(0, 5), (2, 0)],
)
def test_mxfp4_indexer_qk_empty_inputs(
    num_query_tokens,
    num_key_tokens,
):
    q_packed = torch.empty(
        (num_query_tokens, INDEXER_NUM_HEADS, INDEXER_PACKED_HEAD_DIM),
        dtype=torch.uint8,
        device="npu",
    )
    q_scales = torch.empty(
        (num_query_tokens, INDEXER_NUM_HEADS, INDEXER_NUM_SCALE_GROUPS),
        dtype=torch.uint8,
        device="npu",
    )
    k_packed = torch.empty(
        (num_key_tokens, 1, INDEXER_PACKED_HEAD_DIM),
        dtype=torch.uint8,
        device="npu",
    )
    k_scales = torch.empty(
        (num_key_tokens, 1, INDEXER_NUM_SCALE_GROUPS),
        dtype=torch.uint8,
        device="npu",
    )

    output = _run_mxfp4_indexer_qk_matmul(
        q_packed,
        q_scales,
        k_packed,
        k_scales,
    )

    assert output.shape == (
        num_query_tokens,
        INDEXER_NUM_HEADS,
        num_key_tokens,
    )
    assert output.dtype == torch.float32


def test_mxfp4_indexer_qk_input_validation():
    q_packed = torch.empty(
        (1, INDEXER_NUM_HEADS, INDEXER_PACKED_HEAD_DIM),
        dtype=torch.uint8,
    )
    q_scales = torch.empty(
        (1, INDEXER_NUM_HEADS, INDEXER_NUM_SCALE_GROUPS),
        dtype=torch.uint8,
    )
    k_packed = torch.empty(
        (5, 1, INDEXER_PACKED_HEAD_DIM),
        dtype=torch.uint8,
    )
    k_scales = torch.empty(
        (5, 1, INDEXER_NUM_SCALE_GROUPS),
        dtype=torch.uint8,
    )
    e2m1_lut = create_e2m1_half_unit_lut(q_packed.device)

    with pytest.raises(TypeError, match="e2m1_lut must have dtype torch.int8"):
        _mxfp4_indexer_qk_matmul(
            q_packed,
            q_scales,
            k_packed,
            k_scales,
            e2m1_lut.to(torch.int16),
        )

    with pytest.raises(ValueError, match="e2m1_lut must have shape"):
        _mxfp4_indexer_qk_matmul(
            q_packed,
            q_scales,
            k_packed,
            k_scales,
            e2m1_lut[:-1],
        )

    with pytest.raises(TypeError, match="q_packed must have dtype torch.uint8"):
        _run_mxfp4_indexer_qk_matmul(
            q_packed.to(torch.int8),
            q_scales,
            k_packed,
            k_scales,
        )

    with pytest.raises(ValueError, match="q_packed must have shape"):
        _run_mxfp4_indexer_qk_matmul(
            q_packed[:, :63, :],
            q_scales,
            k_packed,
            k_scales,
        )

    with pytest.raises(ValueError, match="q_scales must have shape"):
        _run_mxfp4_indexer_qk_matmul(
            q_packed,
            q_scales[:, :, :3],
            k_packed,
            k_scales,
        )

    with pytest.raises(ValueError, match="exactly one shared key head"):
        _run_mxfp4_indexer_qk_matmul(
            q_packed,
            q_scales,
            k_packed.expand(-1, 2, -1),
            k_scales,
        )

    with pytest.raises(ValueError, match=r"k_packed.shape\[-1\] must be 64"):
        _run_mxfp4_indexer_qk_matmul(
            q_packed,
            q_scales,
            k_packed[:, :, :63],
            k_scales,
        )

    with pytest.raises(ValueError, match="k_scales must have shape"):
        _run_mxfp4_indexer_qk_matmul(
            q_packed,
            q_scales,
            k_packed,
            k_scales[:, :, :3],
        )
