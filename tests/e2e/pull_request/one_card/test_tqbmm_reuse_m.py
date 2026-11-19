# SPDX-License-Identifier: Apache-2.0
"""A3 paired-M B-reuse regression gates. Requires actual Triton/NPU execution."""

import pytest
import torch
import torch_npu  # noqa: F401

from tools.dsv4_a3_wo.benchmark_tqbmm import dynamic_qbmm_reference
from vllm_ascend.ops.triton.transpose_quant_batch_matmul import PreparedDynamicWoA, transpose_quant_batch_matmul
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or get_ascend_device_type() != AscendDeviceType.A3,
    reason="Experimental paired-M kernel requires A3",
)


@torch.inference_mode()
@pytest.mark.parametrize("shape", [(1, 1, 32, 16), (17, 2, 65, 33), (129, 3, 129, 65), (257, 1, 4096, 33)])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("stages", [1, 2])
def test_paired_m_exact_integer_golden_and_legacy(shape, transposed, stages):
    tokens, groups, inner, columns = shape
    torch.manual_seed(7)
    x = torch.randint(-128, 128, (tokens, groups, inner), dtype=torch.int8)
    w_physical = torch.randint(-128, 128, (groups, columns, inner), dtype=torch.int8)
    w = w_physical.transpose(1, 2)
    sx = torch.pow(2.0, -(torch.arange(tokens * groups).reshape(tokens, groups) % 4) - 4)
    sw = torch.pow(2.0, (torch.arange(groups * columns).reshape(groups, columns) % 5) - 4)
    integer = torch.bmm(x.double().transpose(0, 1), w.double()).transpose(0, 1)
    golden = (integer.float() * sw[None]) * sx[:, :, None]
    weight = w_physical.npu().transpose(1, 2)
    if not transposed:
        weight = weight.contiguous()
    tensors = x.npu(), weight, sx.npu(), sw.npu()
    options = {"output_dtype": torch.float32, "block_shape": (64, 32, 32), "num_stages": stages}
    legacy = transpose_quant_batch_matmul(*tensors, reuse_m=1, **options)
    paired = transpose_quant_batch_matmul(*tensors, reuse_m=2, **options)
    torch.testing.assert_close(paired.cpu(), golden, rtol=0, atol=0)
    torch.testing.assert_close(paired, legacy, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("stages", [1, 2])
def test_paired_m_fixed_k_pitch_and_nonzero_offsets(stages):
    x = torch.randint(-8, 9, (129, 2, 4096), dtype=torch.int8)
    physical = torch.randint(-8, 9, (2, 33, 4096), dtype=torch.int8)
    start, end = 32, 97
    weight = physical.npu().transpose(1, 2)[:, start:end]
    assert weight.stride() == (33 * 4096, 1, 4096)
    assert weight.storage_offset() == start
    golden = (
        torch.bmm(x[:, :, start:end].double().transpose(0, 1), physical[:, :, start:end].double().transpose(1, 2))
        .transpose(0, 1)
        .float()
    )
    result = transpose_quant_batch_matmul(
        x.npu()[:, :, start:end],
        weight,
        torch.ones(129, 2, device="npu"),
        torch.ones(2, 33, device="npu"),
        output_dtype=torch.float32,
        block_shape=(64, 32, 32),
        num_stages=stages,
        reuse_m=2,
    )
    torch.testing.assert_close(result.cpu(), golden, rtol=0, atol=0)


@torch.inference_mode()
def test_paired_prepared_dynamic_graph_with_changed_input():
    torch.manual_seed(11)
    activations = torch.randn(129, 2, 65, device="npu", dtype=torch.bfloat16)
    weight = torch.randint(-128, 128, (2, 33, 65), device="npu", dtype=torch.int8).transpose(1, 2)
    scale = torch.ones(2, 33, device="npu") * 0.03125
    plan = PreparedDynamicWoA(weight, scale, block_shape=(64, 32, 32), num_stages=2, reuse_m=2)
    first = plan(activations)
    torch.testing.assert_close(first, dynamic_qbmm_reference(activations, weight, scale), rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        captured = plan(activations)
    activations.mul_(0.5)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured, dynamic_qbmm_reference(activations, weight, scale), rtol=0, atol=0)
    second = plan(activations)
    assert first.data_ptr() != second.data_ptr()
