# SPDX-License-Identifier: Apache-2.0
"""Run on A3: integer arithmetic, layouts/tails, scale groups and ACL graph."""

import pytest
import torch
import torch_npu

from tools.dsv4_a3_wo.benchmark_tqbmm import dynamic_qbmm_reference, qbmm_reference
from vllm_ascend.ops.triton.transpose_quant_batch_matmul import (
    dynamic_quant_grouped_wo_a,
    transpose_quant_batch_matmul,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or get_ascend_device_type() != AscendDeviceType.A3,
    reason="Experimental TQBMM requires A3",
)


@pytest.mark.parametrize("inner", [32, 4096])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("transpose_weight_tile", [False, True])
def test_transposed_weight_multi_k_regression(inner, signed, transposed, transpose_weight_tile):
    """Regression: only strided, multi-K inputs failed on the reported A3."""
    tokens, groups, columns = 16, 1, 32
    torch.manual_seed(0)
    low = -8 if signed else 0
    x_cpu = torch.randint(low, 9, (tokens, groups, inner), dtype=torch.int8)
    w_cpu = torch.randint(low, 9, (groups, columns, inner), dtype=torch.int8)
    expected = torch.bmm(x_cpu.double().transpose(0, 1), w_cpu.double().transpose(1, 2))
    expected = expected.transpose(0, 1).float()
    weight = w_cpu.npu().transpose(1, 2)
    if not transposed:
        weight = weight.contiguous()
    actual = transpose_quant_batch_matmul(
        x_cpu.npu(),
        weight,
        torch.ones(tokens, groups, device="npu", dtype=torch.float32),
        torch.ones(groups, columns, device="npu", dtype=torch.float32),
        output_dtype=torch.float32,
        block_shape=(16, 32, 32),
        num_stages=1,
        transpose_weight_tile=transpose_weight_tile,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("start,end", [(0, 32), (32, 64), (2048, 2080), (4064, 4096), (0, 64), (0, 65), (0, 4096)])
@pytest.mark.parametrize("transpose_weight_tile", [False, True])
def test_transposed_weight_fixed_pitch_k_slices(start, end, transpose_weight_tile):
    tokens, groups, physical_k, columns = 16, 1, 4096, 32
    torch.manual_seed(0)
    x_cpu = torch.randint(-8, 9, (tokens, groups, physical_k), dtype=torch.int8)
    w_cpu = torch.randint(-8, 9, (groups, columns, physical_k), dtype=torch.int8)
    x = x_cpu.npu()[:, :, start:end]
    weight = w_cpu.npu().transpose(1, 2)[:, start:end, :]
    assert weight.stride() == (columns * physical_k, 1, physical_k)
    assert weight.storage_offset() == start
    expected = (
        torch.bmm(
            x_cpu[:, :, start:end].double().transpose(0, 1),
            w_cpu[:, :, start:end].double().transpose(1, 2),
        )
        .transpose(0, 1)
        .float()
    )
    actual = transpose_quant_batch_matmul(
        x,
        weight,
        torch.ones(tokens, groups, device="npu", dtype=torch.float32),
        torch.ones(groups, columns, device="npu", dtype=torch.float32),
        output_dtype=torch.float32,
        block_shape=(16, 32, 32),
        num_stages=1,
        transpose_weight_tile=transpose_weight_tile,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("transpose_weight_tile", [False, True])
def test_transposed_multi_k_tail_and_distinct_group_scales(transpose_weight_tile):
    tokens, groups, inner, columns = 17, 2, 4099, 33
    torch.manual_seed(0)
    x_cpu = torch.randint(-8, 9, (tokens, groups, inner), dtype=torch.int8)
    w_cpu = torch.randint(-8, 9, (groups, columns, inner), dtype=torch.int8)
    # Exact powers of two, different for tokens, groups and output channels.
    sx = torch.pow(2.0, -(torch.arange(tokens * groups).reshape(tokens, groups) % 4) - 4)
    sw = torch.pow(2.0, torch.arange(groups * columns).reshape(groups, columns) % 5 - 4)
    golden = torch.bmm(x_cpu.double().transpose(0, 1), w_cpu.double().transpose(1, 2)).transpose(0, 1)
    expected = ((golden * sw.double()[None]) * sx.double()[:, :, None]).float()
    actual = transpose_quant_batch_matmul(
        x_cpu.npu(),
        w_cpu.npu().transpose(1, 2),
        sx.npu(),
        sw.npu(),
        output_dtype=torch.float32,
        block_shape=(16, 32, 32),
        num_stages=1,
        transpose_weight_tile=transpose_weight_tile,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 1, 64, 32), (17, 2, 65, 33), (65, 3, 129, 131), (513, 2, 64, 129)])
@pytest.mark.parametrize("transposed_weight", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_integer_reference_and_group_scales(shape, transposed_weight, output_dtype):
    tokens, groups, inner, columns = shape
    torch.manual_seed(0)
    x = torch.randint(-8, 9, (tokens, groups, inner), dtype=torch.int8)
    w = torch.randint(-8, 9, (groups, columns, inner), dtype=torch.int8).transpose(1, 2)
    if not transposed_weight:
        w = w.contiguous()
    # Different powers of two for every group expose scale broadcast mistakes.
    sx = torch.tensor([2.0 ** (-group - 3) for group in range(groups)]).expand(tokens, groups).contiguous()
    sw = torch.tensor([2.0 ** (group - 4) for group in range(groups)])[:, None].expand(groups, columns).contiguous()
    # FP64 CPU golden: independent of CANN and FP32/HF32 matmul modes.
    expected = torch.bmm(x.double().transpose(0, 1), w.double()).transpose(0, 1)
    expected = ((expected * sw.double()[None]) * sx.double()[:, :, None]).to(output_dtype)
    # Copy first, then transpose, to guarantee a strided NPU weight view.
    w_npu = w.transpose(1, 2).contiguous().npu().transpose(1, 2) if transposed_weight else w.npu()
    actual = transpose_quant_batch_matmul(x.npu(), w_npu, sx.npu(), sw.npu(), output_dtype=output_dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_noncontiguous_inputs_and_scales():
    x = torch.randint(-8, 9, (3, 19, 96), dtype=torch.int8, device="npu").transpose(0, 1)
    w = torch.randint(-8, 9, (3, 96, 64), dtype=torch.int8, device="npu")
    sx = torch.rand(3, 19, device="npu").transpose(0, 1)
    sw = torch.rand(64, 3, device="npu").transpose(0, 1)
    expected = qbmm_reference(x, w, sx, sw, torch.bfloat16)
    torch.testing.assert_close(transpose_quant_batch_matmul(x, w, sx, sw), expected, rtol=0.01, atol=0.02)


@pytest.mark.parametrize("value", [-128, -127, 0, 127])
@pytest.mark.parametrize("transpose_weight_tile", [False, True])
def test_extreme_int8_values_and_zero_rows(value, transpose_weight_tile):
    x = torch.full((1, 2, 257), value, dtype=torch.int8, device="npu")
    w = torch.full((2, 257, 33), value, dtype=torch.int8, device="npu")
    sx = torch.full((1, 2), 0.125, device="npu")
    sw = torch.full((2, 33), 0.25, device="npu")
    actual = transpose_quant_batch_matmul(
        x, w, sx, sw, output_dtype=torch.float32, transpose_weight_tile=transpose_weight_tile
    )
    expected = torch.full((1, 2, 33), value * value * 257 * 0.125 * 0.25)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens,groups", [(1, 1), (16, 1), (65, 2), (512, 1)])
def test_real_wo_dimensions_and_dynamic_quant(tokens, groups):
    x = torch.randn(tokens, groups, 4096, device="npu", dtype=torch.bfloat16)
    w = torch.randint(-127, 128, (groups, 1024, 4096), device="npu", dtype=torch.int8).transpose(1, 2)
    sw = torch.rand(groups, 1024, device="npu") * 0.01
    expected = dynamic_qbmm_reference(x, w, sw)
    torch.testing.assert_close(dynamic_quant_grouped_wo_a(x, w, sw), expected, rtol=0.01, atol=0.02)


def test_zero_tokens():
    actual = transpose_quant_batch_matmul(
        torch.empty(0, 2, 64, dtype=torch.int8, device="npu"),
        torch.empty(2, 64, 32, dtype=torch.int8, device="npu"),
        torch.empty(0, 2, device="npu"),
        torch.ones(2, 32, device="npu"),
    )
    assert actual.shape == (0, 2, 32)


def test_nz_storage_is_rejected():
    x = torch.ones(1, 1, 64, dtype=torch.int8, device="npu")
    w = torch_npu.npu_format_cast(torch.ones(1, 64, 32, dtype=torch.int8, device="npu"), 29)
    with pytest.raises(ValueError, match="ND storage"):
        transpose_quant_batch_matmul(x, w, torch.ones(1, 1, device="npu"), torch.ones(1, 32, device="npu"))


def test_graph_replay_reads_updated_inputs():
    x = torch.randn(16, 1, 4096, device="npu", dtype=torch.bfloat16)
    w = torch.randint(-8, 9, (1, 4096, 1024), device="npu", dtype=torch.int8)
    sw = torch.rand(1, 1024, device="npu") * 0.01
    dynamic_quant_grouped_wo_a(x, w, sw)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = dynamic_quant_grouped_wo_a(x, w, sw)
    x.mul_(0.5)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(actual, dynamic_qbmm_reference(x, w, sw), rtol=0.01, atol=0.02)
