# SPDX-License-Identifier: Apache-2.0
"""A3 two-axis reuse gates. Actual compilation and NPU execution are required."""

import pytest
import torch
import torch_npu  # noqa: F401

from tools.dsv4_a3_wo.benchmark_tqbmm import dynamic_qbmm_reference
from vllm_ascend.ops.triton.transpose_quant_batch_matmul import PreparedDynamicWoA, transpose_quant_batch_matmul
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or get_ascend_device_type() != AscendDeviceType.A3,
    reason="Two-axis reuse requires A3",
)


@torch.inference_mode()
@pytest.mark.parametrize("shape", [(1, 1, 32, 1), (65, 2, 129, 131), (129, 1, 4096, 65)])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("reuse_m", [1, 2])
@pytest.mark.parametrize("stages", [1, 2])
def test_two_axis_exact_cpu_golden(shape, transposed, reuse_m, stages):
    tokens, groups, inner, columns = shape
    torch.manual_seed(17)
    x_cpu = torch.randint(-128, 128, (tokens, groups, inner), dtype=torch.int8)
    backing = torch.randint(-128, 128, (groups, columns, inner), dtype=torch.int8)
    sx = torch.pow(2.0, -(torch.arange(tokens * groups).reshape(tokens, groups) % 4) - 4)
    sw = torch.pow(2.0, (torch.arange(groups * columns).reshape(groups, columns) % 5) - 4)
    integer = torch.bmm(x_cpu.double().transpose(0, 1), backing.double().transpose(1, 2)).transpose(0, 1)
    expected = (integer.float() * sw[None]) * sx[:, :, None]
    weight = backing.npu().transpose(1, 2)
    if not transposed:
        weight = weight.contiguous()
    result = transpose_quant_batch_matmul(
        x_cpu.npu(),
        weight,
        sx.npu(),
        sw.npu(),
        output_dtype=torch.float32,
        block_shape=(32, 32, 32),
        num_stages=stages,
        reuse_m=reuse_m,
        reuse_n=2,
    )
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("reuse_m", [1, 2])
def test_two_axis_fixed_physical_k_pitch(reuse_m):
    torch.manual_seed(19)
    x_cpu = torch.randint(-8, 9, (65, 2, 4096), dtype=torch.int8)
    backing = torch.randint(-8, 9, (2, 67, 4096), dtype=torch.int8)
    x = x_cpu.npu()[:, :, 32:97]
    weight = backing.npu().transpose(1, 2)[:, 32:97]
    assert weight.stride() == (67 * 4096, 1, 4096)
    integer = torch.bmm(
        x_cpu[:, :, 32:97].double().transpose(0, 1), backing[:, :, 32:97].double().transpose(1, 2)
    ).transpose(0, 1)
    result = transpose_quant_batch_matmul(
        x,
        weight,
        torch.ones(65, 2, device="npu"),
        torch.ones(2, 67, device="npu"),
        output_dtype=torch.float32,
        block_shape=(32, 32, 32),
        reuse_m=reuse_m,
        reuse_n=2,
    )
    torch.testing.assert_close(result.cpu(), integer.float(), rtol=0, atol=0)


@torch.inference_mode()
def test_two_axis_prepared_graph_changed_input_and_fresh_output():
    torch.manual_seed(23)
    activations = torch.randn(65, 2, 65, dtype=torch.bfloat16, device="npu")
    weight = torch.randint(-128, 128, (2, 67, 65), dtype=torch.int8, device="npu").transpose(1, 2)
    scale = torch.ones(2, 67, device="npu") * 0.03125
    plan = PreparedDynamicWoA(weight, scale, block_shape=(32, 32, 32), num_stages=2, reuse_m=2, reuse_n=2)
    first = plan(activations)
    torch.testing.assert_close(first, dynamic_qbmm_reference(activations, weight, scale), rtol=0, atol=0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = plan(activations)
    activations.mul_(0.5)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured, dynamic_qbmm_reference(activations, weight, scale), rtol=0, atol=0)
    second = plan(activations)
    assert second.data_ptr() != first.data_ptr()
