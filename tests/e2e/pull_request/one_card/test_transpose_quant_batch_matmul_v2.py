# SPDX-License-Identifier: Apache-2.0
"""A3 V2 compiler/device gates; no full-model service required."""

import pytest
import torch

from tools.dsv4_a3_wo.benchmark_tqbmm import dynamic_qbmm_reference
from vllm_ascend.ops.triton.transpose_quant_batch_matmul_v2 import (
    PreparedDynamicWoAV2,
    PreparedTQBMMV2,
    V2Config,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or get_ascend_device_type() != AscendDeviceType.A3,
    reason="Experimental V2 requires A3 and Triton-Ascend",
)


def _run_plan_to_cpu_with_stages(plan):
    """Diagnostic barriers only; never use this helper for performance timing.

    A faulthandler stack at `plan().cpu()` cannot distinguish dispatch from
    device completion or D2H copy. Flush markers and isolate each boundary.
    Pre-launch synchronization also checks pending uploads independently.
    """
    print("[tqbmm-v2] input-sync.begin", flush=True)
    torch.npu.synchronize()
    print("[tqbmm-v2] input-sync.done", flush=True)
    print("[tqbmm-v2] launch.begin", flush=True)
    output = plan()
    print("[tqbmm-v2] launch.returned", flush=True)
    print("[tqbmm-v2] device-sync.begin", flush=True)
    torch.npu.synchronize()
    print("[tqbmm-v2] device-sync.done", flush=True)
    print("[tqbmm-v2] cpu-copy.begin", flush=True)
    actual = output.cpu()
    print("[tqbmm-v2] cpu-copy.done", flush=True)
    return actual


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("panel_k", [64, 128, 256])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("span", [(0, 32), (32, 97), (0, 4096), (0, 129), (32, 289)])
def test_v2_integer_golden_and_persistent_tails(prefetch, panel_k, transposed, span):
    t, g, physical_k, n = 67, 2, 4096, 65
    torch.manual_seed(0)
    x_cpu = torch.randint(-8, 9, (t, g, physical_k), dtype=torch.int8, device="cpu")
    w_cpu = torch.randint(-8, 9, (g, n, physical_k), dtype=torch.int8, device="cpu")
    start, end = span
    expected = (
        torch.bmm(x_cpu[:, :, start:end].double().transpose(0, 1), w_cpu[:, :, start:end].double().transpose(1, 2))
        .transpose(0, 1)
        .float()
    )
    assert expected.device.type == "cpu"
    print(
        f"[tqbmm-v2] cpu-golden.done span={span} transposed={transposed} panel_k={panel_k} prefetch={prefetch}",
        flush=True,
    )
    x = x_cpu.npu()[:, :, start:end]
    weight = w_cpu.npu().transpose(1, 2)[:, start:end, :]
    if not transposed:
        weight = weight.contiguous()
    plan = PreparedTQBMMV2(
        x,
        weight,
        torch.ones(t, g, dtype=torch.float32, device="npu"),
        torch.ones(g, n, dtype=torch.float32, device="npu"),
        output_dtype=torch.float32,
        config=V2Config((32, 32, 64), panel_k, prefetch, programs=3, num_stages=2),
    )
    actual = _run_plan_to_cpu_with_stages(plan)
    print("[tqbmm-v2] compare.begin", flush=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("[tqbmm-v2] compare.done", flush=True)


@pytest.mark.parametrize("prefetch", [False, True])
def test_v2_dynamic_graph_changed_input(prefetch):
    torch.manual_seed(2)
    x = torch.randn(129, 2, 4096, dtype=torch.bfloat16, device="npu")
    w = torch.randint(-8, 9, (2, 65, 4096), dtype=torch.int8, device="npu").transpose(1, 2)
    sw = torch.rand(2, 65, dtype=torch.float32, device="npu")
    plan = PreparedDynamicWoAV2(w, sw, config=V2Config(prefetch=prefetch))
    first, second = plan(x), plan(x)
    assert first.data_ptr() != second.data_ptr()
    expected = dynamic_qbmm_reference(x, w, sw)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = plan(x)
    x.mul_(0.5)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured, dynamic_qbmm_reference(x, w, sw), rtol=0, atol=0)
