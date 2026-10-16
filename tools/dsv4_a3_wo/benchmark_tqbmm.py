# SPDX-License-Identifier: Apache-2.0
"""A3 TQBMM correctness and timing probe; no model service is required.

Run from the repository root with ``python -m tools.dsv4_a3_wo.benchmark_tqbmm``.
"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu

from tools.dsv4_a3_wo.qbmm_probe import check_offset, error_metrics, load_group_count, load_scale, load_tensor
from vllm_ascend.ops.triton.transpose_quant_batch_matmul import (
    dynamic_quant_grouped_wo_a,
    transpose_quant_batch_matmul,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def qbmm_reference(x, weight, sx, sw, output_dtype):
    """Same INT8 data/scales as Triton; include slice copies and output stack."""
    return torch.stack(
        [
            torch_npu.npu_quant_matmul(
                x[:, group].contiguous(),
                weight[group].contiguous(),
                sw[group].contiguous(),
                pertoken_scale=sx[:, group].contiguous(),
                output_dtype=output_dtype,
            )
            for group in range(x.shape[1])
        ],
        dim=1,
    )


def dynamic_qbmm_reference(x, weight, sw):
    outputs = []
    for group in range(x.shape[1]):
        quantized, scale = torch_npu.npu_dynamic_quant(x[:, group].contiguous(), dst_type=torch.int8)
        outputs.append(
            torch_npu.npu_quant_matmul(
                quantized,
                weight[group].contiguous(),
                sw[group].contiguous(),
                pertoken_scale=scale.reshape(-1).float(),
                output_dtype=x.dtype,
            )
        )
    return torch.stack(outputs, dim=1)


def elapsed_ms(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def load_wo_a(args):
    model = args.quant_model.resolve()
    groups = load_group_count(model)
    roots = (
        f"model.layers.{args.layer}.self_attn",
        f"model.layers.{args.layer}.attn",
        f"layers.{args.layer}.self_attn",
        f"layers.{args.layer}.attn",
    )
    for root in roots:
        try:
            weight = load_tensor(model, f"{root}.wo_a.weight")
        except KeyError:
            continue
        scale = load_scale(model, f"{root}.wo_a.weight")
        check_offset(model, f"{root}.wo_a.weight")
        break
    else:
        raise KeyError(f"No wo_a.weight for layer {args.layer}")
    if weight.dtype != torch.int8 or weight.ndim != 2 or weight.shape[0] % groups:
        raise ValueError("Expected raw ModelSlim INT8 wo_a [G*N,K]")
    if scale.numel() != weight.shape[0]:
        raise ValueError("Expected one weight scale per wo_a output channel")
    if not torch.isfinite(scale.float()).all():
        raise ValueError("Non-finite wo_a scale in checkpoint")
    if groups % args.tp_size or not 0 <= args.tp_rank < args.tp_size:
        raise ValueError("TP must divide o_groups and tp_rank must be in [0,tp_size)")
    columns = weight.shape[0] // groups
    local_groups = groups // args.tp_size
    first = args.tp_rank * local_groups
    weight = weight.reshape(groups, columns, -1)[first : first + local_groups].transpose(1, 2)
    scale = scale.reshape(groups, columns)[first : first + local_groups].float()
    return weight.to(args.device), scale.to(args.device)


@torch.inference_mode()
def run(args):
    if not torch.npu.is_available() or get_ascend_device_type() != AscendDeviceType.A3:
        raise RuntimeError("This probe requires an Ascend A3 NPU and Triton-Ascend")
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    if args.quant_model:
        weight, sw = load_wo_a(args)
    else:
        weight = torch.randint(-16, 17, (args.groups, args.inner, args.columns), dtype=torch.int8, device=args.device)
        sw = torch.rand((args.groups, args.columns), device=args.device, dtype=torch.float32) * 0.05
    groups, inner, columns = weight.shape
    activations = torch.randn((args.tokens, groups, inner), dtype=torch.bfloat16, device=args.device)
    x, sx = torch_npu.npu_dynamic_quant(activations.view(-1, inner), dst_type=torch.int8)
    x = x.view_as(activations)
    sx = sx.reshape(args.tokens, groups).float()
    block_shape = tuple(args.block_shape) if args.block_shape else None

    def fused():
        return transpose_quant_batch_matmul(x, weight, sx, sw, block_shape=block_shape)

    def reference():
        return qbmm_reference(x, weight, sx, sw, torch.bfloat16)

    actual, expected = fused(), reference()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
    dynamic_actual = dynamic_quant_grouped_wo_a(activations, weight, sw, block_shape=block_shape)
    dynamic_expected = dynamic_qbmm_reference(activations, weight, sw)
    torch.testing.assert_close(dynamic_actual, dynamic_expected, rtol=0.01, atol=0.02)
    report = {
        "shape_T_G_K_N": [args.tokens, groups, inner, columns],
        "weight_stride": list(weight.stride()),
        "quant_model": str(args.quant_model) if args.quant_model else None,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "same_int8_inputs": error_metrics(actual, expected),
        "including_dynamic_quant": error_metrics(dynamic_actual, dynamic_expected),
        "note": "Compares with the existing INT8 composition, not original BF16/FP8 model accuracy.",
    }
    if args.graph:
        # Warmup/compilation happened above, outside graph capture.
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = dynamic_quant_grouped_wo_a(activations, weight, sw, block_shape=block_shape)
        # Replay with changed inputs, not just a repeated captured answer.
        activations.mul_(0.5)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, dynamic_qbmm_reference(activations, weight, sw), rtol=0.01, atol=0.02)
        report["graph_replay_with_changed_input"] = "passed"
    if not args.check_only:
        fused_ms = elapsed_ms(fused, args.warmup, args.iterations)
        reference_ms = elapsed_ms(reference, args.warmup, args.iterations)
        dynamic_ms = elapsed_ms(
            lambda: dynamic_quant_grouped_wo_a(activations, weight, sw, block_shape=block_shape),
            args.warmup,
            args.iterations,
        )
        baseline_ms = elapsed_ms(lambda: dynamic_qbmm_reference(activations, weight, sw), args.warmup, args.iterations)
        report["timing_ms"] = {
            "triton_prequantized": fused_ms,
            "qbmm_composition_prequantized": reference_ms,
            "prequantized_speedup": reference_ms / fused_ms,
            "triton_including_quant": dynamic_ms,
            "qbmm_composition_including_quant": baseline_ms,
            "including_quant_speedup": baseline_ms / dynamic_ms,
        }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--inner", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--quant-model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--block-shape", type=int, nargs=3, metavar=("M", "N", "K"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if min(args.tokens, args.groups, args.inner, args.columns, args.iterations, args.tp_size) <= 0 or args.warmup < 0:
        parser.error("dimensions, iterations and tp-size must be positive; warmup must be >= 0")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
