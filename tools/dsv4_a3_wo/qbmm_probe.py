"""A3 DeepSeek-V4-Flash wo_a/wo_b feasibility probe (not a vLLM integration).

Read one msModelSlim W8A8 layer from safetensors, then compute
transpose + npu_quant_matmul on NPU. It uses 2-D QBMM per wo_a group because
A3's per-token and per-channel scale vectors are specified as 1-D. This is a
correctness probe, not a throughput proxy for a fused batched kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def find_tensor_file(model_dir: Path, tensor_name: str) -> Path:
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(index_files) > 1:
        raise ValueError(f"Multiple safetensors indexes in {model_dir}")
    if index_files:
        weight_map = json.loads(index_files[0].read_text(encoding="utf-8"))["weight_map"]
        if tensor_name not in weight_map:
            raise KeyError(f"{tensor_name} missing from {index_files[0]}")
        path = model_dir / weight_map[tensor_name]
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as archive:
            if tensor_name in archive.keys():
                return path
    raise KeyError(f"{tensor_name} missing from {model_dir}/*.safetensors")


def load_tensor(model_dir: Path, tensor_name: str) -> torch.Tensor:
    path = find_tensor_file(model_dir, tensor_name)
    with safe_open(path, framework="pt", device="cpu") as archive:
        return archive.get_tensor(tensor_name)


def load_scale(model_dir: Path, weight_name: str) -> torch.Tensor:
    root = weight_name.removesuffix(".weight")
    errors = []
    for name in (f"{root}.weight_scale", f"{root}.scale"):
        try:
            return load_tensor(model_dir, name)
        except KeyError as exc:
            errors.append(str(exc))
    raise KeyError("No per-channel scale for " + root + "; " + "; ".join(errors))


def check_offset(model_dir: Path, weight_name: str) -> None:
    root = weight_name.removesuffix(".weight")
    for name in (f"{root}.weight_offset", f"{root}.offset"):
        try:
            offset = load_tensor(model_dir, name)
        except KeyError:
            continue
        if torch.count_nonzero(offset).item():
            raise ValueError(f"{name} has nonzero offsets; this symmetric W8A8 probe does not support them")


def load_group_count(model_dir: Path) -> int:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Pass --groups or provide {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for candidate in (config.get("text_config", {}), config):
        if "o_groups" in candidate:
            return int(candidate["o_groups"])
    raise KeyError(f"o_groups not found in {config_path}; pass --groups")


def validate_weights(
    a_weight: torch.Tensor,
    a_scale: torch.Tensor,
    b_weight: torch.Tensor,
    b_scale: torch.Tensor,
    groups: int,
) -> tuple[int, int, int]:
    if a_weight.dtype != torch.int8 or b_weight.dtype != torch.int8:
        raise TypeError(f"Expected INT8 wo_a and wo_b; got {a_weight.dtype}, {b_weight.dtype}")
    if a_weight.ndim != 2 or b_weight.ndim != 2:
        raise ValueError(f"Expected raw 2-D checkpoint weights; got {a_weight.shape}, {b_weight.shape}")
    a_out, a_in = a_weight.shape
    b_out, b_in = b_weight.shape
    if groups <= 0 or a_out % groups or b_in != a_out:
        raise ValueError(f"Incompatible wo_a/wo_b/groups: {a_weight.shape}, {b_weight.shape}, {groups}")
    if a_scale.numel() != a_out or b_scale.numel() != b_out:
        raise ValueError(
            f"Per-channel scales must have {a_out} and {b_out} elements; "
            f"got {a_scale.shape} and {b_scale.shape}"
        )
    if not torch.isfinite(a_scale.float()).all() or not torch.isfinite(b_scale.float()).all():
        raise ValueError("Non-finite wo_a/wo_b scale")
    return a_in, a_out // groups, b_out


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.float().flatten()
    reference = reference.float().flatten()
    delta = actual - reference
    return {
        "max_abs": delta.abs().max().item(),
        "nrmse": (delta.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12)).item(),
        "cosine": torch.nn.functional.cosine_similarity(actual[None, :], reference[None, :]).item(),
    }


def reference_matmul(
    quantized_x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return (
        (quantized_x.float() @ weight.transpose(-1, -2).float())
        * x_scale.float().reshape(-1, 1)
        * weight_scale.float().reshape(1, -1)
    ).to(torch.bfloat16)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    import torch_npu

    model_dir = args.quant_model.resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    groups = args.groups if args.groups is not None else load_group_count(model_dir)
    roots = [
        f"model.layers.{args.layer}.self_attn",
        f"model.layers.{args.layer}.attn",
        f"layers.{args.layer}.self_attn",
        f"layers.{args.layer}.attn",
    ]
    root = next(
        (root for root in roots if (model_dir / "model.safetensors.index.json").exists() and
         _indexed_weight_exists(model_dir, f"{root}.wo_a.weight")),
        None,
    )
    if root is None:
        root = next((root for root in roots if _weight_exists(model_dir, f"{root}.wo_a.weight")), None)
    if root is None:
        raise KeyError(f"No wo_a.weight found for layer {args.layer}; tried {roots}")

    a_name = f"{root}.wo_a.weight"
    b_name = f"{root}.wo_b.weight"
    a_weight = load_tensor(model_dir, a_name)
    b_weight = load_tensor(model_dir, b_name)
    a_scale = load_scale(model_dir, a_name)
    b_scale = load_scale(model_dir, b_name)
    check_offset(model_dir, a_name)
    check_offset(model_dir, b_name)
    group_hidden, rank, output_hidden = validate_weights(a_weight, a_scale, b_weight, b_scale, groups)
    print(
        f"checkpoint={model_dir} layer={args.layer} groups={groups} "
        f"wo_a={tuple(a_weight.shape)} wo_b={tuple(b_weight.shape)}",
        flush=True,
    )

    device = torch.device(args.device)
    if device.type != "npu" or not torch.npu.is_available():
        raise RuntimeError("Run on an Ascend NPU; this probe does not have a CPU QBMM fallback")
    torch.manual_seed(args.seed)
    x = torch.randn(args.tokens, groups, group_hidden, dtype=torch.bfloat16, device=device)
    a_weight = a_weight.reshape(groups, rank, group_hidden).to(device)
    a_scale = a_scale.reshape(groups, rank).float().to(device)
    b_weight = b_weight.to(device)
    b_scale = b_scale.reshape(output_hidden).float().to(device)

    a_outputs = []
    a_references = []
    for group in range(groups):
        x_quant, x_scale = torch_npu.npu_dynamic_quant(x[:, group, :].contiguous(), dst_type=torch.int8)
        group_weight = a_weight[group]
        group_scale = a_scale[group]
        # torch native transpose view + 2-D QBMM: [T,K] @ [K,R] -> [T,R].
        a_outputs.append(
            torch_npu.npu_quant_matmul(
                x_quant,
                group_weight.transpose(-1, -2),
                group_scale,
                pertoken_scale=x_scale.reshape(-1).float(),
                output_dtype=torch.bfloat16,
            )
        )
        a_references.append(reference_matmul(x_quant, group_weight, x_scale, group_scale))
    a_output = torch.stack(a_outputs, dim=1).reshape(args.tokens, groups * rank)
    a_reference = torch.stack(a_references, dim=1).reshape(args.tokens, groups * rank)

    b_quant, b_x_scale = torch_npu.npu_dynamic_quant(a_output.contiguous(), dst_type=torch.int8)
    b_output = torch_npu.npu_quant_matmul(
        b_quant,
        b_weight.transpose(-1, -2),
        b_scale,
        pertoken_scale=b_x_scale.reshape(-1).float(),
        output_dtype=torch.bfloat16,
    )
    b_reference = reference_matmul(b_quant, b_weight, b_x_scale, b_scale)
    torch.npu.synchronize()
    a_metrics = error_metrics(a_output, a_reference)
    b_metrics = error_metrics(b_output, b_reference)
    report = {
        "layer": args.layer,
        "groups": groups,
        "tokens": args.tokens,
        "wo_a_weight": list(a_weight.shape),
        "wo_b_weight": list(b_weight.shape),
        "wo_a_qbmm_vs_dequantized_int8": a_metrics,
        "wo_b_qbmm_vs_dequantized_int8": b_metrics,
        "note": "This compares QBMM against dequantized INT8 arithmetic, not original BF16 model accuracy.",
    }
    for metrics in (a_metrics, b_metrics):
        if metrics["nrmse"] > args.max_nrmse or metrics["cosine"] < args.min_cosine:
            raise AssertionError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _indexed_weight_exists(model_dir: Path, name: str) -> bool:
    path = model_dir / "model.safetensors.index.json"
    return name in json.loads(path.read_text(encoding="utf-8"))["weight_map"]


def _weight_exists(model_dir: Path, name: str) -> bool:
    try:
        find_tensor_file(model_dir, name)
    except KeyError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--groups", type=int, help="Checkpoint o_groups; inferred from config.json if omitted")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-nrmse", type=float, default=0.02)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    args = parser.parse_args()
    if args.tokens <= 0 or args.layer < 0:
        parser.error("--tokens must be positive and --layer must be nonnegative")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
