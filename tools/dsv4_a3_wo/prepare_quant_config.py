"""Prepare experimental DeepSeek V4 Flash W8A8 configs for wo_a/wo_b.

The default mode keeps the existing preview recipe behavior. For the 0731
model, use --target-model-type DeepSeek-V4-Flash-0731 with msModelSlim's
deepseek_v4_flash_w4a4c8.yaml as the source. The 0731 mode changes its three
linear quantization steps to dynamic INT8 W8A8 and includes wo_a/wo_b. It is
an unverified experiment, not an official A3 recipe. The source is never
modified; the output stays beside it so relative dataset paths still work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

PREVIEW_MODEL_TYPE = "DeepSeek-V4-Flash"
RELEASE_MODEL_TYPE = "DeepSeek-V4-Flash-0731"
WO_EXCLUSIONS = frozenset(("*wo_a", "*wo_b"))


def _check_paths(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    if destination.parent.resolve() != source.parent.resolve():
        raise ValueError("Output must be beside the original YAML to retain relative dataset paths")


def _dynamic_int8_qconfig() -> dict:
    return {
        "act": {
            "scope": "per_token",
            "dtype": "int8",
            "symmetric": True,
            "method": "minmax",
        },
        "weight": {
            "scope": "per_channel",
            "dtype": "int8",
            "symmetric": True,
            "method": "minmax",
        },
    }


def prepare_config(source: Path, destination: Path) -> Path:
    _check_paths(source, destination)

    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    model_types = config.get("metadata", {}).get("verified_model_types", [])
    if PREVIEW_MODEL_TYPE not in model_types:
        raise ValueError("Expected a DeepSeek-V4-Flash config, not V4.1 or V4-Pro")

    attention_steps = [
        step
        for step in config["spec"]["process"]
        if step.get("type") == "linear_quant" and "*attn*" in step.get("include", [])
    ]
    if len(attention_steps) != 1:
        raise ValueError(f"Expected one attention linear_quant step; found {len(attention_steps)}")
    step = attention_steps[0]
    excluded = step.get("exclude", [])
    for name in WO_EXCLUSIONS:
        if excluded.count(name) != 1:
            raise ValueError(f"Expected exactly one {name} exclusion")
    step["exclude"] = [name for name in excluded if name not in WO_EXCLUSIONS]

    qconfig = step["qconfig"]
    if (
        qconfig["act"]["scope"] != "per_token"
        or qconfig["act"]["dtype"] != "int8"
        or qconfig["act"]["symmetric"] is not True
        or qconfig["weight"]["scope"] != "per_channel"
        or qconfig["weight"]["dtype"] != "int8"
        or qconfig["weight"]["symmetric"] is not True
    ):
        raise ValueError("Attention qconfig is not symmetric W8A8 dynamic per-token/per-channel")
    config["metadata"]["config_id"] = "deepseek_v4_flash_w8a8_wo_qbmm_probe"
    destination.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return destination


def prepare_0731_config(source: Path, destination: Path, *, quantize_wo: bool = True) -> Path:
    """Derive experimental A3 INT8 W8A8 from the official 0731 recipe.

    The official recipe's sensitive-layer and MTP exclusions are preserved.
    No claim is made that QuaRot/SmoothQuant or full-model accuracy is verified.
    """
    _check_paths(source, destination)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config.get("metadata", {}).get("verified_model_types") != [RELEASE_MODEL_TYPE]:
        raise ValueError("Expected the official DeepSeek-V4-Flash-0731 recipe")

    steps = config["spec"]["process"]
    if len(steps) != 3 or any(step.get("type") != "linear_quant" for step in steps):
        raise ValueError("Expected the three linear_quant steps of the 0731 recipe")
    expected_includes = [["*attn*"], ["*ffn*"], ["*ffn.shared_experts*"]]
    if [step.get("include") for step in steps] != expected_includes:
        raise ValueError("Unexpected 0731 attention or FFN include patterns")

    attention_exclusions = steps[0].get("exclude", [])
    if any(attention_exclusions.count(name) != 1 for name in WO_EXCLUSIONS):
        raise ValueError("Expected one wo_a and one wo_b exclusion in the 0731 recipe")
    if any("*mtp*" not in step.get("exclude", []) for step in steps):
        raise ValueError("Expected MTP to remain excluded from all 0731 quantization steps")

    for step in steps:
        step["qconfig"] = _dynamic_int8_qconfig()
    if quantize_wo:
        steps[0]["exclude"] = [name for name in attention_exclusions if name not in WO_EXCLUSIONS]

    metadata = config["metadata"]
    suffix = "with_wo" if quantize_wo else "without_wo"
    metadata["config_id"] = f"deepseek_v4_flash_0731_a3_w8a8_int8_{suffix}_experimental"
    metadata["score"] = 0
    metadata["label"].update(w_bit=8, a_bit=8, is_sparse=False, kv_cache=False)
    metadata["verified_model_types"] = []
    metadata["verified_tags"] = {}

    destination.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-model-type",
        choices=(PREVIEW_MODEL_TYPE, RELEASE_MODEL_TYPE),
        default=PREVIEW_MODEL_TYPE,
    )
    parser.add_argument(
        "--exclude-wo",
        action="store_true",
        help="In 0731 mode, generate a W8A8 control recipe that keeps wo_a/wo_b unquantized.",
    )
    args = parser.parse_args()
    if args.target_model_type == RELEASE_MODEL_TYPE:
        output = prepare_0731_config(args.source, args.output, quantize_wo=not args.exclude_wo)
    else:
        if args.exclude_wo:
            parser.error("--exclude-wo is only valid for the 0731 target model")
        output = prepare_config(args.source, args.output)
    print(output)


if __name__ == "__main__":
    main()
