"""Enable DeepSeek-V4-Flash wo_a/wo_b W8A8 in a copied msModelSlim config.

Run this against msModelSlim's deepseek_v4_flash_w8a8.yaml. The source is never
modified; the output is deliberately placed beside it so relative dataset
paths keep their original meaning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def prepare_config(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    if destination.parent.resolve() != source.parent.resolve():
        raise ValueError("Output must be beside the original YAML to retain relative dataset paths")

    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    model_types = config.get("metadata", {}).get("verified_model_types", [])
    if "DeepSeek-V4-Flash" not in model_types:
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
    for name in ("*wo_a", "*wo_b"):
        if excluded.count(name) != 1:
            raise ValueError(f"Expected exactly one {name} exclusion")
    step["exclude"] = [name for name in excluded if name not in ("*wo_a", "*wo_b")]

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_config(args.source, args.output))


if __name__ == "__main__":
    main()
