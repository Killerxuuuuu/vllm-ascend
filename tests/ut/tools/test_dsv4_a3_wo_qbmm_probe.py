from pathlib import Path

import pytest
import torch
import yaml

from tools.dsv4_a3_wo.prepare_quant_config import prepare_0731_config, prepare_config
from tools.dsv4_a3_wo.qbmm_probe import validate_weights


def test_prepare_config_enables_only_flash_wo_layers(tmp_path: Path) -> None:
    source = tmp_path / "deepseek_v4_flash_w8a8.yaml"
    output = tmp_path / "deepseek_v4_flash_w8a8_wo.yaml"
    source.write_text(
        """metadata:
  config_id: deepseek_v4_flash_w8a8
  verified_model_types: [DeepSeek-V4-Flash]
spec:
  process:
    - type: linear_quant
      include: ['*attn*']
      exclude: ['*wo_a', '*wo_b', '*compressor.wkv']
      qconfig:
        act: {scope: per_token, dtype: int8, symmetric: true}
        weight: {scope: per_channel, dtype: int8, symmetric: true}
""",
        encoding="utf-8",
    )
    prepare_config(source, output)
    modified = yaml.safe_load(output.read_text(encoding="utf-8"))
    original = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert original["spec"]["process"][0]["exclude"] == ["*wo_a", "*wo_b", "*compressor.wkv"]
    assert modified["spec"]["process"][0]["exclude"] == ["*compressor.wkv"]
    with pytest.raises(FileExistsError):
        prepare_config(source, output)


def test_prepare_config_rejects_wrong_model(tmp_path: Path) -> None:
    source = tmp_path / "wrong.yaml"
    source.write_text("metadata: {verified_model_types: [DeepSeek-V4.1-Flash]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="DeepSeek-V4-Flash"):
        prepare_config(source, tmp_path / "probe.yaml")


@pytest.mark.parametrize("quantize_wo", [False, True])
def test_prepare_0731_config_quantizes_linear_steps_and_preserves_exclusions(
    tmp_path: Path, quantize_wo: bool
) -> None:
    source = tmp_path / "deepseek_v4_flash_w4a4c8.yaml"
    output = tmp_path / "deepseek_v4_flash_0731_w8a8.yaml"
    original = {
        "apiversion": "modelslim_v1",
        "metadata": {
            "config_id": "deepseek_v4_flash_w4a4c8",
            "score": 90,
            "verified_model_types": ["DeepSeek-V4-Flash-0731"],
            "verified_tags": {"DeepSeek-V4-Flash-0731": [["vLLM-Ascend", "Ascend_950"]]},
            "label": {"w_bit": 4, "a_bit": 4, "is_sparse": False, "kv_cache": True},
        },
        "spec": {
            "process": [
                {
                    "type": "linear_quant",
                    "include": ["*attn*"],
                    "exclude": ["*wo_a", "*wo_b", "*compressor.wkv", "*mtp*"],
                    "qconfig": {"act": {"dtype": "mxfp8"}, "weight": {"dtype": "mxfp8"}},
                },
                {
                    "type": "linear_quant",
                    "include": ["*ffn*"],
                    "exclude": ["*gate", "*shared_experts*", "*mtp*"],
                    "qconfig": {"act": {"dtype": "mxfp4"}, "weight": {"dtype": "mxfp4"}},
                },
                {
                    "type": "linear_quant",
                    "include": ["*ffn.shared_experts*"],
                    "exclude": ["*mtp*"],
                    "qconfig": {"act": {"dtype": "mxfp8"}, "weight": {"dtype": "mxfp8"}},
                },
            ],
            "dataset": "mix_calib.jsonl",
            "save": [{"type": "ascendv1_saver"}],
        },
    }
    source.write_text(yaml.safe_dump(original), encoding="utf-8")

    prepare_0731_config(source, output, quantize_wo=quantize_wo)

    result = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert yaml.safe_load(source.read_text(encoding="utf-8")) == original
    assert result["metadata"]["verified_model_types"] == []
    assert result["metadata"]["verified_tags"] == {}
    assert result["metadata"]["label"] == {
        "w_bit": 8,
        "a_bit": 8,
        "is_sparse": False,
        "kv_cache": False,
    }
    assert result["spec"]["dataset"] == "mix_calib.jsonl"
    steps = result["spec"]["process"]
    assert steps[0]["exclude"] == (
        ["*compressor.wkv", "*mtp*"] if quantize_wo else original["spec"]["process"][0]["exclude"]
    )
    assert steps[1]["exclude"] == original["spec"]["process"][1]["exclude"]
    assert steps[2]["exclude"] == original["spec"]["process"][2]["exclude"]
    for step in steps:
        assert step["qconfig"]["act"] == {
            "scope": "per_token",
            "dtype": "int8",
            "symmetric": True,
            "method": "minmax",
        }
        assert step["qconfig"]["weight"] == {
            "scope": "per_channel",
            "dtype": "int8",
            "symmetric": True,
            "method": "minmax",
        }
    with pytest.raises(FileExistsError):
        prepare_0731_config(source, output)


def test_prepare_0731_config_rejects_preview_recipe(tmp_path: Path) -> None:
    source = tmp_path / "preview.yaml"
    source.write_text("metadata: {verified_model_types: [DeepSeek-V4-Flash]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0731"):
        prepare_0731_config(source, tmp_path / "output.yaml")


def test_validate_quantized_wo_shapes() -> None:
    a = torch.empty((12, 64), dtype=torch.int8)
    a_scale = torch.ones((12, 1), dtype=torch.float32)
    b = torch.empty((128, 12), dtype=torch.int8)
    b_scale = torch.ones((128, 1), dtype=torch.float32)
    assert validate_weights(a, a_scale, b, b_scale, groups=3) == (64, 4, 128)
    with pytest.raises(ValueError, match="Per-channel scales"):
        validate_weights(a, a_scale[:-1], b, b_scale, groups=3)
    with pytest.raises(ValueError, match="Incompatible"):
        validate_weights(a, a_scale, b, b_scale, groups=5)
    with pytest.raises(TypeError, match="INT8"):
        validate_weights(a.float(), a_scale, b, b_scale, groups=3)
