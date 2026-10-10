from pathlib import Path

import pytest
import torch
import yaml

from tools.dsv4_a3_wo.prepare_quant_config import prepare_config
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
