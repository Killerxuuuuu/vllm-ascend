# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch

from vllm_ascend.ops.linear import AscendColumnParallelLinear
from vllm_ascend.utils import AscendDeviceType


def test_a3_wo_a_quant_params_use_regular_weight_loader_after_weight_is_grouped():
    layer = AscendColumnParallelLinear.__new__(AscendColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.self_attn.wo_a"
    layer.n_local_groups = 1
    layer.o_lora_rank = 1024
    layer.weight = torch.nn.Parameter(torch.empty(1024, 4096, dtype=torch.int8), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.empty(1024, 1), requires_grad=False)
    layer.weight_offset = torch.nn.Parameter(torch.empty(1024, 1), requires_grad=False)
    local_weight = torch.ones_like(layer.weight)
    local_scale = torch.full_like(layer.weight_scale, 0.5)
    local_offset = torch.zeros_like(layer.weight_offset)

    def load_local_weight(_layer, param, loaded_weight):
        param.data.copy_(loaded_weight)

    with (
        patch("vllm_ascend.ops.linear.get_ascend_device_type", return_value=AscendDeviceType.A3),
        patch.object(
            AscendColumnParallelLinear.__mro__[1],
            "weight_loader",
            autospec=True,
            side_effect=load_local_weight,
        ) as base_loader,
    ):
        layer.weight_loader(layer.weight, local_weight)
        assert layer.weight.shape == (1, 4096, 1024)

        layer.weight_loader(layer.weight_scale, local_scale)
        layer.weight_loader(layer.weight_offset, local_offset)

    assert layer.weight_scale.shape == (1024, 1)
    torch.testing.assert_close(layer.weight_scale, local_scale)
    assert layer.weight_offset.shape == (1024, 1)
    torch.testing.assert_close(layer.weight_offset, local_offset)
    assert base_loader.call_count == 3
