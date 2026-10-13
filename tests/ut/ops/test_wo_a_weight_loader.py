# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch

from vllm_ascend.ops.linear import AscendColumnParallelLinear


def test_grouped_wo_a_reload_does_not_reshape_quantization_params():
    layer = AscendColumnParallelLinear.__new__(AscendColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.self_attn.wo_a"
    layer.n_local_groups = 2
    layer.o_lora_rank = 3
    layer.weight = torch.nn.Parameter(torch.empty(2, 4, 3, dtype=torch.int8), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.empty(6), requires_grad=False)
    layer.weight_offset = torch.nn.Parameter(torch.empty(6), requires_grad=False)
    weight = torch.arange(24, dtype=torch.int8).reshape(6, 4)
    scale = torch.full((6,), 0.5)
    offset = torch.zeros(6)

    def load_parameter(_layer, param, loaded_weight):
        param.data.copy_(loaded_weight)

    with (
        patch("vllm_ascend.ops.linear._should_reshape_wo_a_to_3d", return_value=True),
        patch.object(
            AscendColumnParallelLinear.__mro__[1],
            "weight_loader",
            autospec=True,
            side_effect=load_parameter,
        ) as base_loader,
    ):
        layer.weight_loader(layer.weight, weight)
        layer.weight_loader(layer.weight_scale, scale)
        layer.weight_loader(layer.weight_offset, offset)

    torch.testing.assert_close(layer.weight, weight.reshape(2, 3, 4).transpose(1, 2))
    torch.testing.assert_close(layer.weight_scale, scale)
    torch.testing.assert_close(layer.weight_offset, offset)
    assert base_loader.call_count == 2
