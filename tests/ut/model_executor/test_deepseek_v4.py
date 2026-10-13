from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.deepseek_v4 import model as deepseek_v4


class StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


@pytest.mark.parametrize("result_count", [3, 4])
def test_hc_pre_accepts_both_operator_signatures(monkeypatch, result_count):
    layer = deepseek_v4.DeepseekV2DecoderLayer.__new__(deepseek_v4.DeepseekV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.hc_mult = 4
    layer.hc_sinkhorn_iters = 20
    layer.norm_eps = 1e-6
    layer.hc_eps = 1e-6

    x, hc_fn, hc_scale, hc_base = (torch.empty(1) for _ in range(4))
    outputs = tuple(torch.empty(1) for _ in range(result_count))
    calls = []

    def fake_hc_pre(*args, **kwargs):
        calls.append((args, kwargs))
        return outputs

    monkeypatch.setattr(deepseek_v4.torch.ops._C_ascend, "npu_hc_pre_v2", fake_hc_pre, raising=False)

    result = layer.hc_pre(x, hc_fn, hc_scale, hc_base)

    assert len(result) == 3
    assert all(actual is expected for actual, expected in zip(result, outputs[:3]))
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert all(actual is expected for actual, expected in zip(args, (x, hc_fn, hc_scale, hc_base)))
    assert kwargs == {
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "norm_eps": 1e-6,
        "hc_eps": 1e-6,
    }


def test_routed_moe_receives_configured_swiglu_limit(monkeypatch):
    fused_moe_kwargs = {}

    class StubFusedMoEFactory(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            fused_moe_kwargs.update(kwargs)

    monkeypatch.setattr(deepseek_v4, "FusedMoEFactory", StubFusedMoEFactory)
    monkeypatch.setattr(deepseek_v4, "ReplicatedLinear", StubModule)
    monkeypatch.setattr(deepseek_v4, "DeepseekV2MLP", StubModule)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        deepseek_v4,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1), rank_in_group=0),
    )
    monkeypatch.setattr(deepseek_v4, "get_ascend_config", lambda: SimpleNamespace(mix_placement=False))
    monkeypatch.setattr(deepseek_v4.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False)
    monkeypatch.setattr(deepseek_v4.rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: False)

    config = SimpleNamespace(
        hidden_act="silu",
        hidden_size=16,
        moe_intermediate_size=8,
        n_group=1,
        n_routed_experts=8,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_experts_per_tok=2,
        num_hash_layers=0,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        swiglu_limit=10.0,
        topk_group=1,
    )
    parallel_config = SimpleNamespace(
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        use_sequence_parallel_moe=False,
    )

    deepseek_v4.DeepseekV4MoE(config, parallel_config, prefix="model.layers.1.mlp")

    assert fused_moe_kwargs["swiglu_limit"] == config.swiglu_limit
