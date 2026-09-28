# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.fa_fake_quant import fake_quantize_int4, hadamard_rotate


@pytest.mark.parametrize("width", [8, 512])
def test_hadamard_rotation_is_self_inverse_and_preserves_qk_scores(width):
    torch.manual_seed(0)
    q = torch.randn(2, 4, width)
    k = torch.randn(3, width)

    rotated_q = hadamard_rotate(q)
    rotated_k = hadamard_rotate(k)

    torch.testing.assert_close(hadamard_rotate(rotated_q), q, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        torch.einsum("thd,sd->ths", rotated_q, rotated_k),
        torch.einsum("thd,sd->ths", q, k),
        rtol=1e-4,
        atol=1e-4,
    )


def test_shared_kv_rotation_is_undone_on_attention_output():
    torch.manual_seed(1)
    q = torch.randn(2, 4, 512)
    shared_kv = torch.randn(3, 512)

    scores = torch.einsum("thd,sd->ths", q, shared_kv)
    reference = torch.einsum("ths,sd->thd", scores.softmax(-1), shared_kv)

    rotated_q = hadamard_rotate(q)
    rotated_kv = hadamard_rotate(shared_kv)
    rotated_scores = torch.einsum("thd,sd->ths", rotated_q, rotated_kv)
    rotated_output = torch.einsum("ths,sd->thd", rotated_scores.softmax(-1), rotated_kv)

    torch.testing.assert_close(hadamard_rotate(rotated_output), reference, rtol=1e-4, atol=1e-4)


def test_rotated_int4_qdq_keeps_shape_dtype_and_finite_values():
    x = torch.randn(2, 4, 512, dtype=torch.bfloat16)
    result = fake_quantize_int4(hadamard_rotate(x))

    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert torch.isfinite(result).all()


def test_hadamard_rotation_rejects_non_power_of_two_width():
    with pytest.raises(ValueError, match="power-of-two"):
        hadamard_rotate(torch.ones(2, 3))
    with pytest.raises(ValueError, match="nonempty last dimension"):
        hadamard_rotate(torch.tensor(1.0))
