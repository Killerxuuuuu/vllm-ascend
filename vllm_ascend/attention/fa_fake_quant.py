# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Floating-point INT4 loss simulation; not a packed INT4 implementation."""

import torch

INT4_SYMMETRIC_MAX = 7


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Apply a normalized, self-inverse Walsh-Hadamard transform per head.

    Keep the butterfly in FP32 so rotating BF16 Q/K/V does not accumulate
    rounding error at every stage. The last dimension must be a power of two.
    """
    if x.ndim == 0:
        raise ValueError("Hadamard rotation requires a nonempty last dimension")
    width = x.shape[-1]
    if width == 0 or width & (width - 1):
        raise ValueError(f"Hadamard rotation requires a power-of-two head dimension, got {width}")
    if x.numel() == 0:
        return x.clone()

    result = x.float()
    stride = 1
    while stride < width:
        pairs = result.reshape(*x.shape[:-1], width // (2 * stride), 2, stride)
        left, right = pairs.unbind(dim=-2)
        result = torch.stack((left + right, left - right), dim=-2).reshape(*x.shape)
        stride *= 2
    return (result * (width**-0.5)).to(x.dtype)


def fake_quantize_int4(x: torch.Tensor) -> torch.Tensor:
    """Round each last-axis vector to symmetric INT4, then dequantize.

    One FP32 absmax scale per token/head (the entire last dimension),
    zero point 0, range [-7, 7], round-to-nearest-even. Inputs must be finite.
    RoPE and non-RoPE dimensions share the scale. The result retains the
    input dtype/device/shape, but does not alias or modify the input.
    No scale is stored in the KV cache: the returned values are floating point.
    """
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"INT4 FA fake quantization requires FP16/BF16/FP32, got {x.dtype}")
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("INT4 FA fake quantization requires a nonempty last dimension")
    if x.numel() == 0:
        return x.clone()
    values = x.float()
    absmax = values.abs().amax(dim=-1, keepdim=True)
    # Avoid division by zero for zero rows and FP32 scale underflow. This is
    # entirely device-side; do not use .item() in the attention hot path.
    scale = (absmax / INT4_SYMMETRIC_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    codes = torch.round(values / scale).clamp(-INT4_SYMMETRIC_MAX, INT4_SYMMETRIC_MAX)
    return (codes * scale).to(x.dtype)
