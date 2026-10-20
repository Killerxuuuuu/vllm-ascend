# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Experimental A3 INT8 TQBMM: [T,G,K] @ [G,K,N] -> [T,G,N].

Inputs are already quantized. Dynamic activation quantization is deliberately
outside this operator. All operands must use ordinary ND storage, not NZ.
"""

import torch
import torch_npu
from vllm.triton_utils import tl, triton

# Worst case: K products of (-128) * (-128), accumulated in signed INT32.
MAX_INT8_ACCUMULATION_K = (2**31 - 1) // (128 * 128)
# Transposed INT8 loads can acquire 32-byte padding in the A3 vector lowering.
# Keep tiles small until larger configurations pass device correctness tests.
DEFAULT_BLOCK_N = 32
DEFAULT_BLOCK_K = 32
DEFAULT_GROUP_SIZE_M = 4


@triton.jit
def _transpose_quant_batch_matmul_kernel(
    x_ptr,
    w_ptr,
    sx_ptr,
    sw_ptr,
    y_ptr,
    T: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xt: tl.constexpr,
    stride_xg: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wg: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_sxt: tl.constexpr,
    stride_sxg: tl.constexpr,
    stride_swg: tl.constexpr,
    stride_swn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    TRANSPOSE_WEIGHT_TILE: tl.constexpr = False,
):
    # Group nearby M tiles when traversing N, improving weight cache locality.
    pid = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    num_m = tl.cdiv(T, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    window = GROUP_SIZE_M * num_n
    window_id = pid // window
    first_m = window_id * GROUP_SIZE_M
    active_m = tl.minimum(num_m - first_m, GROUP_SIZE_M)
    pid_m = first_m + (pid % window) % active_m
    pid_n = (pid % window) // active_m

    rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    cols = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    inner = tl.arange(0, BLOCK_K).to(tl.int64)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for step in range(tl.cdiv(K, BLOCK_K)):
        # Reconstruct addresses from the immutable bases, not loop-carried
        # pointer tensors. The same absolute K indices drive loads and masks.
        # A [G,N,K] backing allocation viewed as [G,K,N] has stride_wk=1,
        # stride_wn=physical K pitch; do not substitute the logical slice K.
        current_k = inner + step * BLOCK_K
        x = x_ptr + rows[:, None] * stride_xt + group * stride_xg + current_k[None, :] * stride_xk
        if TRANSPOSE_WEIGHT_TILE:
            # Experimental only: expose the contiguous physical K dimension
            # as the load's last axis. Transpose the *tile*, not the full weight.
            # Absolute offsets, INT8 values, masks and dot ordering are unchanged.
            w_nk = w_ptr + group * stride_wg + cols[:, None] * stride_wn + current_k[None, :] * stride_wk
            a = tl.load(x, mask=(rows[:, None] < T) & (current_k[None, :] < K), other=0)
            b_nk = tl.load(w_nk, mask=(cols[:, None] < N) & (current_k[None, :] < K), other=0)
            b = tl.trans(b_nk)
        else:
            # Keep the device-validated direct-load path as the default.
            w = w_ptr + group * stride_wg + current_k[:, None] * stride_wk + cols[None, :] * stride_wn
            a = tl.load(x, mask=(rows[:, None] < T) & (current_k[None, :] < K), other=0)
            b = tl.load(w, mask=(current_k[:, None] < K) & (cols[None, :] < N), other=0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)

    sx = tl.load(sx_ptr + rows * stride_sxt + group * stride_sxg, mask=rows < T, other=0)
    sw = tl.load(sw_ptr + group * stride_swg + cols * stride_swn, mask=cols < N, other=0)
    result = (accumulator.to(tl.float32) * sw[None, :]) * sx[:, None]
    # Store directly in token-major order: no [G,T,N] temporary or stack.
    y = y_ptr + (rows[:, None] * G + group) * N + cols[None, :]
    tl.store(y, result.to(y_ptr.dtype.element_ty), mask=(rows[:, None] < T) & (cols[None, :] < N))


def _validate_inputs(x, weight, x_scale, weight_scale, output_dtype):
    if x.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("TQBMM requires INT8 activation and weight tensors")
    if x.ndim != 3 or weight.ndim != 3:
        raise ValueError("TQBMM expects x=[T,G,K] and weight=[G,K,N]")
    tokens, groups, inner = x.shape
    if groups <= 0 or inner <= 0 or weight.shape[-1] <= 0:
        raise ValueError("TQBMM requires positive G, K and N")
    if weight.shape[:2] != (groups, inner):
        raise ValueError("TQBMM activation/weight G,K dimensions do not match")
    if inner > MAX_INT8_ACCUMULATION_K:
        raise ValueError("TQBMM K is too large to guarantee no INT32 accumulator overflow")
    if x_scale.shape != (tokens, groups) or weight_scale.shape != (groups, weight.shape[-1]):
        raise ValueError("TQBMM requires x_scale=[T,G] and weight_scale=[G,N]")
    if x_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        raise TypeError("TQBMM requires FP32 scales (not MX or packed scales)")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("TQBMM output dtype must be BF16, FP16 or FP32")
    if any(tensor.device != x.device for tensor in (weight, x_scale, weight_scale)):
        raise ValueError("All TQBMM tensors must be on the same device")


def _validate_num_stages(num_stages):
    if type(num_stages) is not int or num_stages not in (1, 2):
        raise ValueError("num_stages must be 1 (disable auto multi-buffering) or 2 (automatic mode)")


def transpose_quant_batch_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    block_shape: tuple[int, int, int] | None = None,
    num_stages: int = 1,
    transpose_weight_tile: bool = False,
) -> torch.Tensor:
    """INT8 matmul with per-token-per-group / per-group-per-channel scales.

    Supports strided ND inputs (including [G,N,K].transpose(1,2) weights),
    zero tokens, and masked M/N/K tails. Bias, offsets and MX scales are not
    supported. ``block_shape`` overrides the conservative, untuned defaults.
    ``num_stages=1`` disables Ascend automatic multi-buffering for correctness
    isolation; ``2`` restores automatic mode, not an exact two-buffer count.
    Disabling multi-buffering alone did not fix the reported strided multi-K
    failure. Addresses are now reconstructed per iteration for isolation;
    device validation is still required. No copies, CPU fallback or
    catch-and-retry path are added.
    ``transpose_weight_tile`` enables an unverified physical [N,K] tile load
    followed by a tile transpose. The validated direct load remains the default.
    """
    _validate_inputs(x, weight, x_scale, weight_scale, output_dtype)
    _validate_num_stages(num_stages)
    if type(transpose_weight_tile) is not bool:
        raise TypeError("transpose_weight_tile must be a bool")
    if x.device.type != "npu":
        raise ValueError("TQBMM is an Ascend NPU kernel; run it on A3")
    tokens, groups, inner = x.shape
    columns = weight.shape[-1]
    if block_shape is None:
        block_m = 16 if tokens <= 16 else 32 if tokens <= 32 else 64
        block_shape = (block_m, DEFAULT_BLOCK_N, DEFAULT_BLOCK_K)
    if len(block_shape) != 3 or any(
        not isinstance(size, int) or size < 16 or size & (size - 1) for size in block_shape
    ):
        raise ValueError("block_shape must contain three powers of two, each >= 16")
    block_m, block_n, block_k = block_shape
    if block_k < 32:
        raise ValueError("INT8 dot requires block K >= 32")
    output = torch.empty((tokens, groups, columns), dtype=output_dtype, device=x.device)
    if tokens == 0:
        return output
    if any(torch_npu.get_npu_format(tensor) not in (0, 2) for tensor in (x, weight, x_scale, weight_scale)):
        raise ValueError("TQBMM requires ND storage; NZ is not supported")
    grid = (triton.cdiv(tokens, block_m) * triton.cdiv(columns, block_n), groups)
    _transpose_quant_batch_matmul_kernel[grid](
        x,
        weight,
        x_scale,
        weight_scale,
        output,
        tokens,
        groups,
        inner,
        columns,
        *x.stride(),
        *weight.stride(),
        *x_scale.stride(),
        *weight_scale.stride(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=DEFAULT_GROUP_SIZE_M,
        TRANSPOSE_WEIGHT_TILE=transpose_weight_tile,
        num_stages=num_stages,
    )
    return output


def dynamic_quant_grouped_wo_a(
    activations, weight, weight_scale, *, block_shape=None, num_stages=1, transpose_weight_tile=False
):
    """Keep CANN dynamic quantization, then run one fused TQBMM for all groups."""
    if activations.ndim != 3 or activations.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("Triton wo_a expects BF16/FP16 activations [T,G,K]")
    tokens, groups, inner = activations.shape
    if groups <= 0 or inner <= 0:
        raise ValueError("Triton wo_a requires positive G and K")
    _validate_num_stages(num_stages)
    # Flattening T,G quantizes each row over K, exactly as the per-group path.
    if tokens:
        quantized, scales = torch_npu.npu_dynamic_quant(
            activations.contiguous().view(tokens * groups, inner), dst_type=torch.int8
        )
    else:
        quantized = torch.empty((0, inner), dtype=torch.int8, device=activations.device)
        scales = torch.empty((0,), dtype=torch.float32, device=activations.device)
    return transpose_quant_batch_matmul(
        quantized.view(tokens, groups, inner),
        weight,
        scales.reshape(tokens, groups).float(),
        weight_scale.reshape(groups, -1).float(),
        output_dtype=activations.dtype,
        block_shape=block_shape,
        num_stages=num_stages,
        transpose_weight_tile=transpose_weight_tile,
    )
