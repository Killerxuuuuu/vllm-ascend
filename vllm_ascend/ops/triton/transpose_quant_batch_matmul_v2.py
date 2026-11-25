# SPDX-License-Identifier: Apache-2.0
"""A3 structural experiment: persistent tiles, bounded K lookahead, one acc.

Fully tiled shapes use wide panels; boundary shapes load dot-width operands.
Physical L1/L0 placement and overlap must be verified in IR/profiling.
No production dispatch or V1 defaults are changed.
"""

from dataclasses import dataclass

import torch
import torch_npu
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.transpose_quant_batch_matmul import PreparedDynamicWoA, PreparedTQBMM
from vllm_ascend.ops.triton.triton_utils import extract_slice

MAX_LOGICAL_ACCUMULATOR_BYTES = 64 * 1024
MAX_PANEL_K = 512
GROUP_SIZE_M = 4
# Experimental macro only. For a 128x128 INT8 output tile LocalMmad's
# 32-KiB half of each L0 operand buffer holds 256 K elements per row.
L0_PAIR_CONTROL_TILE = (128, 128, 128)
L0_PAIR_PANEL_K = 512


@triton.jit
def _v2_coordinates(
    tile,
    T: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr = GROUP_SIZE_M,
):
    num_m = tl.cdiv(T, BM)
    num_n = tl.cdiv(N, BN)
    window = GROUP_M * num_n
    first_m = (tile // window) * GROUP_M
    active_m = tl.minimum(num_m - first_m, GROUP_M)
    row_tile = first_m + (tile % window) % active_m
    col_tile = (tile % window) // active_m
    rows = (row_tile * BM + tl.arange(0, BM)).to(tl.int64)
    cols = (col_tile * BN + tl.arange(0, BN)).to(tl.int64)
    return rows, cols


@triton.jit
def _v2_load_operands(
    x_ptr,
    w_ptr,
    rows,
    cols,
    group,
    start_k,
    T: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    SXT: tl.constexpr,
    SXG: tl.constexpr,
    SXK: tl.constexpr,
    SWG: tl.constexpr,
    SWK: tl.constexpr,
    SWN: tl.constexpr,
    WIDTH_K: tl.constexpr,
):
    # Reconstruct every GM address from immutable bases and absolute K.
    k = start_k + tl.arange(0, WIDTH_K).to(tl.int64)
    x_address = x_ptr + rows[:, None] * SXT + group * SXG + k[None, :] * SXK
    w_address = w_ptr + group * SWG + k[:, None] * SWK + cols[None, :] * SWN
    a = tl.load(x_address, mask=(rows[:, None] < T) & (k[None, :] < K), other=0)
    b = tl.load(w_address, mask=(k[:, None] < K) & (cols[None, :] < N), other=0)
    return a, b


@triton.jit
def _v2_dot_full_panel(a_panel, b_panel, acc, DOT_K: tl.constexpr, PANEL_K: tl.constexpr):
    if PANEL_K == DOT_K:
        return tl.dot(a_panel, b_panel, acc, out_dtype=tl.int32)
    else:
        for micro in tl.static_range(PANEL_K // DOT_K):
            a = extract_slice(a_panel, (0, micro * DOT_K), (a_panel.shape[0], DOT_K), (1, 1))
            b = extract_slice(b_panel, (micro * DOT_K, 0), (DOT_K, b_panel.shape[1]), (1, 1))
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        return acc


@triton.jit
def _v2_accumulate_panel(
    x_ptr,
    w_ptr,
    rows,
    cols,
    group,
    start_k,
    acc,
    T: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    SXT: tl.constexpr,
    SXG: tl.constexpr,
    SXK: tl.constexpr,
    SWG: tl.constexpr,
    SWK: tl.constexpr,
    SWN: tl.constexpr,
    DOT_K: tl.constexpr,
    PANEL_K: tl.constexpr,
    ACTIVE_K: tl.constexpr,
    PREFETCH: tl.constexpr,
    WIDE: tl.constexpr,
):
    if WIDE:
        # Preserve the original bulk loads and full-panel lookahead on the
        # aligned path, including the single unpaired full panel at the end.
        a_panel, b_panel = _v2_load_operands(
            x_ptr, w_ptr, rows, cols, group, start_k, T, K, N, SXT, SXG, SXK, SWG, SWK, SWN, PANEL_K
        )
        if ACTIVE_K == 2 * PANEL_K:
            a_next_panel, b_next_panel = _v2_load_operands(
                x_ptr, w_ptr, rows, cols, group, start_k + PANEL_K, T, K, N, SXT, SXG, SXK, SWG, SWK, SWN, PANEL_K
            )
        acc = _v2_dot_full_panel(a_panel, b_panel, acc, DOT_K, PANEL_K)
        if ACTIVE_K == 2 * PANEL_K:
            acc = _v2_dot_full_panel(a_next_panel, b_next_panel, acc, DOT_K, PANEL_K)
    else:
        # Never slice a wider loaded tensor or carry operand buffers around the K
        # loop. Each dot sees its native A[BM,DOT_K], B[DOT_K,BN] directly from GM.
        micro_tiles: tl.constexpr = (ACTIVE_K + DOT_K - 1) // DOT_K
        issue_width: tl.constexpr = 2 if PREFETCH else 1
        for part in tl.static_range(micro_tiles // issue_width):
            k0 = start_k + part * issue_width * DOT_K
            a, b = _v2_load_operands(x_ptr, w_ptr, rows, cols, group, k0, T, K, N, SXT, SXG, SXK, SWG, SWK, SWN, DOT_K)
            if PREFETCH:
                a_next, b_next = _v2_load_operands(
                    x_ptr, w_ptr, rows, cols, group, k0 + DOT_K, T, K, N, SXT, SXG, SXK, SWG, SWK, SWN, DOT_K
                )
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            if PREFETCH:
                acc = tl.dot(a_next, b_next, acc, out_dtype=tl.int32)
        if PREFETCH and micro_tiles % 2:
            a_last, b_last = _v2_load_operands(
                x_ptr,
                w_ptr,
                rows,
                cols,
                group,
                start_k + (micro_tiles - 1) * DOT_K,
                T,
                K,
                N,
                SXT,
                SXG,
                SXK,
                SWG,
                SWK,
                SWN,
                DOT_K,
            )
            acc = tl.dot(a_last, b_last, acc, out_dtype=tl.int32)
    return acc


@triton.jit
def _transpose_quant_batch_matmul_v2_kernel(
    x_ptr,
    w_ptr,
    sx_ptr,
    sw_ptr,
    y_ptr,
    T: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    SXT: tl.constexpr,
    SXG: tl.constexpr,
    SXK: tl.constexpr,
    SWG: tl.constexpr,
    SWK: tl.constexpr,
    SWN: tl.constexpr,
    SSXT: tl.constexpr,
    SSXG: tl.constexpr,
    SSWG: tl.constexpr,
    SSWN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    DOT_K: tl.constexpr,
    PANEL_K: tl.constexpr,
    PROGRAMS: tl.constexpr,
    PREFETCH: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    num_tiles: tl.constexpr = ((T + BM - 1) // BM) * ((N + BN - 1) // BN)
    # Shape specialization, not a runtime branch per tile. With ragged M/N
    # even a full K panel contains masked rows/columns in the persistent loop.
    # Combine constexpr booleans with binary operators: the A3 frontend
    # rejects a single Python BoolOp with three operands before TTIR exists.
    wide: tl.constexpr = (T % BM == 0) & (N % BN == 0) & (K % PANEL_K == 0)
    step_k: tl.constexpr = 2 * PANEL_K if PREFETCH else PANEL_K
    full_steps: tl.constexpr = K // step_k
    tail_k: tl.constexpr = K % step_k
    for tile in range(pid, num_tiles, PROGRAMS):
        rows, cols = _v2_coordinates(tile, T, N, BM, BN)
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        if full_steps:
            for step in range(full_steps):
                acc = _v2_accumulate_panel(
                    x_ptr,
                    w_ptr,
                    rows,
                    cols,
                    group,
                    step * step_k,
                    acc,
                    T,
                    K,
                    N,
                    SXT,
                    SXG,
                    SXK,
                    SWG,
                    SWK,
                    SWN,
                    DOT_K,
                    PANEL_K,
                    step_k,
                    PREFETCH,
                    wide,
                )
        if tail_k:
            acc = _v2_accumulate_panel(
                x_ptr,
                w_ptr,
                rows,
                cols,
                group,
                full_steps * step_k,
                acc,
                T,
                K,
                N,
                SXT,
                SXG,
                SXK,
                SWG,
                SWK,
                SWN,
                DOT_K,
                PANEL_K,
                tail_k,
                PREFETCH,
                wide,
            )
        sx = tl.load(sx_ptr + rows * SSXT + group * SSXG, mask=rows < T, other=0)
        sw = tl.load(sw_ptr + group * SSWG + cols * SSWN, mask=cols < N, other=0)
        result = (acc.to(tl.float32) * sw[None, :]) * sx[:, None]
        address = y_ptr + (rows[:, None] * G + group) * N + cols[None, :]
        tl.store(address, result.to(y_ptr.dtype.element_ty), mask=(rows[:, None] < T) & (cols[None, :] < N))


def validate_v2_options(block_shape, panel_k, prefetch, programs):
    if len(block_shape) != 3 or any(type(v) is not int or v < 16 or v > 128 or v & (v - 1) for v in block_shape):
        raise ValueError("V2 dot tiles must be powers of two in [16,128]")
    bm, bn, dot_k = block_shape
    if dot_k < 32:
        raise ValueError("V2 INT8 dot K must be >=32")
    if type(panel_k) is not int or panel_k > MAX_PANEL_K or panel_k not in (dot_k, 2 * dot_k, 4 * dot_k):
        raise ValueError("V2 panel K must be 1/2/4 times dot K and <=512")
    if bm * bn * 4 > MAX_LOGICAL_ACCUMULATOR_BYTES:
        raise ValueError("V2 logical accumulator exceeds the 64 KiB experimental budget")
    if type(prefetch) is not bool:
        raise TypeError("V2 prefetch must be bool")
    if type(programs) is not int or programs < 0:
        raise ValueError("V2 programs must be 0 (query AIC cores) or a positive integer")


def _v2_programs(device, requested):
    if requested:
        return requested
    # Same A2/A3 property used by Triton-Ascend's persistent-matmul tutorial.
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(device.index)
    cores = properties.get("num_aicore")
    if type(cores) is not int or cores <= 0:
        raise RuntimeError("Cannot query num_aicore; pass an explicit --programs value")
    return cores


@dataclass(frozen=True)
class V2Config:
    block_shape: tuple = (64, 64, 64)
    panel_k: int = 128
    prefetch: bool = True
    programs: int = 0
    num_stages: int = 2

    def __post_init__(self):
        object.__setattr__(self, "block_shape", tuple(self.block_shape))
        validate_v2_options(self.block_shape, self.panel_k, self.prefetch, self.programs)
        if type(self.num_stages) is not int or self.num_stages not in (1, 2):
            raise ValueError("V2 num_stages must be 1 or 2")


def _launch_v2(arguments, config, cores):
    tokens, groups, _, columns = arguments[5:9]
    if not tokens:
        return
    bm, bn, dot_k = config.block_shape
    programs = min(cores, triton.cdiv(tokens, bm) * triton.cdiv(columns, bn))
    _transpose_quant_batch_matmul_v2_kernel[(programs, groups)](
        *arguments,
        BM=bm,
        BN=bn,
        DOT_K=dot_k,
        PANEL_K=config.panel_k,
        PROGRAMS=programs,
        PREFETCH=config.prefetch,
        num_stages=config.num_stages,
    )


@triton.jit
def _transpose_quant_batch_matmul_v2_kernel_macro(
    x_ptr,
    w_ptr,
    sx_ptr,
    sw_ptr,
    y_ptr,
    T: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    SXT: tl.constexpr,
    SXG: tl.constexpr,
    SXK: tl.constexpr,
    SWG: tl.constexpr,
    SWK: tl.constexpr,
    SWN: tl.constexpr,
    SSXT: tl.constexpr,
    SSXG: tl.constexpr,
    SSWG: tl.constexpr,
    SSWN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TILE_MAJOR_W: tl.constexpr = False,
):
    # One program owns one output rectangle. Subblocks of this wider dot
    # consume common A rows/B columns, rather than reloading them per small tile.
    pid = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    ntiles: tl.constexpr = (N + BN - 1) // BN
    m0 = (pid // ntiles) * BM
    n0 = (pid % ntiles) * BN
    a0 = tl.make_block_ptr(x_ptr + group * SXG, (T, K), (SXT, SXK), (m0, 0), (BM, BK), (1, 0))
    kn_layout: tl.constexpr = TILE_MAJOR_W | (SWN == 1)
    if TILE_MAJOR_W:
        # Physical [G, ceil(N/BN), K, BN]. A single K step now spans
        # BK contiguous rows of BN bytes; the last N tile is zero padded.
        panel = w_ptr + (group * ntiles + pid % ntiles) * K * BN
        b0 = tl.make_block_ptr(panel, (K, BN), (BN, 1), (0, 0), (BK, BN), (1, 0))
    elif SWN == 1:
        b0 = tl.make_block_ptr(w_ptr + group * SWG, (K, N), (SWK, SWN), (0, n0), (BK, BN), (1, 0))
    else:
        # Physical [N,K] view: the contiguous dimension is last on both input DMAs.
        b0 = tl.make_block_ptr(w_ptr + group * SWG, (N, K), (SWN, SWK), (n0, 0), (BN, BK), (1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for step in range((K + BK - 1) // BK):
        k0 = step * BK
        # Absolute advances; do not carry a block pointer or an NZ panel through the loop.
        ap = tl.advance(a0, (0, k0))
        if (T % BM == 0) & (K % BK == 0):
            a = tl.load(ap)
        else:
            a = tl.load(ap, boundary_check=(0, 1), padding_option="zero")
        if kn_layout:
            bp = tl.advance(b0, (k0, 0))
        else:
            bp = tl.advance(b0, (0, k0))
        if (N % BN == 0) & (K % BK == 0):
            b = tl.load(bp)
        else:
            b = tl.load(bp, boundary_check=(0, 1), padding_option="zero")
        if not kn_layout:
            b = tl.trans(b)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)

    cols = n0.to(tl.int64) + tl.arange(0, BN)
    sw = tl.load(sw_ptr + group * SSWG + cols * SSWN, mask=cols < N, other=0)
    # Like the Ascend Triton matmul tutorial, convert/write row slices rather
    # than materializing a second full macro tile of FP32 epilogue values.
    half_m: tl.constexpr = BM // 2
    for part in tl.static_range(2):
        rows = m0.to(tl.int64) + part * half_m + tl.arange(0, half_m)
        sx = tl.load(sx_ptr + rows * SSXT + group * SSXG, mask=rows < T, other=0)
        sub = extract_slice(acc, (part * half_m, 0), (half_m, BN), (1, 1))
        result = (sub.to(tl.float32) * sw[None, :]) * sx[:, None]
        address = y_ptr + (rows[:, None] * G + group) * N + cols[None, :]
        tl.store(address, result.to(y_ptr.dtype.element_ty), mask=(rows[:, None] < T) & (cols[None, :] < N))


def _pack_macro_weight(weight, bn):
    """Snapshot logical INT8 W[G,K,N] as contiguous [G,ceil(N/BN),K,BN] panels."""
    groups, inner, columns = weight.shape
    full, tail = divmod(columns, bn)
    panels = full + int(tail != 0)
    # Allocate base format, including for a transposed/sliced source. Even a
    # single panel must own its storage: later source updates cannot change it.
    packed = weight.new_empty((groups * panels * inner * bn,)).reshape(groups, panels, inner, bn)
    if full:
        source = weight[:, :, : full * bn].reshape(groups, inner, full, bn).swapaxes(1, 2)
        packed[:, :full].copy_(source)
    if tail:
        packed[:, full, :, :tail].copy_(weight[:, :, full * bn :])
        packed[:, full, :, tail:].zero_()
    return packed


class PreparedMacroTQBMMV2(PreparedTQBMM):
    """Fixed-buffer macro plan. tile_major_weight snapshots W once, outside launch/capture.

    Rebuild that opt-in plan when weights change. X and scales still refer to live buffers.
    """

    def __init__(
        self,
        x,
        weight,
        sx,
        sw,
        *,
        block_shape,
        num_stages,
        output_dtype=torch.bfloat16,
        out=None,
        tile_major_weight=False,
    ):
        if type(tile_major_weight) is not bool:
            raise TypeError("tile_major_weight must be bool")
        if len(block_shape) != 3:
            raise ValueError("Macro kernel requires (BM,BN,BK)")
        validate_v2_options(block_shape, block_shape[2], False, 0)
        super().__init__(
            x, weight, sx, sw, block_shape=block_shape, num_stages=num_stages, output_dtype=output_dtype, out=out
        )
        if x.stride()[-1] != 1 or min(weight.stride()[1:]) != 1:
            raise ValueError("Macro block IO requires X contiguous along K and W contiguous along K or N")
        # make_block_ptr offsets are signed i32; the backing addresses and strides are not.
        if max(x.shape[0], x.shape[2], weight.shape[2], self._grid[0]) > (1 << 31) - max(block_shape):
            raise ValueError("Macro tile indices exceed the signed 32-bit block pointer offset range")
        object.__setattr__(self, "_tile_major_weight", tile_major_weight)
        if tile_major_weight:
            packed = _pack_macro_weight(weight, block_shape[1])
            if torch_npu.get_npu_format(packed) not in (0, 2):
                raise ValueError("Macro weight panels require ND storage")
            x_arg, _, *rest = self._arguments
            object.__setattr__(self, "_arguments", (x_arg, packed, *rest))

    def bind_l0_pair(self):
        """H: wider L1 dot (two nominal L0 fragments), same buffers as G.

        Bind once, outside capture/timing. G and H overwrite the same output;
        serialize calls and consume/snapshot it before launching either again.
        Physical overlap is determined by the installed backend, not by Triton.
        """
        if not self._tile_major_weight or self._block_shape != L0_PAIR_CONTROL_TILE or self._num_stages != 2:
            raise ValueError("L0 pair requires the tile-major 128x128x128, stages=2 control")
        inner = self._arguments[0].shape[-1]
        if inner % L0_PAIR_PANEL_K:
            raise ValueError(f"L0 pair requires K divisible by {L0_PAIR_PANEL_K}")
        return lambda: self._run_macro(L0_PAIR_PANEL_K)

    def __call__(self):
        return self._run_macro(self._block_shape[2])

    def _run_macro(self, bk):
        if not self._empty:
            bm, bn, _ = self._block_shape
            _transpose_quant_batch_matmul_v2_kernel_macro[self._grid](
                *self._arguments, BM=bm, BN=bn, BK=bk, TILE_MAJOR_W=self._tile_major_weight, num_stages=self._num_stages
            )
        return self.output


@dataclass(frozen=True, init=False, eq=False)
class PreparedTQBMMV2:
    _binding: PreparedTQBMM
    _config: V2Config
    _cores: int

    def __init__(self, x, weight, sx, sw, *, config=None, output_dtype=torch.bfloat16, out=None):
        config = config if config is not None else V2Config()
        binding = PreparedTQBMM(
            x,
            weight,
            sx,
            sw,
            block_shape=config.block_shape,
            num_stages=config.num_stages,
            output_dtype=output_dtype,
            out=out,
        )
        object.__setattr__(self, "_binding", binding)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_cores", _v2_programs(x.device, config.programs) if x.shape[0] else 1)

    @property
    def output(self):
        return self._binding.output

    def __call__(self):
        _launch_v2(self._binding._arguments, self._config, self._cores)
        return self.output


@dataclass(frozen=True, init=False, eq=False)
class PreparedDynamicWoAV2:
    _static: PreparedDynamicWoA
    _config: V2Config
    _cores: int

    def __init__(self, weight, sw, *, config=None):
        config = config if config is not None else V2Config()
        # Reuse V1's static-weight validation, FP32 scale normalization and
        # detached lifetime contract; its kernel is never launched here.
        static = PreparedDynamicWoA(weight, sw, block_shape=config.block_shape, num_stages=config.num_stages)
        object.__setattr__(self, "_static", static)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_cores", _v2_programs(weight.device, config.programs))

    def __call__(self, activations):
        groups, inner, columns = self._static._dimensions
        if activations.ndim != 3 or activations.shape[1:] != (groups, inner):
            raise ValueError("V2 activation must match [T,G,K]")
        if (
            activations.dtype not in (torch.bfloat16, torch.float16)
            or activations.device != self._static._weight.device
        ):
            raise ValueError("V2 activation must be BF16/FP16 on the weight device")
        tokens = activations.shape[0]
        output = torch.empty((tokens, groups, columns), dtype=activations.dtype, device=activations.device)
        if not tokens:
            return output
        q, sx = torch_npu.npu_dynamic_quant(activations.contiguous().view(tokens * groups, inner), dst_type=torch.int8)
        if (
            q.dtype != torch.int8
            or q.shape != (tokens * groups, inner)
            or not q.is_contiguous()
            or sx.dtype != torch.float32
            or sx.numel() != tokens * groups
            or not sx.is_contiguous()
            or q.device != activations.device
            or sx.device != activations.device
            or any(torch_npu.get_npu_format(v) not in (0, 2) for v in (q, sx))
        ):
            raise ValueError("V2 requires contiguous ND INT8/FP32 dynamic quant outputs")
        arguments = (
            q,
            self._static._weight,
            sx,
            self._static._scale,
            output,
            tokens,
            groups,
            inner,
            columns,
            groups * inner,
            inner,
            1,
            *self._static._weight_strides,
            groups,
            1,
            *self._static._scale_strides,
        )
        _launch_v2(arguments, self._config, self._cores)
        return output


def transpose_quant_batch_matmul_v2(x, weight, sx, sw, *, config=None, output_dtype=torch.bfloat16, out=None):
    """Safe standalone V2 entry; fixed-output lifetime follows PreparedTQBMM."""
    return PreparedTQBMMV2(x, weight, sx, sw, config=config, output_dtype=output_dtype, out=out)()
