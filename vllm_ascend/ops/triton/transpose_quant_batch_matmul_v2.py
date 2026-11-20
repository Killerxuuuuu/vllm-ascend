# SPDX-License-Identifier: Apache-2.0
"""A3 structural experiment: persistent tiles, paired K panels, one acc.

Uses standard Triton, not A5-only UB->L1 copy APIs. Cache panels are source
SSA values; physical L1/L0 placement and asynchronous overlap must be verified
in generated IR/profiling. No production dispatch or V1 defaults are changed.
"""

from dataclasses import dataclass

import torch
import torch_npu
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.transpose_quant_batch_matmul import PreparedDynamicWoA, PreparedTQBMM

MAX_LOGICAL_ACCUMULATOR_BYTES = 64 * 1024
MAX_PANEL_K = 512
GROUP_SIZE_M = 4


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
def _v2_load_panel(
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
    PANEL_K: tl.constexpr,
):
    # Reconstruct every GM address from immutable bases and absolute K.
    k = start_k + tl.arange(0, PANEL_K).to(tl.int64)
    x_address = x_ptr + rows[:, None] * SXT + group * SXG + k[None, :] * SXK
    w_address = w_ptr + group * SWG + k[:, None] * SWK + cols[None, :] * SWN
    a = tl.load(x_address, mask=(rows[:, None] < T) & (k[None, :] < K), other=0)
    b = tl.load(w_address, mask=(k[:, None] < K) & (cols[None, :] < N), other=0)
    return a, b


@triton.jit
def _v2_split_panel_halves(a_panel, b_panel, BM: tl.constexpr, BN: tl.constexpr, HALF_K: tl.constexpr):
    # Move the two *contiguous K halves* to the final, size-two axis. split
    # lowers to static extract_slice operations, not indexed gather/scalar
    # tensor extraction. These are SSA shape/layout ops; physical movement
    # remains a compiler decision and must be checked in the generated IR.
    a_halves = tl.permute(tl.reshape(a_panel, (BM, 2, HALF_K)), (0, 2, 1))
    b_halves = tl.permute(tl.reshape(b_panel, (2, HALF_K, BN)), (1, 2, 0))
    a_low, a_high = tl.split(a_halves)
    b_low, b_high = tl.split(b_halves)
    return a_low, b_low, a_high, b_high


@triton.jit
def _v2_panel_subtile(
    a_panel,
    b_panel,
    BM: tl.constexpr,
    BN: tl.constexpr,
    DOT_K: tl.constexpr,
    PANEL_K: tl.constexpr,
    MICRO: tl.constexpr,
):
    # Only 1/2/4 dot tiles per panel are accepted by validate_v2_options.
    # Preserve increasing K order without any runtime indices. In particular,
    # B's former axis-0 gather became 64x32 scalar loops on A3, with an
    # inserted full-index-table GM store *inside* each scalar iteration.
    if PANEL_K == DOT_K:
        return a_panel, b_panel
    else:
        a_low, b_low, a_high, b_high = _v2_split_panel_halves(a_panel, b_panel, BM, BN, PANEL_K // 2)
        if MICRO < PANEL_K // DOT_K // 2:
            a_half, b_half = a_low, b_low
        else:
            a_half, b_half = a_high, b_high
        if PANEL_K == 2 * DOT_K:
            return a_half, b_half
        else:
            a_first, b_first, a_second, b_second = _v2_split_panel_halves(a_half, b_half, BM, BN, DOT_K)
            if MICRO % 2 == 0:
                return a_first, b_first
            else:
                return a_second, b_second


@triton.jit
def _v2_dot_panel(
    a_panel,
    b_panel,
    acc,
    BM: tl.constexpr,
    BN: tl.constexpr,
    DOT_K: tl.constexpr,
    PANEL_K: tl.constexpr,
    ACTIVE_K: tl.constexpr,
):
    # Keep the original K order and INT32 accumulation/FP32 scaling contract.
    # Do not emit dot operations whose entire K subtile is padding. ACTIVE_K
    # is compile-time: the tail is specialized, not a runtime branch on Cube.
    # tl.cdiv is a JIT tensor operation, even for literal inputs. Keep this
    # arithmetic in constexpr space so static_range never receives int32[].
    micro_tiles: tl.constexpr = (ACTIVE_K + DOT_K - 1) // DOT_K
    for micro in tl.static_range(micro_tiles):
        a, b = _v2_panel_subtile(a_panel, b_panel, BM, BN, DOT_K, PANEL_K, micro)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
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
    full_panels: tl.constexpr = K // PANEL_K
    tail_k: tl.constexpr = K % PANEL_K
    panels_per_step: tl.constexpr = 2 if PREFETCH else 1
    for tile in range(pid, num_tiles, PROGRAMS):
        rows, cols = _v2_coordinates(tile, T, N, BM, BN)
        # One output accumulator per tile, reset before any dot for that tile.
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        # Panel tensors must NOT be loop-carried: A3 layout inference converts
        # dot operands to 4D NZ, while masked lookahead can remain 2D ND. Passing
        # one as the other's scf.for iter_arg triggers a cbuf type mismatch.
        # Instead load/consume pairs within each iteration; only acc is carried.
        if full_panels >= panels_per_step:
            for step in range(full_panels // panels_per_step):
                start_k = step * panels_per_step * PANEL_K
                a_current, b_current = _v2_load_panel(
                    x_ptr,
                    w_ptr,
                    rows,
                    cols,
                    group,
                    start_k,
                    T,
                    K,
                    N,
                    SXT,
                    SXG,
                    SXK,
                    SWG,
                    SWK,
                    SWN,
                    PANEL_K,
                )
                if PREFETCH:
                    # Lexical lookahead: load the second panel before the
                    # first panel's dot, without crossing a loop boundary.
                    a_next, b_next = _v2_load_panel(
                        x_ptr,
                        w_ptr,
                        rows,
                        cols,
                        group,
                        start_k + PANEL_K,
                        T,
                        K,
                        N,
                        SXT,
                        SXG,
                        SXK,
                        SWG,
                        SWK,
                        SWN,
                        PANEL_K,
                    )
                acc = _v2_dot_panel(a_current, b_current, acc, BM, BN, DOT_K, PANEL_K, PANEL_K)
                if PREFETCH:
                    acc = _v2_dot_panel(a_next, b_next, acc, BM, BN, DOT_K, PANEL_K, PANEL_K)
        if PREFETCH and full_panels % 2:
            # The last unpaired FULL panel uses the same dot extent as pairs.
            a_odd, b_odd = _v2_load_panel(
                x_ptr,
                w_ptr,
                rows,
                cols,
                group,
                (full_panels - 1) * PANEL_K,
                T,
                K,
                N,
                SXT,
                SXG,
                SXK,
                SWG,
                SWK,
                SWN,
                PANEL_K,
            )
            acc = _v2_dot_panel(a_odd, b_odd, acc, BM, BN, DOT_K, PANEL_K, PANEL_K)
        if tail_k:
            # A partial panel is separate for BOTH prefetch modes. Preserve the
            # requested panel width, but only compute ceil(remaining K/DOT_K)
            # subtiles; never pass an entirely padding K tile to dot.
            a_tail, b_tail = _v2_load_panel(
                x_ptr,
                w_ptr,
                rows,
                cols,
                group,
                full_panels * PANEL_K,
                T,
                K,
                N,
                SXT,
                SXG,
                SXK,
                SWG,
                SWK,
                SWN,
                PANEL_K,
            )
            acc = _v2_dot_panel(a_tail, b_tail, acc, BM, BN, DOT_K, PANEL_K, tail_k)
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
