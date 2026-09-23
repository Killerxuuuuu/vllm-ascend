# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic IO control only. Not a top-k implementation or production path."""

from vllm.triton_utils import tl, triton


@triton.jit
def _topk_io_control_kernel(
    values_ptr,
    scratch_ptr,
    out_values_ptr,
    out_indices_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
):
    lanes = tl.arange(0, WIDTH)
    values = tl.load(values_ptr + lanes, mask=lanes < N, other=0.0)
    # Consume all input elements so the compiler cannot discard the reads.
    # This adds scratch traffic absent from production: NOT identical IO.
    tl.store(scratch_ptr + lanes, values, mask=lanes < N)
    output_lanes = tl.arange(0, OUT_WIDTH)
    selected = tl.load(
        values_ptr + output_lanes,
        mask=(output_lanes < N) & (output_lanes < K),
        other=-float("inf"),
    )
    indices = tl.where(output_lanes < N, output_lanes, -1).to(tl.int32)
    tl.store(out_values_ptr + output_lanes, selected, mask=output_lanes < K)
    tl.store(out_indices_ptr + output_lanes, indices, mask=output_lanes < K)
