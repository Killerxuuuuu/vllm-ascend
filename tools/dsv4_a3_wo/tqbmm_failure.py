# SPDX-License-Identifier: Apache-2.0
"""Post-failure diagnostics; never import this before the first comparison.

The CPU oracle uses INT64 products/sums and FP32 scale order, then the caller's
CPU output cast. It checks selected mismatches, not the entire matrix product.
Snapshots contain local model weights: do not upload failure.pt publicly.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np

MAX_ERROR_TILES = 16
MAX_GOLDEN_POINTS = 16
DEFAULT_BLOCK_SHAPE = (64, 32, 32)


def finite_value(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def summarize_failure(*, x, weight, sx, sw, actual, native, block_shape, reuse_m, output_cast, reuse_n=1):
    """Arrays are CPU NumPy snapshots; output_cast must also run on CPU."""
    tokens, groups, columns = actual.shape
    block_m, block_n, _ = block_shape
    flat_bad = np.flatnonzero((actual != native).reshape(-1))
    coords = np.column_stack(np.unravel_index(flat_bad, actual.shape))
    tile_keys = np.column_stack((coords[:, 1], coords[:, 0] // block_m, coords[:, 2] // block_n))
    tiles, counts = np.unique(tile_keys, axis=0, return_counts=True)
    tile_order = np.argsort(-counts, kind="stable")[:MAX_ERROR_TILES]

    # Work on mismatches only, not a full FP64 copy of both large outputs.
    with np.errstate(invalid="ignore", over="ignore"):
        errors = np.abs(actual.reshape(-1)[flat_bad].astype(np.float64) - native.reshape(-1)[flat_bad])
    errors = np.nan_to_num(errors, nan=np.inf, posinf=np.inf)
    half = MAX_GOLDEN_POINTS // 2
    worst = flat_bad[np.argsort(-errors, kind="stable")[:half]]
    selected = list(dict.fromkeys([*map(int, worst), *map(int, flat_bad[:half])]))
    points = []
    for index in selected:
        t, remainder = divmod(index, groups * columns)
        g, n = divmod(remainder, columns)
        integer = int(np.sum(x[t, g].astype(np.int64) * weight[g, :, n].astype(np.int64), dtype=np.int64))
        with np.errstate(invalid="ignore", over="ignore"):
            scaled = np.float32(np.float32(np.float32(integer) * sw[g, n]) * sx[t, g])
        golden = float(output_cast(float(scaled)))
        observed, reference = float(actual[t, g, n]), float(native[t, g, n])
        points.append(
            {
                "position": [t, g, n],
                "M_pair": t // (block_m * reuse_m),
                "accumulator": (t // block_m) % reuse_m,
                "N_pair": n // (block_n * reuse_n),
                "N_accumulator": (n // block_n) % reuse_n,
                "cpu_integer_sum": integer,
                "cpu_scaled_fp32": finite_value(scaled),
                "cpu_output": finite_value(golden),
                "triton": finite_value(observed),
                "native": finite_value(reference),
                "triton_matches_cpu": observed == golden,
                "native_matches_cpu": reference == golden,
            }
        )
    return {
        "shape_T_G_K_N": [tokens, groups, x.shape[-1], columns],
        "mismatched_elements": int(flat_bad.size),
        "max_abs": finite_value(errors.max()) if errors.size else 0.0,
        "top_error_tiles": [{"group_Mtile_Ntile": tiles[i].tolist(), "errors": int(counts[i])} for i in tile_order],
        "cpu_golden_points": points,
        "sample_triton_matches_cpu": sum(p["triton_matches_cpu"] for p in points),
        "sample_native_matches_cpu": sum(p["native_matches_cpu"] for p in points),
        "note": "CPU oracle covers sampled mismatches only; snapshots are taken after the failed comparison.",
    }


def dump_failure(*, args, x, weight, sx, sw, actual, native, torch_module):
    """No new NPU kernels or warmup; preserve an already-failed invocation."""
    root = Path(args.failure_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="prequantized-", dir=root))
    metadata = {
        "schema_version": 1,
        "check": "same_int8_inputs",
        "torch": str(torch_module.__version__),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "weight_stride": list(weight.stride()),
        "output_dtype": str(actual.dtype),
    }
    cpu = {
        name: tensor.detach().cpu()
        for name, tensor in {"x": x, "weight": weight, "sx": sx, "sw": sw, "actual": actual, "native": native}.items()
    }
    torch_module.save({"metadata": metadata, **cpu}, directory / "failure.pt")
    print(f"Failure tensor snapshot saved: {directory / 'failure.pt'}", flush=True)
    shape = args.block_shape
    if shape is None:
        tokens = x.shape[0]
        shape = (16 if tokens <= 16 else 32 if tokens <= 32 else 64, *DEFAULT_BLOCK_SHAPE[1:])
    report = summarize_failure(
        x=cpu["x"].numpy(),
        weight=cpu["weight"].numpy(),
        sx=cpu["sx"].float().numpy(),
        sw=cpu["sw"].float().numpy(),
        actual=cpu["actual"].float().numpy(),
        native=cpu["native"].float().numpy(),
        block_shape=tuple(shape),
        reuse_m=args.reuse_m,
        reuse_n=getattr(args, "reuse_n", 1),
        output_cast=lambda value: (
            torch_module.tensor(value, dtype=torch_module.float32, device="cpu").to(actual.dtype).item()
        ),
    )
    (directory / "failure.json").write_text(
        json.dumps({**metadata, **report}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return directory
