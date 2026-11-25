# SPDX-License-Identifier: Apache-2.0
"""One isolated macro launch with the production probe shape, for msprof op simulator."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("G", "H"))
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    # Parse --help without initializing an NPU. Under msprof, match only the
    # macro kernel, not the tensor setup ops; there is no warmup or graph here.
    import torch
    import torch_npu

    from benchmark_tqbmm_v2 import configure_internal_format
    from vllm_ascend.ops.triton.transpose_quant_batch_matmul_v2 import (
        L0_PAIR_CONTROL_TILE,
        L0_PAIR_PANEL_K,
        PreparedMacroTQBMMV2,
    )

    torch.npu.set_device(args.device)
    configure_internal_format(torch_npu)
    t, g, k, n = 8192, 1, 4096, 1024
    with torch.inference_mode():
        q = torch.ones((t, g, k), dtype=torch.int8, device=args.device)
        w = torch.ones((g, k, n), dtype=torch.int8, device=args.device)
        sx = torch.ones((t, g), dtype=torch.float32, device=args.device)
        sw = torch.ones((g, n), dtype=torch.float32, device=args.device)
        plan = PreparedMacroTQBMMV2(
            q, w, sx, sw, block_shape=L0_PAIR_CONTROL_TILE, num_stages=2, tile_major_weight=True
        )
        run = plan if args.case == "G" else plan.bind_l0_pair()
        dot_k = L0_PAIR_CONTROL_TILE[2] if args.case == "G" else L0_PAIR_PANEL_K
        print(f"case={args.case} shape={(t, g, k, n)} dot_k={dot_k}", flush=True)
        torch.npu.synchronize()
        run()
        torch.npu.synchronize()


if __name__ == "__main__":
    main()
