# SPDX-License-Identifier: Apache-2.0
"""Correctness-gated, staged large-T search; --dry-run needs no torch/NPU."""

from tools.dsv4_a3_wo.tune_tqbmm_large_t import main

if __name__ == "__main__":
    raise SystemExit(main())
