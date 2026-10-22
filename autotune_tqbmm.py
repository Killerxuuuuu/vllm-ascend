# SPDX-License-Identifier: Apache-2.0
"""Run a serial A3 TQBMM parameter search from the repository root.

Example:
    python autotune_tqbmm.py --quant-model /path/to/int8-model --tokens 1 16 64 512

No torch/NPU imports occur in this launcher. --dry-run is safe on a CPU host.
The measured winner is saved, never applied to kernel/model defaults.
"""

import sys

from tools.dsv4_a3_wo.tune_tqbmm import main

if __name__ == "__main__":
    raise SystemExit(main(["--search-space", "balanced", *sys.argv[1:]]))
