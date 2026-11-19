# SPDX-License-Identifier: Apache-2.0
"""Compare the measured large-T control with experimental paired-M B reuse.

No server/kernel defaults are changed. --dry-run works without torch/NPU.
"""

import sys

from tools.dsv4_a3_wo.tune_tqbmm_large_t import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--control",
                "128",
                "64",
                "128",
                "2",
                "--reuse-m",
                "1",
                "2",
                "--screen-tokens",
                "1024",
                "8192",
                *sys.argv[1:],
            ]
        )
    )
