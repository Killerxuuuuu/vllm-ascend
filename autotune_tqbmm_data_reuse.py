# SPDX-License-Identifier: Apache-2.0
"""Correctness-gated A/B reuse + wider-N experiment; no production dispatch changes."""

import sys

from tools.dsv4_a3_wo.tune_tqbmm_large_t import main as run_tuning


def main(argv=None):
    # Matched ablations: old/wider N, one/two M tiles, one/two N tiles.
    # Keep N>=128,K>=128 excluded. A larger logical region is represented by
    # separate dot accumulators, not one oversized dot or a full-weight copy.
    candidates = [
        argument
        for shape in ((128, 64, 128), (128, 128, 64), (64, 128, 64), (64, 64, 128))
        for stage in (1, 2)
        for argument in ("--candidate", *map(str, shape), str(stage))
    ]
    return run_tuning(
        [
            "--control",
            "128",
            "64",
            "128",
            "1",
            "--reuse-m",
            "1",
            "2",
            "--reuse-n",
            "1",
            "2",
            "--screen-tokens",
            "1024",
            "8192",
            "--allow-internal-format",
            "on",
            *candidates,
            *(sys.argv[1:] if argv is None else argv),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
