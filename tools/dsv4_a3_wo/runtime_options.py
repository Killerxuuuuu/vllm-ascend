# SPDX-License-Identifier: Apache-2.0
"""Decode torch_npu runtime metadata without initializing any device."""


def internal_format_enabled(value):
    # bool(b"disable") is True: do not use Python truthiness for this option.
    if value in (True, b"enable", "enable"):
        return True
    if value in (False, b"disable", "disable"):
        return False
    raise ValueError(f"Unrecognized ALLOW_INTERNAL_FORMAT option: {value!r}")
