# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Device utility functions."""

from __future__ import annotations

import torch

XPU_AVAILABLE = None


def is_xpu_available() -> bool:
    """Check if Intel XPU (GPU) device is available.

    This function checks whether Intel XPU support is available in the current
    PyTorch installation and if an XPU device is accessible. The result is cached
    in a global variable to avoid repeated checks.

    Returns:
        bool: True if XPU device is available, False otherwise.

    Note:
        The function uses a global cache variable XPU_AVAILABLE to store the
        result of the first check, improving performance on subsequent calls.
    """
    global XPU_AVAILABLE  # noqa: PLW0603
    if XPU_AVAILABLE is None:
        XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()
    return XPU_AVAILABLE
