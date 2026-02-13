# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for device management."""


def get_torch_device() -> str:
    """Get the appropriate torch device string based on availability.
    Checks for XPU, CUDA, MPS, and falls back to CPU if none are available.
    """
    import torch

    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"

    return "cpu"


def get_lightning_strategy() -> str:
    """Get the appropriate lightning strategy string based on available hardware.

    XPU device requires a specific strategy, while others are covered by 'auto' strategy."""
    import torch

    if torch.xpu.is_available():
        return "xpu_single"

    return "auto"
