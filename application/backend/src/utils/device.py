# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for device management."""


def get_torch_device(device: str | None = None) -> str:
    """Get the torch device string to use for training.

    When *device* is provided it is returned as-is, allowing the caller to
    override auto-detection.  When ``None``, the best available accelerator
    is chosen automatically (XPU > CUDA > CPU).
    """
    if device is not None:
        return device

    import torch

    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"

    return "cpu"


def get_lightning_strategy(device: str | None = None) -> str:
    """Get the Lightning strategy string for the given device.

    XPU requires a specific strategy; all other devices are covered by
    ``'auto'``.  When *device* is ``None`` the decision is based on
    hardware auto-detection.
    """
    if device is not None:
        if device == "xpu":
            import physicalai.devices.xpu

            return "xpu_single"
    return "auto"


_DEFAULT_PRECISION_BY_DEVICE: dict[str, str] = {
    "cuda": "bf16-mixed",
    "xpu": "32",
}


def get_default_precision(device: str) -> str | None:
    """Return the default training precision for a given device type.

    Returns a Lightning-compatible precision string, or None to let
    Lightning choose (which resolves to ``"32-true"``).
    """
    return _DEFAULT_PRECISION_BY_DEVICE.get(device)

    import torch

    if torch.xpu.is_available():
        import physicalai.devices.xpu  # noqa: F401 — registers XPU accelerator/strategy with Lightning

        return "xpu_single"

    return "auto"
