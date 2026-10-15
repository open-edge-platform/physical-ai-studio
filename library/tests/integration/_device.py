# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hardware-aware device selection for gated LeRobot integration tests."""

from __future__ import annotations

import os

from physicalai.engine import resolve_accelerator

_DEVICE_ENV = "PHYSICALAI_LEROBOT_DEVICE"
_ALLOWED_DEVICES = {"auto", "cpu", "cuda", "xpu"}


def select_lerobot_test_device(requested: str | None = None, *, torch_module: object | None = None) -> str:
    """Select a test device and reject an unusable GPU instead of falling back.

    Returns:
        LeRobot's native device string: ``cpu``, ``cuda``, or ``xpu``.

    Raises:
        RuntimeError: If hardware-aware selection chooses a GPU unavailable to host PyTorch.
        ValueError: If the requested device is unsupported.
    """
    value = (requested if requested is not None else os.environ.get(_DEVICE_ENV, "auto")).strip().lower()
    if value not in _ALLOWED_DEVICES:
        msg = f"{_DEVICE_ENV} must be one of {sorted(_ALLOWED_DEVICES)}, got {value!r}."
        raise ValueError(msg)

    accelerator = resolve_accelerator(value)
    device = "cuda" if accelerator == "cu128" else accelerator
    if device == "cpu":
        return device

    if torch_module is None:
        import torch  # noqa: PLC0415  # Gated GPU checks need the host availability probe only at run time.

        torch_module = torch
    backend = getattr(torch_module, device, None)
    is_available = getattr(backend, "is_available", None)
    try:
        available = bool(backend is not None and callable(is_available) and is_available())
    except Exception:
        available = False
    if available:
        return device

    origin = f"Auto-selection selected {device}" if value == "auto" else f"The explicitly requested {device} device"
    msg = (
        f"{origin} cannot be used by the host torch runtime (torch.{device}.is_available() is false). "
        "Install the matching host torch extra and verify the GPU driver/runtime, or set "
        f"{_DEVICE_ENV}=cpu only for an intentional CPU test."
    )
    raise RuntimeError(msg)
