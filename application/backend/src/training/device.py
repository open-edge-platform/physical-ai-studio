# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning accelerator/strategy resolution for training runs.

One preference order for every runner. Callers that used to reimplement this
(the studio backend and the standalone trainer service each had their own
version, which disagreed) resolve through here instead so a job trains on the
same device whether it runs in-process or on a remote trainer.

Distinct from :mod:`physicalai.devices`, which resolves ``torch.device``
objects for inference and tensor movement: this module answers the narrower
question of what to pass to :class:`physicalai.train.Trainer`.
"""

from __future__ import annotations

ACCELERATOR_PREFERENCE: tuple[str, ...] = ("xpu", "cuda", "mps", "cpu")
"""Auto-detection order. XPU first: this is the primary supported accelerator."""


def resolve_accelerator(device_type: str | None = None) -> str:
    """Return the Lightning accelerator string for a training run.

    Args:
        device_type: Explicit accelerator (e.g. ``"xpu"``, ``"cuda"``, ``"cpu"``).
            When None the best available accelerator is auto-detected following
            :data:`ACCELERATOR_PREFERENCE`.

    Returns:
        The accelerator name to pass to the trainer.

    Example:
        >>> resolve_accelerator("cpu")
        'cpu'
    """
    if device_type is not None:
        return device_type

    import torch

    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


def resolve_strategy(device_type: str | None = None) -> str:
    """Return the Lightning strategy string for a training run.

    XPU needs its own single-device strategy; everything else is covered by
    ``"auto"``.

    Args:
        device_type: Explicit accelerator, or None to auto-detect.

    Returns:
        The strategy name to pass to the trainer.

    Example:
        >>> resolve_strategy("cpu")
        'auto'
    """
    return "xpu_single" if resolve_accelerator(device_type) == "xpu" else "auto"


def resolve_devices(device_index: int | None = None) -> list[int] | int:
    """Return the Lightning ``devices`` value for a training run.

    Args:
        device_index: Zero-based index of the accelerator to train on, or None
            to let Lightning pick one device.

    Returns:
        ``[device_index]`` when an index is given, otherwise ``1``.

    Example:
        >>> resolve_devices(2)
        [2]
        >>> resolve_devices()
        1
    """
    return [device_index] if device_index is not None else 1
