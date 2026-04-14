# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot control interfaces.

Public API::

    from physicalai.robot import Robot, connect, verify_robot
    from physicalai.robot import SO101  # requires: pip install physicalai[so101]
"""

from __future__ import annotations

from physicalai.robot.connect import connect
from physicalai.robot.interface import Robot
from physicalai.robot.verify import verify_robot

__all__ = [  # noqa: F822, RUF022
    "Robot",
    "SO101",
    "connect",
    "verify_robot",
]


def __getattr__(name: str) -> object:
    """Lazy-load concrete robot implementations.

    This avoids pulling in hardware SDKs (e.g. ``feetech-servo-sdk``)
    at package import time.

    Args:
        name: The attribute name being looked up.

    Returns:
        The requested class (e.g. ``SO101``).

    Raises:
        AttributeError: If ``name`` does not match a known lazy-loaded symbol.
    """
    if name == "SO101":
        from physicalai.robot.so101 import SO101  # noqa: PLC0415

        return SO101

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
