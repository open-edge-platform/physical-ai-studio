# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot lifecycle utilities.

Provides the ``connect()`` context manager for safe robot connection management.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicalai.robot.interface import Robot


@contextmanager
def connect(robot: Robot) -> Generator[Robot, None, None]:
    """Context manager for safe robot lifecycle.

    Calls ``robot.connect()`` on entry and ``robot.disconnect()`` on exit,
    including when exceptions occur. Analogous to the built-in ``open()``
    for files.

    Args:
        robot: Any object satisfying the :class:`Robot` protocol.

    Yields:
        The connected robot instance.

    Example:
        >>> with connect(robot) as r:
        ...     obs = r.get_observation()
        ...     r.send_action(action)
    """
    robot.connect()
    try:
        yield robot
    finally:
        robot.disconnect()
