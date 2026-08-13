"""Helpers for wiring Studio robots onto the physicalai transport."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID


def shared_robot_name(robot_id: UUID) -> str:
    """Return the transport-safe ``SharedRobot`` name for a Studio robot.

    ``SharedRobot`` keys its Zenoh topics by name, so physicalai restricts the
    name to letters, digits, ``_`` and ``-``. Studio robot names are free-form
    user text ("Left Arm"), which the transport rejects, so builders must not
    pass the display name through.

    The robot's id is used instead because it:

    * always satisfies the transport's character rule,
    * is unique per robot, so two robots sharing a display name never collide
      on a single owner process, and
    * is stable across renames, so renaming a robot in the UI cannot orphan a
      running owner.

    Args:
        robot_id: Identifier of the Studio robot the owner is created for.

    Returns:
        A name accepted by ``SharedRobot``.
    """
    return str(robot_id)
