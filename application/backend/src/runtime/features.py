from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from physicalai.robot.interface import RobotObservation


def feature_names(joint_names: list[str]) -> list[str]:
    """Return the position feature names used by Studio robot data."""
    return [f"{name}.pos" for name in joint_names]


def _named_positions(joint_names: list[str], values: Iterable[float]) -> dict[str, float]:
    return dict(zip(feature_names(joint_names), (float(value) for value in values), strict=True))


def observation_to_dict(
    joint_names: list[str],
    observation: RobotObservation,
) -> dict[str, float]:
    """Map joint positions from an observation to Studio's named features."""
    return _named_positions(joint_names, observation.joint_positions)


def action_to_dict(joint_names: list[str], action: np.ndarray) -> dict[str, float]:
    """Map a sent action vector onto the same ``<joint>.pos`` keys as observations."""
    return _named_positions(joint_names, action)


def sanitize_camera_name(name: str) -> str:
    """Return the stable, filesystem-safe camera feature key."""
    return re.sub(r"[^a-z0-9 _-]+", "_", name.lower())
