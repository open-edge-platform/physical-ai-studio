from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from physicalai.robot.interface import RobotObservation


def feature_names(joint_names: list[str], *, include_velocities: bool) -> list[str]:
    """Return position feature names followed by optional velocity names."""
    positions = [f"{name}.pos" for name in joint_names]
    if not include_velocities:
        return positions
    return positions + [f"{name}.vel" for name in joint_names]


def _named_positions(joint_names: list[str], values: Iterable[float]) -> dict[str, float]:
    names = feature_names(joint_names, include_velocities=False)
    return dict(zip(names, (float(value) for value in values), strict=True))


def observation_to_dict(
    joint_names: list[str],
    observation: RobotObservation,
    *,
    include_velocities: bool,
) -> dict[str, float]:
    """Map a robot observation to Studio's named scalar features."""
    values = [float(value) for value in observation.joint_positions]
    if include_velocities:
        sensor_data = observation.sensor_data
        if sensor_data is None or "velocities" not in sensor_data:
            raise RuntimeError("Robot observation is missing velocity data")
        values.extend(float(value) for value in sensor_data["velocities"])
        return dict(zip(feature_names(joint_names, include_velocities=True), values, strict=True))
    return _named_positions(joint_names, values)


def action_to_dict(joint_names: list[str], action: np.ndarray) -> dict[str, float]:
    """Map a sent action vector onto the same ``<joint>.pos`` keys as observations."""
    return _named_positions(joint_names, action)


def sanitize_camera_name(name: str) -> str:
    """Return the stable, filesystem-safe camera feature key."""
    return re.sub(r"[^a-z0-9 _-]+", "_", name.lower())
