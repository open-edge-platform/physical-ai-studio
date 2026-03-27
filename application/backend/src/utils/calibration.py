import os
from pathlib import Path
from typing import Literal

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS, TELEOPERATORS

from schemas import CalibrationConfig


def get_calibrations() -> list[CalibrationConfig]:
    """Get all calibrations known to lerobot"""
    teleoperators_path = HF_LEROBOT_CALIBRATION / TELEOPERATORS
    robots_path = HF_LEROBOT_CALIBRATION / ROBOTS

    return [
        *get_calibration_of_folder(teleoperators_path, "leader"),
        *get_calibration_of_folder(robots_path, "follower"),
    ]


def get_calibration_of_folder(folder: Path, robot_type: Literal["leader", "follower"]) -> list[CalibrationConfig]:
    """Get all calibration configs available for either teleoperator or robot"""
    calibrations = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            calibrations.append(CalibrationConfig(path=full_path, id=Path(full_path).stem, robot_type=robot_type))

    return calibrations
