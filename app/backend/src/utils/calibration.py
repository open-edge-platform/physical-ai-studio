import os
from pathlib import Path
from schemas import CalibrationConfig
from lerobot.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS, ROBOTS

from typing import Literal, List

def get_calibrations() -> List[CalibrationConfig]:
    teleoperators_path = HF_LEROBOT_CALIBRATION / TELEOPERATORS
    robots_path = HF_LEROBOT_CALIBRATION / ROBOTS

    return [
        *get_calibration_of_folder(teleoperators_path, "teleoperator"),
        *get_calibration_of_folder(robots_path, "robot")
    ]


def get_calibration_of_folder(folder: str, robot_type: Literal["teleoperator", "robot"]):
    calibrations = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            calibrations.append(CalibrationConfig(
                path=full_path,
                id=Path(full_path).stem,
                robot_type=robot_type
            ))

    return calibrations
