from typing import Annotated

from fastapi import APIRouter, Query
from loguru import logger
from physicalai.capture import DeviceInfo, discover_all

from schemas import CalibrationConfig, Camera, CameraProfile, Robot, SerialPortInfo
from schemas.robot import RobotType
from utils.calibration import get_calibrations
from utils.camera_factory import DRIVER_KEY_MAP
from utils.serial_robot_tools import find_robots, identify_so101_robot_visually
from utils.trossen_robot_tools import identify_trossen_robot_visually

router = APIRouter(prefix="/api/hardware", tags=["Hardware"])


def _fingerprint_from_device_info(info: DeviceInfo) -> str:
    return info.hardware_id if (info.id_stable and info.hardware_id) else info.device_id


def _build_camera_list(discovered: dict[str, list[DeviceInfo]]) -> list[Camera]:
    """Convert discovered devices to Camera response models."""
    res: list[Camera] = []
    sp = CameraProfile(width=640, height=480, fps=30)  # TODO: Implement proper default camera profile retrieval

    for driver, devices in discovered.items():
        backend_driver = DRIVER_KEY_MAP.get(driver)
        if backend_driver is None:
            continue
        for info in devices:
            res.append(
                Camera(
                    name=info.name,
                    fingerprint=_fingerprint_from_device_info(info),
                    driver=backend_driver,
                    default_stream_profile=sp,
                ),
            )
    return res


@router.get("/cameras")
async def get_cameras(
    all: Annotated[bool, Query(description="Include cameras in use by other processes")] = False,
) -> list[Camera]:
    """Get all cameras.

    When `all=true`, cameras currently in use by another process are also included.
    """
    discovered = discover_all(only_usable=not all)
    logger.debug("Discovered cameras: {}", discovered)
    return _build_camera_list(discovered)


@router.get("/serial_devices")
async def get_robots() -> list[SerialPortInfo]:
    """Get all connected Robots"""
    return await find_robots()


@router.get("/calibrations")
async def get_lerobot_calibrations() -> list[CalibrationConfig]:
    """Get calibrations known to huggingface leRobot"""
    return get_calibrations()


@router.post("/identify")
async def identify_robot(robot: Robot, joint: str | None = None) -> None:
    """Visually identify the robot by moving given joint on robot"""
    if robot.type in {RobotType.SO101_LEADER, RobotType.SO101_FOLLOWER}:
        await identify_so101_robot_visually(robot, joint)

    if robot.type in {RobotType.TROSSEN_WIDOWXAI_LEADER, RobotType.TROSSEN_WIDOWXAI_FOLLOWER}:
        await identify_trossen_robot_visually(robot)
