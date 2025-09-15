from fastapi import APIRouter
from lerobot.find_cameras import find_all_realsense_cameras

from schemas import CalibrationConfig, Camera, RobotPortInfo
from utils.calibration import get_calibrations
from utils.camera import find_all_opencv_cameras
from utils.robot import find_robots, identify_robot_visually

router = APIRouter()


@router.get("/cameras")
async def get_cameras() -> list[Camera]:
    """Get all cameras"""
    return [Camera(**config) for config in find_all_realsense_cameras() + find_all_opencv_cameras()]


@router.get("/robots")
async def get_robots() -> list[RobotPortInfo]:
    """Get all connected Robots"""
    return await find_robots()


@router.get("/calibrations")
async def get_lerobot_calibrations() -> list[CalibrationConfig]:
    """Get calibrations known to huggingface leRobot"""
    return get_calibrations()


@router.put("/identify")
async def identify_robot(robot: RobotPortInfo, joint: str | None = None) -> None:
    """Visually identify the robot by moving given joint on robot"""
    await identify_robot_visually(robot, joint)
