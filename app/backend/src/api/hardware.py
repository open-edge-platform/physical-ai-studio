from fastapi import APIRouter
from uuid import uuid4
from schemas import Camera, RobotPortInfo, CalibrationConfig
from utils.camera import find_all_opencv_cameras, find_all_realsense_cameras
from utils.robot import find_robots, identify_robot_visually
from utils.calibration import get_calibrations

router = APIRouter()


@router.get("/cameras")
async def get_cameras() -> list[Camera]:
    """Get all cameras"""
    return [Camera(**config) for config in find_all_opencv_cameras()] + [Camera(**config) for config in find_all_realsense_cameras()]

@router.get("/robots")
async def get_robots() -> list[RobotPortInfo]:
    """Get all connected Robots"""
    return await find_robots();

@router.get("/calibrations")
async def get_lerobot_calibrations() -> list[CalibrationConfig]:
    """Get calibrations known to huggingface leRobot"""
    return get_calibrations()

@router.put("/identify")
async def identify_robot(robot: RobotPortInfo, joint: str | None = None):
    await identify_robot_visually(robot, joint)
