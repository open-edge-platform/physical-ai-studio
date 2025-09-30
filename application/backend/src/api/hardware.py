from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from lerobot.find_cameras import find_all_realsense_cameras

from schemas import CalibrationConfig, Camera, RobotPortInfo
from utils.calibration import get_calibrations
from utils.camera import find_all_opencv_cameras, gen_frames
from utils.robot import find_robots, identify_robot_visually

router = APIRouter(prefix="/api/hardware", tags=["Hardware"])


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


@router.post("/identify")
async def identify_robot(robot: RobotPortInfo, joint: str | None = None) -> None:
    """Visually identify the robot by moving given joint on robot"""
    await identify_robot_visually(robot, joint)


@router.get("/camera_feed")
async def get_camera_feed(id: str, type: Literal["RealSense", "OpenCV"]) -> StreamingResponse:
    """Get a streaming response from the camera"""
    return StreamingResponse(gen_frames(id, type), media_type="multipart/x-mixed-replace; boundary=frame")
