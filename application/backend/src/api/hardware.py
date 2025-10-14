from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from frame_source import FrameSourceFactory

from schemas import CalibrationConfig, Camera, CameraProfile, RobotPortInfo
from utils.calibration import get_calibrations
from utils.camera import gen_frames
from utils.robot import find_robots, identify_robot_visually

router = APIRouter(prefix="/api/hardware", tags=["Hardware"])


@router.get("/cameras")
async def get_cameras() -> list[Camera]:
    """Get all cameras"""
    cameras = FrameSourceFactory.discover_devices(sources=["webcam", "realsense", "genicam", "basler"])
    res = []
    sp = CameraProfile(width=640, height=480, fps=30)  # TODO: Implement proper default camera profile retrieval

    for driver, cams in cameras.items():
        for cam in cams:
            res.append(Camera(name=cam["name"], port_or_device_id=cam["id"], driver=driver, default_stream_profile=sp))

    return res


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
async def get_camera_feed(id: str, driver: str) -> StreamingResponse:
    """Get a streaming response from the camera"""
    return StreamingResponse(gen_frames(id, driver), media_type="multipart/x-mixed-replace; boundary=frame")
