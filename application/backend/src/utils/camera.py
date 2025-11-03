from pathlib import Path

from lerobot.cameras import Camera as LeRobotCamera
from lerobot.cameras import CameraConfig as LeRobotCameraConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

from schemas import CameraConfig

VIDEO4LINUX_PATH = "/sys/class/video4linux"


def build_camera_config(camera_config: CameraConfig) -> LeRobotCameraConfig:
    """Build either realsense or opencv camera config from CameraConfig BaseModel"""
    if camera_config.driver == "realsense":
        return RealSenseCameraConfig(
            serial_number_or_name=camera_config.port_or_device_id,
            fps=camera_config.fps,
            width=camera_config.width,
            height=camera_config.height,
            use_depth=camera_config.use_depth,
        )
    if camera_config.driver == "webcam":
        path = camera_config.port_or_device_id.split(":")[0]
        return OpenCVCameraConfig(
            index_or_path=Path(path),
            width=camera_config.width,
            height=camera_config.height,
            fps=camera_config.fps,
        )
    raise ValueError(f"Unknown CameraConfig driver: {camera_config.driver}")


def initialize_camera(cfg: LeRobotCameraConfig) -> LeRobotCamera:
    """Initialize a LeRobot Camera object from LeRobot CameraConfig"""
    if cfg.type == "opencv":
        return OpenCVCamera(cfg)
    if cfg.type == "intelrealsense":
        return RealSenseCamera(cfg)
    raise ValueError(f"The motor type '{cfg.type}' is not valid.")
