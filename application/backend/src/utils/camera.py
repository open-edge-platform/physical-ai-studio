import asyncio
import base64
import os
import re
import time
from collections.abc import Generator
from typing import Any

import cv2
from fastapi import WebSocket
from lerobot.cameras import Camera
from lerobot.cameras import CameraConfig as LeRobotCameraConfig
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.errors import DeviceNotConnectedError
from frame_source import FrameSourceFactory
from lerobot.find_cameras import find_all_opencv_cameras as le_robot_find_all_opencv_cameras

from schemas import CameraConfig

VIDEO4LINUX_PATH = "/sys/class/video4linux"


def get_realsense_dev_ports() -> list[str]:
    """
    Use video4linux to get the /dev/video* ports of realsense cameras
    This is useful because they are not to be used with OpenCV and cannot easily be filtered
    """
    realsense_ports = []
    try:
        for port in os.listdir(VIDEO4LINUX_PATH):
            full_port = os.path.join("/dev", port)
            name_path = os.path.join(VIDEO4LINUX_PATH, port, "name")

            with open(name_path) as name_file:
                if re.search("RealSense", name_file.read()):
                    realsense_ports.append(full_port)
    except FileNotFoundError:
        pass
    return realsense_ports


def add_device_name_to_opencv_camera(camera: dict[str, Any]) -> dict[str, Any]:
    """Uses video4linux to get a better name for the camera"""
    try:
        port = os.path.basename(camera["id"])
        name_path = os.path.join(VIDEO4LINUX_PATH, port, "name")

        with open(name_path) as name_file:
            name = name_file.read().strip("\n")
            camera["name"] = name
    except FileNotFoundError:
        pass
    except TypeError:
        # Camera id can be an int on macOS
        pass

    return camera


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    """Get all cameras that are not realsense and find a more user friendly name"""
    realsense_ports = get_realsense_dev_ports()
    cameras = [cam for cam in le_robot_find_all_opencv_cameras() if cam["id"] not in realsense_ports]
    return [add_device_name_to_opencv_camera(camera) for camera in cameras]


def gen_frames(id: str, driver: str) -> Generator[bytes, None, None]:
    """
    Continuously capture frames, encode them as JPEG,
    and yield them in the multipart format expected by browsers.
    """

    _id: str | int
    if id.isdigit():
        _id = int(id)
    else:
        _id = str(id)
    cam = FrameSourceFactory.create(driver, _id)
    cam.connect()

    while True:
        success, frame = cam.read()
        if not success:
            break

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        jpg_bytes = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n")


async def gen_camera_frames(websocket: WebSocket, stop_event: asyncio.Event, config: CameraConfig) -> None:
    """
    Continuously capture frames, encode them as JPEG,
    and pass that as string to the websocket
    """
    camera_config = build_camera_config(config)
    camera = initialize_camera(camera_config)
    attempts = 0
    while not camera.is_connected and attempts < 3:
        try:
            camera.connect()
            break
        except DeviceNotConnectedError:
            attempts += 1
            await asyncio.sleep(1)

    while not stop_event.is_set():
        start_loop_t = time.perf_counter()
        _, buffer = cv2.imencode(".jpg", camera.read())
        data = base64.b64encode(buffer).decode()
        await asyncio.create_task(websocket.send_text(data))
        dt_s = time.perf_counter() - start_loop_t
        await asyncio.sleep(1 / camera.fps - dt_s)


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
        return OpenCVCameraConfig(
            index_or_path=camera_config.port_or_device_id,
            width=camera_config.width,
            height=camera_config.height,
            fps=camera_config.fps,
        )
    raise ValueError(f"Unknown CameraConfig driver: {camera_config.driver}")


def initialize_camera(cfg: LeRobotCameraConfig) -> Camera:
    """Initialize a LeRobot Camera object from LeRobot CameraConfig"""
    if cfg.type == "opencv":
        return OpenCVCamera(cfg)
    if cfg.type == "intelrealsense":
        return RealSenseCamera(cfg)
    raise ValueError(f"The motor type '{cfg.type}' is not valid.")
