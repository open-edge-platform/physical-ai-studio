import os
import re
from typing import Any

from lerobot.find_cameras import find_all_opencv_cameras as le_robot_find_all_opencv_cameras

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
