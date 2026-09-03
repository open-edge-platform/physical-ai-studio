from typing import Annotated

from fastapi import APIRouter, Query
from loguru import logger
from physicalai.capture import DeviceInfo, discover_all

from schemas import Camera, CameraProfile
from utils.camera_factory import DRIVER_KEY_MAP

router = APIRouter(prefix="/api/hardware", tags=["Hardware"])


def _fingerprint_from_device_info(info: DeviceInfo) -> dict[str, object] | None:
    return info.hardware_payload


def _build_camera_list(discovered: dict[str, list[DeviceInfo]]) -> list[Camera]:
    """Convert discovered devices to Camera response models."""
    res: list[Camera] = []
    sp = CameraProfile(width=640, height=480, fps=30)

    for driver, devices in discovered.items():
        backend_driver = DRIVER_KEY_MAP.get(driver)
        if backend_driver is None:
            continue
        for info in devices:
            fingerprint = _fingerprint_from_device_info(info)
            if not fingerprint:
                logger.warning("Skipping camera without a hardware fingerprint: {}", info.name)
                continue
            res.append(
                Camera(
                    name=info.name,
                    fingerprint=fingerprint,
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
