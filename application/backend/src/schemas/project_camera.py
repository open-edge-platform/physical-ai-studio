from abc import ABC
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from schemas.base import BaseIDModel

# Supported camera drivers (subset of FrameSourceFactory.MediaSource)
SupportedCameraDriver = Literal[
    "usb_camera",
    "ipcam",
    "basler",
    "realsense",
    "genicam",
]


class BaseCamera(BaseIDModel, ABC):
    driver: SupportedCameraDriver

    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    name: str = Field(..., description="Human-readable camera name")
    fingerprint: str = Field(..., description="Camera fingerprint/source identifier for FrameSourceFactory")
    hardware_name: str | None = Field(..., description="Camera hardware name from discovery")


# ============================================================================
# Payload Models (Configuration Only)
# ============================================================================


class USBCameraPayload(BaseModel):
    """Configuration for WebcamCaptureNokhwa."""

    width: int = Field(..., ge=160, le=4096, description="Frame width in pixels")
    height: int = Field(..., ge=120, le=2160, description="Frame height in pixels")
    fps: int = Field(..., ge=1, le=120, description="Frames per second")
    exposure: int | None = Field(None, ge=-13, le=-1, description="Manual exposure value (-13 to -1)")
    gain: int | None = Field(None, ge=0, le=255, description="Camera gain (0-255)")


# ============================================================================
# Camera Models (Metadata + Payload)
# ============================================================================


class USBCamera(BaseCamera):
    """USB Camera using WebcamCaptureNokhwa (omni_camera backend)."""

    driver: Literal["usb_camera"] = "usb_camera"  # type: ignore[assignment]
    payload: USBCameraPayload

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "front_camera",
                "driver": "usb_camera",
                "fingerprint": "USB\\VID_1234&PID_5678:0",
                "hardware_name": "Logitech C920 HD Pro Webcam",
                "payload": {
                    "width": 1920,
                    "height": 1080,
                    "fps": 30,
                },
            }
        }
    )


# Discriminated union of all camera types
Camera = Annotated[
    USBCamera,
    Field(discriminator="driver"),
]

CameraAdapter: TypeAdapter[Camera] = TypeAdapter(Camera)
