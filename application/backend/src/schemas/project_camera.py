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


class IPCameraPayload(BaseModel):
    """Configuration for IPCameraCapture."""

    stream_url: str = Field(..., description="RTSP or HTTP stream URL")
    username: str | None = Field(None, description="Camera login username")
    password: str | None = Field(None, description="Camera login password")
    width: int | None = Field(None, ge=160, le=4096, description="Frame width in pixels (if supported)")
    height: int | None = Field(None, ge=120, le=2160, description="Frame height in pixels (if supported)")
    fps: int = Field(25, ge=1, le=60, description="Expected frame rate")


class BaslerCameraPayload(BaseModel):
    """Configuration for BaslerCapture."""

    serial_number: str | None = Field(None, description="Camera serial number")
    exposure: float | None = Field(None, ge=1, le=1000000, description="Exposure time in microseconds")
    gain: float | None = Field(None, ge=0, le=40, description="Camera gain in decibels")
    width: int | None = Field(None, ge=1, le=10000, description="Frame width in pixels")
    height: int | None = Field(None, ge=1, le=10000, description="Frame height in pixels")
    fps: int = Field(30, ge=1, le=1000, description="Frames per second")
    is_mono: bool = Field(False, description="Camera outputs monochrome (grayscale) images")


class RealsenseCameraPayload(BaseModel):
    """Configuration for RealsenseCapture."""

    device_index: int = Field(0, ge=0, le=10, description="Camera device index")
    width: int = Field(640, ge=424, le=1920, description="Frame width in pixels")
    height: int = Field(480, ge=240, le=1080, description="Frame height in pixels")
    fps: int = Field(30, ge=6, le=90, description="Frames per second")
    depth_range_min: float = Field(0.3, ge=0.1, le=5.0, description="Minimum depth detection range in meters")
    depth_range_max: float = Field(3.0, ge=1.0, le=10.0, description="Maximum depth detection range in meters")
    output_type: Literal["color", "depth", "both"] = Field("color", description="Type of output frames")


class GenicamCameraPayload(BaseModel):
    """Configuration for GenicamCapture."""

    serial_number: str | None = Field(None, description="Camera serial number")
    cti_files: list[str] = Field(default_factory=list, description="GenTL producer files")
    exposure: float | None = Field(None, ge=1, le=1000000, description="Exposure time in microseconds")
    gain: float | None = Field(None, ge=0, le=40, description="Camera gain in decibels")
    width: int | None = Field(None, ge=1, le=10000, description="Frame width in pixels")
    height: int | None = Field(None, ge=1, le=10000, description="Frame height in pixels")
    fps: int = Field(30, ge=1, le=240, description="Frames per second")
    x: int | None = Field(None, ge=0, description="Horizontal offset in pixels")
    y: int | None = Field(None, ge=0, description="Vertical offset in pixels")


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


class IPCamera(BaseCamera):
    """IP Camera using IPCameraCapture (RTSP/HTTP streams)."""

    driver: Literal["ipcam"] = "ipcam"  # type: ignore[assignment]
    payload: IPCameraPayload

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "security_camera",
                "driver": "ipcam",
                "fingerprint": "rtsp://192.168.1.100:554/stream1",
                "hardware_name": None,
                "payload": {
                    "stream_url": "rtsp://192.168.1.100:554/stream1",
                    "username": "admin",
                    "password": "password123",
                },
            }
        }
    )


class BaslerCamera(BaseCamera):
    """Basler industrial camera using BaslerCapture (pypylon)."""

    driver: Literal["basler"] = "basler"  # type: ignore[assignment]
    payload: BaslerCameraPayload

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "inspection_camera",
                "driver": "basler",
                "fingerprint": "40123456",
                "hardware_name": "Basler acA1920-40gm",
                "payload": {
                    "serial_number": "40123456",
                    "exposure": 10000,
                    "gain": 0.0,
                },
            }
        }
    )


class RealsenseCamera(BaseCamera):
    """Intel RealSense depth camera using RealsenseCapture."""

    driver: Literal["realsense"] = "realsense"  # type: ignore[assignment]
    payload: RealsenseCameraPayload

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "depth_camera",
                "driver": "realsense",
                "fingerprint": "123456789",
                "hardware_name": "Intel RealSense D435",
                "payload": {
                    "device_index": 0,
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
            }
        }
    )


class GenicamCamera(BaseCamera):
    """GenICam compliant camera using GenicamCapture (Harvesters)."""

    driver: Literal["genicam"] = "genicam"  # type: ignore[assignment]
    payload: GenicamCameraPayload

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "genicam_camera",
                "driver": "genicam",
                "fingerprint": "GC123456",
                "hardware_name": "Genicam camera",
                "payload": {
                    "serial_number": "GC123456",
                    "exposure": 10000,
                },
            }
        }
    )


# Discriminated union of all camera types
Camera = Annotated[
    USBCamera | IPCamera | BaslerCamera | RealsenseCamera | GenicamCamera,
    Field(discriminator="driver"),
]

CameraAdapter: TypeAdapter[Camera] = TypeAdapter(Camera)
