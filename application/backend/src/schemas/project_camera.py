from abc import ABC
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from schemas.base import BaseIDModel, UTCDatetime

SupportedCameraDriver = Literal[
    "usb_camera",
    "ipcam",
    "basler",
    "realsense",
    "genicam",
    "simulation",
]


class BaseCamera(BaseIDModel, ABC):
    driver: SupportedCameraDriver

    created_at: UTCDatetime | None = Field(None)
    updated_at: UTCDatetime | None = Field(None)

    name: str = Field(..., description="Human-readable camera name")
    fingerprint: dict[str, Any] | None = Field(
        ...,
        description="Backend-specific camera identity from hardware discovery, or null when reselection is required",
    )
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

    url: str = Field(..., description="RTSP or HTTP stream URL")
    width: int | None = Field(None, ge=160, le=4096, description="Frame width in pixels (if supported)")
    height: int | None = Field(None, ge=120, le=2160, description="Frame height in pixels (if supported)")
    fps: int = Field(25, ge=1, le=60, description="Expected frame rate")


class SimulationCameraPayload(BaseModel):
    """A camera of a simulated robot's simulation.

    Its stream URL, size and frame rate are not stored: the simulation reports them while it runs, on
    an address that changes with every start.
    """

    robot_id: UUID = Field(..., description="The simulated robot whose simulation renders the camera")
    camera: str = Field(..., min_length=1, description="The camera's name in the simulation")


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
                "fingerprint": {"serial": "12345678AB"},
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
                "fingerprint": {"url": "rtsp://192.168.1.100:554/stream1"},
                "hardware_name": None,
                "payload": {
                    "url": "rtsp://192.168.1.100:554/stream1",
                },
            }
        }
    )


class SimulationCamera(BaseCamera):
    """A camera of a simulated robot's simulation; it streams while the simulation runs."""

    driver: Literal["simulation"] = "simulation"  # type: ignore[assignment]
    payload: SimulationCameraPayload

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Sim arm wrist",
                "driver": "simulation",
                "fingerprint": {"robot_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "camera": "wrist"},
                "hardware_name": "wrist",
                "payload": {"robot_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "camera": "wrist"},
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
                "fingerprint": {"serial": "40123456"},
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
                "fingerprint": {"serial": "123456789"},
                "hardware_name": "Intel RealSense D435",
                "payload": {
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
                "fingerprint": {"serial": "GC123456"},
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
    USBCamera | IPCamera | BaslerCamera | RealsenseCamera | GenicamCamera | SimulationCamera,
    Field(discriminator="driver"),
]

CameraAdapter: TypeAdapter[Camera] = TypeAdapter(Camera)

StreamedCamera = USBCamera | IPCamera | BaslerCamera | RealsenseCamera | GenicamCamera
"""A camera Studio streams from itself; a simulation camera streams from its simulation."""


def streamed_camera(camera: Camera) -> StreamedCamera:
    """Return *camera*, which must not be a simulation camera: resolve one to its stream first."""
    if isinstance(camera, SimulationCamera):
        raise ValueError(
            f"Camera {camera.name!r} streams from its simulation; load it through an environment to use it."
        )
    return camera
