from typing import Any

from pydantic import BaseModel, Field, field_validator

from schemas.base import BaseIDModel


class CameraConfig(BaseIDModel):
    fingerprint: str = Field("", description="Camera port or realsense id")
    name: str = Field(min_length=1, max_length=50, description="Camera name")
    driver: str = Field(description="Driver used for Camera access")
    width: int = Field(640, description="Frame width")
    height: int = Field(480, description="Frame height")
    fps: int = Field(30, description="Camera fps")
    use_depth: bool = Field(False, description="Use Depth from RealSense")

    model_config = {
        "json_schema_extra": {
            "example": {
                "port_or_id": "/dev/video0",
                "name": "WebCam",
                "driver": "webcam",
                "width": 640,
                "height": 480,
                "fps": 30,
                "use_depth": False,
            }
        }
    }


class CameraProfile(BaseModel):
    width: int
    height: int
    fps: int

    @field_validator("fps", mode="before")
    def round_fps(cls, v: Any) -> int:
        return round(float(v))


class Camera(BaseModel):
    name: str = Field(description="Camera name")
    fingerprint: str = Field(description="Either serial id for  RealSense or port for OpenCV")
    driver: str = Field(description="Driver used for Camera access")
    default_stream_profile: CameraProfile

    @field_validator("fingerprint", mode="before")
    def cast_id_to_str(cls, v: Any) -> str:
        return str(v)


class SupportedCameraFormat(BaseModel):
    width: int = Field(..., description="Frame width")
    height: int = Field(..., description="Frame height")
    fps: list[int] = Field(..., description="FPS supported by resolution")

    model_config = {
        "json_schema_extra": {
            "example": {
                "width": 640,
                "height": 480,
                "fps": [5, 10, 30],
            }
        }
    }
