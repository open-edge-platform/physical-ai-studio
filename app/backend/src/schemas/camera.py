from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CameraConfig(BaseModel):
    id: str = Field(min_length=1, max_length=50, description="Camera port or realsense id")
    name: str = Field(min_length=1, max_length=50, description="Camera name")
    type: Literal["RealSense", "OpenCV"]
    width: int = Field(640, description="Frame width")
    height: int = Field(480, description="Frame height")
    fps: int = Field(30, description="Camera fps")
    use_depth: bool = Field(False, description="Use Depth from RealSense")


class CameraProfile(BaseModel):
    width: int
    height: int
    fps: int

    @field_validator("fps", mode="before")
    def round_fps(cls, v: any) -> int:
        return round(float(v))


class Camera(BaseModel):
    name: str = Field(description="Camera name")
    id: str = Field(description="Either serial id for  RealSense or port for OpenCV")
    type: Literal["RealSense", "OpenCV"]
    default_stream_profile: CameraProfile

    @field_validator("id", mode="before")
    def cast_id_to_str(cls, v: any) -> str:
        return str(v)
