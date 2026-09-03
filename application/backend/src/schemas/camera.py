from typing import Any

from pydantic import BaseModel, Field, field_validator


class CameraProfile(BaseModel):
    width: int
    height: int
    fps: int

    @field_validator("fps", mode="before")
    def round_fps(cls, v: Any) -> int:
        return round(float(v))


class Camera(BaseModel):
    name: str = Field(description="Camera name")
    fingerprint: dict[str, Any] = Field(description="Backend-specific camera identity from hardware discovery")
    driver: str = Field(description="Driver used for Camera access")
    default_stream_profile: CameraProfile

    @field_validator("fingerprint", mode="before")
    def require_object(cls, v: Any) -> dict[str, Any]:
        if not isinstance(v, dict) or not v:
            raise ValueError("Camera fingerprint must be a non-empty JSON object")
        return v


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
