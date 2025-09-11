from pydantic import BaseModel, Field
from typing import Literal

class CameraConfig(BaseModel):
    id: str = Field(None, min_length=1, max_length=50, description="Camera port or realsense id")
    name: str = Field(None, min_length=1, max_length=50, description="Camera name")
    type: Literal["RealSense", "OpenCV"]
    width: int = Field(640, description="Frame width")
    height: int = Field(480, description="Frame height")
    fps: int = Field(30, description="Camera fps")
    use_depth: bool = Field(False, description="Use Depth from RealSense")
