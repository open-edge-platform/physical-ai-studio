from pydantic import BaseModel, Field
from typing import List
from .camera import CameraConfig
from .robot import RobotConfig

#    {
#        "name": "Duplo",
#        "datasets": [
#            "rhecker/duplo"
#        ],
#        "fps": 30.0,
#        "cameras": {
#            "front": {
#                "id": "323522062395",
#                "type": "RealSense",
#                "width": 640,
#                "height": 480,
#                "use_depth": true
#            },
#            "gripper": {
#                "id": "/dev/video6",
#                "type": "OpenCV",
#                "width": 640,
#                "height": 480
#            }
#        },
#        "robots": {
#            "follower": {
#                "serial_id": "5AA9017083",
#                "id": "khaos"
#            },
#            "leader": {
#                "serial_id": "5A7A016060",
#                "id": "khronos"
#            }
#        }
#    }


class ProjectConfig(BaseModel):
    id: str = Field(..., description="UUID of the project")
    fps: int = Field(30, description="Recording FPS for datasets")
    name: str = Field(None, min_length=1, max_length=50, description="Project name")
    datasets: List[str] = Field([], description="Datasets available for this project")
    cameras: List[CameraConfig]
    robots: List[RobotConfig]
