from datetime import datetime

from pydantic import Field

from schemas import Dataset
from schemas.base import BaseIDModel, BaseIDNameModel

from .camera import CameraConfig


class ProjectConfig(BaseIDModel):
    fps: int = Field(30, description="Recording FPS for datasets")
    cameras: list[CameraConfig] = Field([], description="Project cameras")
    # robots: list[RobotConfig]

    model_config = {
        "json_schema_extra": {
            "example": {
                "fps": "30",
                "cameras": [
                    {
                        "port_or_id": "/dev/video0",
                        "name": "WebCam",
                        "type": "OpenCV",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                        "use_depth": False,
                    }
                ],
            }
        }
    }


class Project(BaseIDNameModel):
    updated_at: datetime | None = Field(None)
    config: ProjectConfig | None = Field(None, description="Project config")
    datasets: list[Dataset] = Field([], description="Datasets")
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "name": "SO101 Teleoperation",
                "updated_at": "2021-06-29T16:24:30.928000+00:00",
                "datasets": [
                    {
                        "id": "fec4a691-76ee-4f66-8dea-aad3110e16d6",
                        "name": "Collect blocks",
                        "path": "/some/path/to/dataset",
                    }
                ],
                "config": {
                    "fps": "30",
                    "cameras": [
                        {
                            "port_or_id": "/dev/video0",
                            "name": "WebCam",
                            "type": "OpenCV",
                            "width": 640,
                            "height": 480,
                            "fps": 30,
                            "use_depth": False,
                        }
                    ],
                },
            }
        }
    }
