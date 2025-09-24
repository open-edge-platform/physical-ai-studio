
from pydantic import Field
from datetime import datetime
from schemas.base import BaseIDNameModel, BaseIDModel

from .camera import CameraConfig
from .robot import RobotConfig


class ProjectConfig(BaseIDModel):
    fps: int = Field(30, description="Recording FPS for datasets")
    #datasets: list[str] = Field([], description="Datasets available for this project")
    #cameras: list[CameraConfig]
    #robots: list[RobotConfig]

    model_config = {
        "json_schema_extra": {
            "example": {
                "fps": "30",
            }
        }
    }


class Project(BaseIDNameModel):
    updated_at: datetime | None = Field(None)
    config: ProjectConfig | None = Field(None, description="Project config")
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "name": "SO101 Teleoperation",
                "updated_at": "2021-06-29T16:24:30.928000+00:00",
                # "config": { "fps": "30" } #optional
            }
        }
    }

