from datetime import datetime
from typing import Annotated
from uuid import UUID

from schemas.base import BaseIDModel, Field


class Model(BaseIDModel):
    name: str
    path: str
    policy: str
    properties: dict
    project_id: Annotated[UUID, Field(description="Project Unique identifier")]
    dataset_id: Annotated[UUID, Field(description="Dataset Unique identifier")]
    snapshot_id: Annotated[UUID, Field(description="Snapshot Unique identifier")]
    created_at: datetime | None = Field(None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "",
                "name": "Dataset X/Y ACT Model",
                "path": "Path/to/model/ckpt",
                "properties": {},
                "policy": "act",
                "dataset_id": "",
                "project_id": "",
                "snapshot_id": "",
                "created_at": "2021-06-29T16:24:30.928000+00:00",
            }
        }
    }
