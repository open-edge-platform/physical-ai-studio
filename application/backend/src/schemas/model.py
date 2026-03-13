from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import ConfigDict

from schemas.base import BaseIDModel, Field


class Model(BaseIDModel):
    name: str
    path: str
    policy: str
    properties: dict
    project_id: Annotated[UUID, Field(description="Project Unique identifier")]
    dataset_id: Annotated[UUID, Field(description="Dataset Unique identifier")]
    snapshot_id: Annotated[UUID, Field(description="Snapshot Unique identifier")]
    train_job_id: UUID | None = Field(None, description="ID of the training job for this model")
    parent_model_id: UUID | None = Field(None, description="Parent model this was retrained from")
    version: int = Field(1, description="Model version, incremented on each retrain")
    created_at: datetime | None = Field(None)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "",
                "name": "Dataset X/Y ACT Model",
                "path": "Path/to/model/ckpt",
                "properties": {},
                "policy": "act",
                "dataset_id": "",
                "project_id": "",
                "snapshot_id": "",
                "train_job_id": "0db0c16d-0d3c-4e0e-bc5a-ca710579e549",
                "parent_model_id": None,
                "version": 1,
                "created_at": "2021-06-29T16:24:30.928000+00:00",
            }
        }
    )
