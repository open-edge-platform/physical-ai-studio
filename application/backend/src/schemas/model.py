from typing import Annotated
from uuid import UUID

from schemas.base import BaseIDModel, Field


class Model(BaseIDModel):
    name: str
    path: str
    policy: str
    properties: dict
    project_id: Annotated[UUID, Field(description="Unique identifier")]
    dataset_id: Annotated[UUID, Field(description="Unique identifier")]

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
            }
        }
    }
