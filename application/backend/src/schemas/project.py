from datetime import datetime

from pydantic import Field

from schemas import Dataset
from schemas.base import BaseIDNameModel


class Project(BaseIDNameModel):
    updated_at: datetime | None = Field(None)
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
            }
        }
    }
