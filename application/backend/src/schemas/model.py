from schemas.base import BaseIDModel

class Model(BaseIDModel):
    name: str
    path: str
    properties: dict

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "",
                "name": "Dataset X/Y ACT Model",
                "path": "Path/to/model/ckpt",
                "properties": {},
            }
        }
    }
