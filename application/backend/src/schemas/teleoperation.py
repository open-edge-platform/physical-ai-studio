from pydantic import BaseModel

from .dataset import Dataset
from .model import Model
from .environment import EnvironmentWithRelations


class TeleoperationConfig(BaseModel):
    task: str
    dataset: Dataset
    environment: EnvironmentWithRelations


class InferenceConfig(BaseModel):
    model: Model
    task_index: int
    environment: EnvironmentWithRelations
    backend: str
