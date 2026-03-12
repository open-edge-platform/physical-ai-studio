from pydantic import BaseModel

from .dataset import Dataset
from .environment import EnvironmentWithRelations
from .model import Model


class TeleoperationConfig(BaseModel):
    task: str
    dataset: Dataset
    environment: EnvironmentWithRelations
    streaming_encoding: bool = True
    vcodec: str = "auto"
    encoder_threads: int | None = None
    encoder_queue_maxsize: int = 60


class InferenceConfig(BaseModel):
    model: Model
    task_index: int
    environment: EnvironmentWithRelations
    backend: str
