from pydantic import BaseModel, Field

from .dataset import Dataset
from .environment import EnvironmentWithRelations
from .model import Model


class StreamingEncodingSettings(BaseModel):
    streaming_encoding: bool = True
    vcodec: str = "auto"
    encoder_threads: int | None = None
    encoder_queue_maxsize: int = 60


class TeleoperationConfig(BaseModel):
    task: str
    dataset: Dataset
    environment: EnvironmentWithRelations
    streaming_encoding_settings: StreamingEncodingSettings = Field(default_factory=StreamingEncodingSettings)


class InferenceConfig(BaseModel):
    model: Model
    task_index: int
    environment: EnvironmentWithRelations
    backend: str
