from pathlib import Path

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.dataset_client import DatasetClient
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from schemas import Dataset


def get_internal_dataset(dataset: Dataset, mode: DatasetAccessMode = DatasetAccessMode.READ_ONLY) -> DatasetClient:
    """Load dataset from dataset data class."""
    return InternalLeRobotDataset(Path(dataset.path), access_mode=mode)


def get_internal_read_dataset(dataset: Dataset) -> DatasetClient:
    """Load a dataset in read-only mode for API endpoints."""
    return get_internal_dataset(dataset, mode=DatasetAccessMode.READ_ONLY)


def get_internal_recording_dataset(dataset: Dataset) -> DatasetClient:
    """Load a dataset in recording-mutation mode for worker flows."""
    return get_internal_dataset(dataset, mode=DatasetAccessMode.RECORDING_MUTATION)
