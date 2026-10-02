from functools import lru_cache
from pathlib import Path

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.dataset_client import DatasetClient
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from schemas import Dataset


def get_internal_dataset(dataset: Dataset, mode: DatasetAccessMode = DatasetAccessMode.READ_ONLY) -> DatasetClient:
    """Load dataset from dataset data class."""
    return InternalLeRobotDataset(Path(dataset.path), access_mode=mode)


@lru_cache(maxsize=2)
def _load_read_dataset(path: str, _metadata_version: tuple[int, int] | None) -> DatasetClient:
    return InternalLeRobotDataset(Path(path), access_mode=DatasetAccessMode.READ_ONLY)


def get_internal_read_dataset(dataset: Dataset) -> DatasetClient:
    """Reuse read-only datasets until their metadata changes."""
    path = Path(dataset.path)
    info = path / "meta/info.json"
    stat = info.stat() if info.is_file() else None
    version = (stat.st_mtime_ns, stat.st_size) if stat else None
    return _load_read_dataset(str(path), version)


def get_internal_recording_dataset(dataset: Dataset) -> DatasetClient:
    """Load a dataset in recording-mutation mode for worker flows."""
    return get_internal_dataset(dataset, mode=DatasetAccessMode.RECORDING_MUTATION)
