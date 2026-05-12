from pathlib import Path

from internal_datasets.dataset_client import DatasetClient
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from schemas import Dataset

# path -> (dataset_instance, mtime_of_meta_info_json_at_load_time)
_cache: dict[Path, tuple[InternalLeRobotDataset, float | None]] = {}


def _info_mtime(path: Path) -> float | None:
    info = path / "meta/info.json"
    return info.stat().st_mtime if info.is_file() else None


def get_internal_dataset(dataset: Dataset) -> DatasetClient:
    """Load dataset from dataset data class, using a mtime-keyed cache."""
    path = Path(dataset.path)
    current_mtime = _info_mtime(path)

    if path in _cache:
        cached_ds, cached_mtime = _cache[path]
        if current_mtime == cached_mtime:
            return cached_ds

    ds = InternalLeRobotDataset(path)
    _cache[path] = (ds, current_mtime)
    return ds


def invalidate_dataset_cache(path: Path) -> None:
    """Evict a path from the dataset cache (e.g. after a mutation commits)."""
    _cache.pop(path, None)
