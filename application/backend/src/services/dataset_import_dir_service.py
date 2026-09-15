"""Import a LeRobot v3 dataset directly from a local folder (no ZIP round-trip).

Intended for CLI/operator use, mirroring `services/model_import_service.py`.
"""

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID, uuid4

from exceptions import InvalidArchiveError
from schemas import Dataset

_REQUIRED_MARKERS = ("meta/info.json", "meta/tasks.parquet")


def _has_v3_data_files(source_dir: Path) -> bool:
    data_dir = source_dir / "data"
    if not data_dir.is_dir():
        return False
    return any(data_dir.rglob("file-*.parquet"))


def _validate_lerobot_v3_directory(source_dir: Path) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise InvalidArchiveError(f"Dataset directory does not exist: {source_dir}")

    missing = [marker for marker in _REQUIRED_MARKERS if not (source_dir / marker).is_file()]
    if not _has_v3_data_files(source_dir):
        missing.append("data/chunk-*/file-*.parquet")

    if missing:
        raise InvalidArchiveError(f"Directory does not look like a LeRobot v3 dataset; missing: {', '.join(missing)}")

    if (source_dir / "meta/tasks.jsonl").is_file() or (source_dir / "meta/episodes.jsonl").is_file():
        raise InvalidArchiveError(
            "Directory looks like a LeRobot v2 dataset (found tasks.jsonl/episodes.jsonl); "
            "only v3 is supported by this command"
        )


class DatasetImportDirService:
    def __init__(self, persist_dataset: Callable[[Dataset], Awaitable[Dataset]]) -> None:
        self._persist_dataset = persist_dataset

    async def import_dataset_directory(
        self,
        *,
        source_dir: Path,
        project_id: UUID,
        environment_id: UUID,
        dataset_name: str,
        default_task: str = "",
        move: bool = False,
    ) -> Dataset:
        """Validate and register a local LeRobot v3 dataset directory (copy or move)."""
        _validate_lerobot_v3_directory(source_dir)

        from settings import get_settings

        settings = get_settings()
        settings.datasets_dir.mkdir(parents=True, exist_ok=True)

        dataset_id = uuid4()
        destination_dir = settings.datasets_dir / str(dataset_id)

        try:
            if move:
                await asyncio.to_thread(shutil.move, str(source_dir), str(destination_dir))
            else:
                await asyncio.to_thread(shutil.copytree, source_dir, destination_dir)

            dataset = Dataset(
                id=dataset_id,
                name=dataset_name,
                default_task=default_task,
                project_id=project_id,
                environment_id=environment_id,
            )
            return await self._persist_dataset(dataset)
        except Exception:
            shutil.rmtree(destination_dir, ignore_errors=True)
            raise
