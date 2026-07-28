import time
from pathlib import Path
from shutil import copytree

from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceAlreadyExistsError
from repositories.snapshot_repo import SnapshotRepository
from schemas import Dataset, Snapshot


class SnapshotService:
    """Allow for snapshotting of dataset to specific folder."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = SnapshotRepository(session)

    async def create_snapshot_for_dataset(self, dataset: Dataset, destination: Path) -> Snapshot:
        if destination.exists():
            raise ResourceAlreadyExistsError("Snapshot", f"Destination directory already exists:{destination}")

        snapshot = Snapshot(
            dataset_id=dataset.id,
            path=str(destination),
        )
        self._copy_dataset(Path(dataset.path), destination)
        await self.repo.save(snapshot)

        return snapshot

    @staticmethod
    def _copy_dataset(source: Path, destination: Path) -> None:
        copytree(source, destination)

    @staticmethod
    def generate_snapshot_folder_name() -> str:
        return time.strftime("snapshot_%Y-%m-%d_%H-%M-%S")
