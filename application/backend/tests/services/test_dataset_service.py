from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from schemas.dataset import Dataset
from services.dataset_service import DatasetService


def _make_dataset() -> Dataset:
    return Dataset.model_validate(
        {
            "id": uuid4(),
            "name": "Dataset 1",
            "default_task": "Task",
            "project_id": uuid4(),
            "environment_id": uuid4(),
        }
    )


def _service_with_dataset(dataset: Dataset) -> DatasetService:
    session = MagicMock(spec=AsyncSession)
    with patch("services.dataset_service.DatasetRepository"):
        service = DatasetService(session)
    service.repo = AsyncMock()
    service.repo.get_by_id.return_value = dataset
    service.repo.delete_by_id.return_value = None
    return service


async def test_delete_dataset_remove_files_missing_dir_does_not_raise(tmp_path) -> None:
    dataset = _make_dataset()
    service = _service_with_dataset(dataset)

    with patch("schemas.dataset.get_settings") as mock_settings:
        mock_settings.return_value.datasets_dir = tmp_path
        assert not (tmp_path / str(dataset.id)).exists()

        await service.delete_dataset(dataset_id=dataset.id, remove_files=True)

    service.repo.delete_by_id.assert_awaited_once_with(dataset.id)


async def test_delete_dataset_remove_files_deletes_existing_dir(tmp_path) -> None:
    dataset = _make_dataset()
    service = _service_with_dataset(dataset)

    with patch("schemas.dataset.get_settings") as mock_settings:
        mock_settings.return_value.datasets_dir = tmp_path
        dataset_dir = tmp_path / str(dataset.id)

        dataset_dir.mkdir(parents=True)
        (dataset_dir / "episode_0.parquet").write_text("data")

        await service.delete_dataset(dataset_id=dataset.id, remove_files=True)

        assert not dataset_dir.exists()
