import io
import json
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from exceptions import InvalidArchiveError
from schemas.dataset import Dataset
from schemas.job import TrainJob
from services.model_import_service import ModelImportService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_id():
    return uuid4()


@pytest.fixture
def dataset_id():
    return uuid4()


@pytest.fixture
def settings(tmp_path):
    return SimpleNamespace(
        models_dir=tmp_path / "models",
        data_import_max_uncompressed_bytes=10 * 1024 * 1024,
        data_import_min_free_bytes=0,
    )


@pytest.fixture
def dataset(dataset_id, project_id, tmp_path):
    return Dataset.model_validate(
        {
            "id": str(dataset_id),
            "name": "dataset",
            "path": str(tmp_path / "dataset"),
            "default_task": "task",
            "project_id": str(project_id),
            "environment_id": str(uuid4()),
        }
    )


@pytest.fixture
def job(project_id, dataset_id):
    return TrainJob.model_validate(
        {
            "id": str(uuid4()),
            "project_id": str(project_id),
            "type": "training",
            "status": "completed",
            "payload": {
                "project_id": str(project_id),
                "dataset_id": str(dataset_id),
                "policy": "act",
                "model_name": "imported",
            },
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(policy: str = "act", artifact: str = "act.pt") -> dict:
    return {
        "policy": {"name": policy, "source": {"class_path": f"physicalai.policies.{policy}.policy"}},
        "model": {"artifacts": {"torch": artifact}},
        "format": "policy_package",
        "version": "1.0",
    }


def _base_files(policy: str = "act", artifact: str = "act.pt") -> dict[str, str]:
    """Minimal valid model files."""
    return {
        "version_0/hparams.yaml": "policy: act\n",
        "version_0/metrics.csv": "step,loss\n1,0.1\n",
        "exports/torch/manifest.json": json.dumps(_make_manifest(policy, artifact)),
        f"exports/torch/{artifact}": "weights",
    }


def _make_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _create_model_directory(base_path: Path, files: dict[str, str]) -> Path:
    model_dir = base_path / "source_model"
    for name, content in files.items():
        file_path = model_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    return model_dir


@contextmanager
def _mock_services(settings, dataset, job=None):
    """Context manager for common service mocks."""
    with ExitStack() as stack:
        # get_settings is sync, not async
        stack.enter_context(patch("services.model_import_service.get_settings", return_value=settings))
        stack.enter_context(
            patch("services.model_import_service.DatasetService.get_dataset_by_id", AsyncMock(return_value=dataset))
        )
        stack.enter_context(
            patch(
                "services.model_import_service.asyncio.to_thread",
                AsyncMock(side_effect=lambda fn, *a, **k: fn(*a, **k)),
            )
        )
        if job is not None:
            stack.enter_context(
                patch("services.model_import_service.JobService.create_job", AsyncMock(return_value=job))
            )
            stack.enter_context(
                patch("services.model_import_service.ModelService.create_model", AsyncMock(side_effect=lambda m: m))
            )
        yield


# ---------------------------------------------------------------------------
# Directory import tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_import_model_directory_success(tmp_path, project_id, dataset_id, settings, dataset, job):
    source_dir = _create_model_directory(tmp_path, _base_files())

    with _mock_services(settings, dataset, job):
        model = await ModelImportService().import_model_directory(
            source_dir=source_dir,
            project_id=project_id,
            dataset_id=dataset_id,
            model_name="imported",
        )

    assert model.project_id == project_id
    assert model.policy == "act"
    assert Path(model.path).is_dir()
    assert source_dir.exists()  # Default is copy, not move


@pytest.mark.anyio
async def test_import_model_directory_move_removes_source(tmp_path, project_id, dataset_id, settings, dataset, job):
    source_dir = _create_model_directory(tmp_path, _base_files())

    with _mock_services(settings, dataset, job):
        model = await ModelImportService().import_model_directory(
            source_dir=source_dir,
            project_id=project_id,
            dataset_id=dataset_id,
            model_name="imported",
            move=True,
        )

    assert Path(model.path).is_dir()
    assert not source_dir.exists()


@pytest.mark.anyio
async def test_import_model_directory_rejects_nonexistent(tmp_path, project_id, dataset_id):
    with pytest.raises(InvalidArchiveError, match="does not exist"):
        await ModelImportService().import_model_directory(
            source_dir=tmp_path / "nonexistent",
            project_id=project_id,
            dataset_id=dataset_id,
            model_name="imported",
        )


@pytest.mark.anyio
async def test_import_model_directory_cleans_up_on_failure(tmp_path, project_id, dataset_id, settings, dataset):
    source_dir = _create_model_directory(tmp_path, _base_files())

    with (
        patch("services.model_import_service.get_settings", return_value=settings),
        patch("services.model_import_service.DatasetService.get_dataset_by_id", AsyncMock(return_value=dataset)),
        patch(
            "services.model_import_service.JobService.create_job",
            AsyncMock(side_effect=RuntimeError("fail")),
        ),
        patch(
            "services.model_import_service.asyncio.to_thread", AsyncMock(side_effect=lambda fn, *a, **k: fn(*a, **k))
        ),
        pytest.raises(RuntimeError, match="fail"),
    ):
        await ModelImportService().import_model_directory(
            source_dir=source_dir,
            project_id=project_id,
            dataset_id=dataset_id,
            model_name="imported",
        )

    assert not settings.models_dir.exists() or not any(settings.models_dir.iterdir())
