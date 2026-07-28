"""End-to-end integration test: upload a dataset, train ACT, export, and infer.

Drives the real FastAPI app, real services, and a real (isolated) SQLite
database - no dependency-injection stubs - through the same lifecycle a user
exercises through the UI:

    upload dataset -> import job -> train job (ACT) -> auto-export -> download
    -> load exported model -> infer

Background workers normally run as separate `multiprocessing.Process`es
(`core.scheduler.Scheduler`). Spawning real processes per test is slow and
hard to synchronize, so this test instantiates `TrainingWorker` /
`DatasetImportWorker` directly and drives their per-job methods in-process -
the same production code path, just called synchronously instead of polled
from an infinite loop. See `docs/development/e2e-integration-testing-plan.md`.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import zipfile
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_scheduler
from main import app
from schemas.base_job import JobStatus
from schemas.job import TrainJobPayload

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_ONNX_BACKEND = "onnx"


def _run_dataset_import_job_step() -> None:
    """Claim and process exactly one pending dataset-import job, in-process."""
    import asyncio

    from schemas.job import DatasetImportJob
    from services.dataset_import.service import DatasetImportService
    from workers.dataset_import_worker import DatasetImportWorker

    async def _drain() -> None:
        job = await DatasetImportService.claim_pending_dataset_import_job()
        assert isinstance(job, DatasetImportJob), "Expected a pending dataset import job"
        worker = DatasetImportWorker(stop_event=mp.Event(), event_queue=mp.Queue())
        await worker._process_job(job)

    asyncio.run(_drain())


def _run_training_job_step() -> None:
    """Claim and run exactly one pending training job to completion, in-process.

    Mirrors the body of `TrainingWorker.run_loop` for a single iteration
    instead of running the (otherwise infinite) polling loop.
    """
    import asyncio

    from schemas import Model
    from services import DatasetService
    from services.job_service import JobService
    from services.snapshot_service import SnapshotService
    from settings import get_settings
    from workers.training_worker import TrainingWorker

    async def _drain() -> None:
        job = await JobService.get_pending_train_job()
        assert job is not None, "Expected a pending training job"
        payload = TrainJobPayload.model_validate(job.payload)

        settings = get_settings()
        model_id = uuid4()
        model_dir = settings.models_dir / str(model_id)

        dataset = await DatasetService.get_dataset_by_id(payload.dataset_id)
        snapshot_dir = settings.snapshot_dir / SnapshotService.generate_snapshot_folder_name()
        snapshot = await SnapshotService.create_snapshot_for_dataset(dataset, destination=snapshot_dir)
        payload.snapshot_id = snapshot.id

        model = Model(
            id=model_id,
            project_id=payload.project_id,
            dataset_id=payload.dataset_id,
            path=str(model_dir),
            name=payload.model_name,
            snapshot_id=snapshot.id,
            policy=payload.policy,
            properties={},
            train_job_id=job.id,
            parent_model_id=payload.base_model_id,
            version=1,
            created_at=None,
        )

        worker = TrainingWorker(stop_event=mp.Event(), interrupt_event=mp.Event(), event_queue=mp.Queue())
        worker.interrupt_event.clear()
        await worker._train_model(job, model, snapshot, payload)

    asyncio.run(_drain())
    # `_train_model` persists the model under `model_id` on success; the caller
    # already knows that id (it's generated before training starts), so no
    # extra lookup is needed here.


def test_dataset_upload_train_export_infer_e2e(
    migrated_db: None,
    synthetic_dataset_archive_bytes: bytes,
) -> None:
    """Full happy path: upload -> import -> train ACT -> export -> download -> infer."""
    client = TestClient(app)

    # --- 1. Project + environment -------------------------------------------------
    project_id = uuid4()
    response = client.post("/api/projects", json={"id": str(project_id), "name": "E2E Project"})
    assert response.status_code == 201, response.text

    environment_id = uuid4()
    response = client.post(
        f"/api/projects/{project_id}/environments",
        json={"id": str(environment_id), "name": "E2E Environment", "robots": [], "cameras": []},
    )
    assert response.status_code == 201, response.text

    # --- 2. Dataset import: prepare -> upload -> detect -> finalize -> commit -----
    response = client.post(
        f"/api/projects/{project_id}/imports/datasets:prepare",
        data={"format_hint": "auto", "dataset_name": "E2E Dataset"},
    )
    assert response.status_code == 202, response.text
    import_job_id = response.json()["id"]

    response = client.put(
        f"/api/projects/{project_id}/imports/datasets/{import_job_id}:upload",
        files={"archive": ("dataset.zip", synthetic_dataset_archive_bytes, "application/zip")},
    )
    assert response.status_code == 202, response.text

    _run_dataset_import_job_step()  # detect + build draft manifest

    response = client.get(f"/api/jobs/{import_job_id}")
    assert response.status_code == 200, response.text
    import_job = response.json()
    assert import_job["payload"]["step"] == "awaiting_user_review", import_job

    response = client.post(
        f"/api/projects/{project_id}/imports/datasets/{import_job_id}:finalize",
        json={"environment_id": str(environment_id), "default_task": "do the thing"},
    )
    assert response.status_code == 202, response.text

    _run_dataset_import_job_step()  # pre-commit validation + commit

    response = client.get(f"/api/jobs/{import_job_id}")
    assert response.status_code == 200, response.text
    import_job = response.json()
    assert import_job["status"] == JobStatus.COMPLETED, import_job
    dataset_id = import_job["payload"]["result_dataset_id"]
    assert dataset_id is not None

    # --- 3. Submit an ACT training job ---------------------------------------------
    response = client.post(
        "/api/jobs:train",
        json={
            "project_id": str(project_id),
            "dataset_id": dataset_id,
            "policy": "act",
            "model_name": "E2E ACT Model",
            "max_steps": 100,  # TrainJobPayload.max_steps has a ge=100 floor
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.5,
            "device": {"type": "cpu"},
            "precision": "32-true",  # bf16-mixed (the default) is unreliable on CPU
        },
    )
    assert response.status_code == 200, response.text
    train_job = response.json()
    train_job_id = train_job["id"]

    _run_training_job_step()

    response = client.get(f"/api/jobs/{train_job_id}")
    assert response.status_code == 200, response.text
    train_job = response.json()
    assert train_job["status"] == JobStatus.COMPLETED, train_job
    assert train_job["progress"] == 100

    # The model id is deterministic from `_run_training_job_step`'s job draining;
    # look it up the same way the UI would: via the project's model list.
    response = client.get(f"/api/projects/{project_id}/models")
    assert response.status_code == 200, response.text
    models = response.json()
    assert len(models) == 1, models
    model = models[0]
    model_id = model["id"]
    assert model["policy"] == "act"
    assert set(model["available_backends"]) >= {_ONNX_BACKEND}

    # --- 4. Model detail + metrics --------------------------------------------------
    response = client.get(f"/api/models/{model_id}")
    assert response.status_code == 200, response.text
    detail = response.json()
    assert any(export["type"] == _ONNX_BACKEND for export in detail["exports"])
    assert detail["training_summary"]["max_steps"] == 100

    # --- 5. Download the onnx export and confirm it is a valid archive -------------
    response = client.get(f"/api/models/{model_id}/exports/{_ONNX_BACKEND}/download")
    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert "manifest.json" in archive.namelist()

    # --- 6. Load the exported model and run inference on a real observation -------
    _assert_exported_model_runs_inference(Path(model["path"]) / "exports" / _ONNX_BACKEND, dataset_id)

    # --- 7. Cleanup: deleting the model removes the DB row and files --------------
    response = client.delete(f"/api/models/{model_id}")
    assert response.status_code == 200, response.text
    assert not Path(model["path"]).exists()


def _assert_exported_model_runs_inference(export_dir: Path, dataset_id: str) -> None:
    """Load the exported ONNX model and run one `select_action` call.

    This is the "deploying for inference" checkpoint: Studio's job is to
    produce an artifact Runtime's `InferenceModel(...)` can load and run - see
    `AGENTS.md` Cross-Repo Rules.
    """
    from physicalai.data import LeRobotDataModule
    from physicalai.data.lerobot import FormatConverter
    from physicalai.inference import InferenceModel

    from settings import get_settings

    # Read from the dataset the model was trained on (the training worker
    # copies it into a snapshot before training, but the source dataset is
    # untouched and equivalent for this check).
    dataset_root = get_settings().datasets_dir / dataset_id
    datamodule = LeRobotDataModule(repo_id="snapshot", root=dataset_root, train_batch_size=1, num_workers=0)
    batch = next(iter(datamodule.train_dataloader()))
    observation = FormatConverter.to_observation(batch)[0:1].to_numpy().to_dict(flatten=False)

    images = observation["images"]
    if isinstance(images, dict):
        # A single camera collapses to one array; the exported model's input
        # signature is named "images" (see manifest.json input_features).
        observation["images"] = next(iter(images.values()))

    inference_model = InferenceModel(export_dir)
    assert inference_model.backend == "onnx"

    action = inference_model.select_action(observation)
    assert action.shape[-1] == 2


def test_submit_training_job_for_missing_project_returns_404() -> None:
    """Submitting against a project that doesn't exist is a clean 404, not a crash."""
    client = TestClient(app)

    response = client.post(
        "/api/jobs:train",
        json={
            "project_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "policy": "act",
            "model_name": "Orphan Job",
            "max_steps": 100,
        },
    )
    assert response.status_code == 404, response.text


def test_interrupt_job_marks_job_canceled(migrated_db: None) -> None:
    """POST /api/jobs/{id}:interrupt flips a running job to CANCELED.

    Overrides `get_scheduler` with a minimal stand-in instead of running the
    real `Scheduler` (which needs the FastAPI lifespan to populate
    `app.state.scheduler`, and spins up real background worker processes).
    """
    import asyncio
    from types import SimpleNamespace

    # submit_train_job requires a real project row (FK constraint), so create
    # one directly rather than pointing the job at a fake project id.
    client = TestClient(app)
    project_id = uuid4()
    response = client.post("/api/projects", json={"id": str(project_id), "name": "Interrupt Project"})
    assert response.status_code == 201, response.text

    async def _submit_and_run(project_id: UUID) -> str:
        from services.job_service import JobService

        job = await JobService.submit_train_job(
            TrainJobPayload(
                project_id=project_id,
                dataset_id=uuid4(),
                policy="act",
                model_name="Interrupt Me",
                max_steps=100,
            )
        )
        await JobService.update_job_status(job.id, status=JobStatus.RUNNING, message="Training started")
        return str(job.id)

    job_id = asyncio.run(_submit_and_run(project_id))

    fake_scheduler = SimpleNamespace(training_interrupt_event=mp.Event())
    app.dependency_overrides[get_scheduler] = lambda: fake_scheduler
    try:
        response = client.post(f"/api/jobs/{job_id}:interrupt")
        assert response.status_code == 200, response.text
    finally:
        app.dependency_overrides.clear()

    assert fake_scheduler.training_interrupt_event.is_set()

    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == JobStatus.CANCELED
