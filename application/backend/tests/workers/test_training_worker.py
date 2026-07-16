# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Pre-import to break circular dependency: scheduler -> training_worker -> scheduler
import core.scheduler  # noqa: F401
from schemas.base_job import JobStatus, JobType
from schemas.dataset import Snapshot
from schemas.job import TrainingPrecision, TrainJobPayload
from schemas.model import Model

if TYPE_CHECKING:
    from pathlib import Path


MODULE = "workers.training_worker"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_payload(
    *, compile_model: bool = True, precision: TrainingPrecision = TrainingPrecision.BF16_MIXED
) -> TrainJobPayload:
    return TrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="test-model",
        max_steps=100,
        batch_size=8,
        num_workers=0,
        auto_scale_batch_size=False,
        compile_model=compile_model,
        precision=precision,
    )


def _make_model(tmp_path: Path) -> Model:
    model_dir = tmp_path / "models" / str(uuid4())
    model_dir.mkdir(parents=True)
    return Model(
        id=uuid4(),
        project_id=uuid4(),
        dataset_id=uuid4(),
        path=str(model_dir),
        name="test-model",
        snapshot_id=uuid4(),
        policy="act",
        properties={},
        train_job_id=uuid4(),
        version=1,
        created_at=None,
    )


def _make_snapshot(tmp_path: Path) -> Snapshot:
    snap_dir = tmp_path / "snapshots" / str(uuid4())
    snap_dir.mkdir(parents=True)
    return Snapshot(id=uuid4(), dataset_id=uuid4(), path=str(snap_dir))


def _make_job(payload: TrainJobPayload) -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.type = JobType.TRAINING
    job.status = JobStatus.PENDING
    job.message = "Job created"
    job.payload = payload.model_dump()
    return job


def _make_settings(tmp_path: Path) -> MagicMock:
    settings = MagicMock()
    settings.models_dir = tmp_path / "models"
    settings.cache_dir = tmp_path / "cache"
    return settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_queue():
    return queue.Queue()


@pytest.fixture
def stop_event():
    return mp.Event()


@pytest.fixture
def interrupt_event():
    return mp.Event()


@pytest.fixture
def worker(stop_event, interrupt_event, event_queue):
    """Build a minimal TrainingWorker without triggering circular imports from scheduler."""
    from workers.training_worker import TrainingWorker

    w = object.__new__(TrainingWorker)
    # Mirror BaseProcessWorker/TrainingWorker wiring: should_stop() reads the
    # private events; _should_interrupt() also reads the public interrupt_event.
    w._stop_event = stop_event
    w._interrupt_event = interrupt_event
    w.interrupt_event = interrupt_event
    w.queue = event_queue
    return w


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTraining:
    """Tests for _train_model delegation to the selected training backend."""

    @pytest.mark.anyio
    async def test_training_failure_propagates_as_failed_job(self, worker, tmp_path):
        """When the backend raises, the job ends as FAILED."""
        payload = _make_payload(compile_model=False)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        backend = MagicMock()
        backend.train = AsyncMock(side_effect=RuntimeError("training failed"))

        dispatcher = MagicMock()
        dispatcher.is_alive = MagicMock(return_value=False)

        failed_job = MagicMock()
        failed_job.id = job.id
        failed_job.status = JobStatus.FAILED

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.get_training_backend", return_value=backend),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService"),
        ):
            MockJobService.update_job_status = AsyncMock(return_value=failed_job)
            MockJobService.update_job = AsyncMock(return_value=MagicMock())

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            backend.train.assert_awaited_once()
            MockJobService.update_job.assert_called_once()
            failed_call = MockJobService.update_job_status.call_args_list[0]
            assert failed_call.kwargs["status"] == JobStatus.FAILED

    @pytest.mark.anyio
    async def test_cancellation_raised_by_backend_marks_canceled(self, worker, tmp_path):
        """A TrainingCanceledError ends the job CANCELED, not FAILED, and creates no model."""
        from services.training_backends import TrainingCanceledError

        payload = _make_payload(compile_model=False)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        backend = MagicMock()
        backend.train = AsyncMock(side_effect=TrainingCanceledError("Training canceled"))

        dispatcher = MagicMock()
        dispatcher.is_alive = MagicMock(return_value=False)

        canceled_job = MagicMock()
        canceled_job.id = job.id
        canceled_job.status = JobStatus.CANCELED

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.get_training_backend", return_value=backend),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            MockJobService.update_job_status = AsyncMock(return_value=canceled_job)
            MockJobService.update_job = AsyncMock(return_value=MagicMock())
            MockModelService.create_model = AsyncMock()

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            MockModelService.create_model.assert_not_called()
            canceled_call = MockJobService.update_job_status.call_args_list[0]
            assert canceled_call.kwargs["status"] == JobStatus.CANCELED

    @pytest.mark.anyio
    async def test_interrupt_after_silent_stop_marks_canceled(self, worker, interrupt_event, tmp_path):
        """A backend that stops cooperatively (no raise) while interrupted ends CANCELED."""
        payload = _make_payload(compile_model=False)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        interrupt_event.set()

        backend = MagicMock()
        backend.train = AsyncMock()

        dispatcher = MagicMock()
        dispatcher.is_alive = MagicMock(return_value=False)

        canceled_job = MagicMock()
        canceled_job.id = job.id
        canceled_job.status = JobStatus.CANCELED

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.get_training_backend", return_value=backend),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            MockJobService.update_job_status = AsyncMock(return_value=canceled_job)
            MockJobService.update_job = AsyncMock(return_value=MagicMock())
            MockModelService.create_model = AsyncMock()

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            MockModelService.create_model.assert_not_called()
            assert MockJobService.update_job_status.call_args_list[0].kwargs["status"] == JobStatus.CANCELED

    @pytest.mark.anyio
    async def test_successful_training_creates_model(self, worker, event_queue, tmp_path):
        """A successful backend run completes the job and persists the model."""
        payload = _make_payload(compile_model=True)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        backend = MagicMock()
        backend.train = AsyncMock()

        dispatcher = MagicMock()
        dispatcher.is_alive = MagicMock(return_value=False)

        completed_job = MagicMock()
        completed_job.id = job.id
        completed_job.status = JobStatus.COMPLETED

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.get_training_backend", return_value=backend),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            MockJobService.update_job_status = AsyncMock(return_value=completed_job)
            MockJobService.update_job = AsyncMock(return_value=MagicMock())
            MockModelService.create_model = AsyncMock(return_value=model)

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            backend.train.assert_awaited_once()
            MockModelService.create_model.assert_awaited_once_with(model)
            assert MockJobService.update_job_status.call_args_list[0].kwargs["status"] == JobStatus.COMPLETED

    @pytest.mark.anyio
    async def test_context_passes_output_and_cache_dirs(self, worker, tmp_path):
        """The worker builds a context pointing at the model and cache directories."""
        payload = _make_payload(compile_model=False)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        captured = {}

        async def _capture(context):
            captured["context"] = context

        backend = MagicMock()
        backend.train = AsyncMock(side_effect=_capture)

        dispatcher = MagicMock()
        dispatcher.is_alive = MagicMock(return_value=False)

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.get_training_backend", return_value=backend),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            MockJobService.update_job_status = AsyncMock(return_value=MagicMock())
            MockJobService.update_job = AsyncMock(return_value=MagicMock())
            MockModelService.create_model = AsyncMock(return_value=model)

            await worker._train_model(job, model, snapshot, payload, base_model=None)

        context = captured["context"]
        assert str(context.output_dir) == model.path
        assert context.cache_dir == tmp_path / "cache" / str(job.id)
        assert context.snapshot is snapshot

    @pytest.mark.anyio
    async def test_suspension_requeues_job_for_reattach(self, worker, tmp_path):
        """A TrainingSuspendedError leaves the job PENDING (reattachable), not terminal."""
        from services.training_backends import TrainingSuspendedError

        payload = _make_payload(compile_model=False)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        backend = MagicMock()
        backend.train = AsyncMock(side_effect=TrainingSuspendedError("shutting down"))

        dispatcher = MagicMock()
        dispatcher.is_alive = MagicMock(return_value=False)

        pending_job = MagicMock()
        pending_job.id = job.id
        pending_job.status = JobStatus.PENDING

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.get_training_backend", return_value=backend),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            MockJobService.update_job_status = AsyncMock(return_value=pending_job)
            MockJobService.update_job = AsyncMock(return_value=MagicMock())
            MockModelService.create_model = AsyncMock()

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            # The job is requeued (PENDING) so the next start reattaches; no model,
            # and it is never marked FAILED or CANCELED.
            MockModelService.create_model.assert_not_called()
            statuses = [c.kwargs["status"] for c in MockJobService.update_job_status.call_args_list]
            assert statuses == [JobStatus.PENDING]

    @pytest.mark.anyio
    async def test_context_wires_reattach_fields_from_payload(self, worker, tmp_path):
        """The context carries the persisted remote_job_id and a suspend predicate."""
        payload = _make_payload(compile_model=False)
        remote_job_id = uuid4()
        payload.remote_job_id = remote_job_id
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        captured = {}

        async def _capture(context):
            captured["context"] = context
            # Evaluate the shutdown predicate while training is active, before the
            # finally-block sets the interrupt event to stop the dispatcher.
            captured["suspend_during_train"] = context.should_suspend()

        backend = MagicMock()
        backend.train = AsyncMock(side_effect=_capture)

        dispatcher = MagicMock()
        dispatcher.is_alive = MagicMock(return_value=False)

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.get_training_backend", return_value=backend),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            MockJobService.update_job_status = AsyncMock(return_value=MagicMock())
            MockJobService.update_job = AsyncMock(return_value=MagicMock())
            MockModelService.create_model = AsyncMock(return_value=model)

            await worker._train_model(job, model, snapshot, payload, base_model=None)

        context = captured["context"]
        assert context.remote_job_id == remote_job_id
        # should_suspend mirrors the worker's global stop signal (shutdown), which
        # is distinct from a per-job cancel (interrupt_event). No stop was requested
        # during training, so it is False.
        assert captured["suspend_during_train"] is False
        assert context.on_remote_job_id is not None

    @pytest.mark.anyio
    async def test_persist_remote_job_id_updates_payload(self, worker, tmp_path):
        """The persist callback retains the snapshot identity with the remote job id."""
        payload = _make_payload(compile_model=False)
        snapshot_id = uuid4()
        payload.snapshot_id = snapshot_id
        job = _make_job(payload)

        with patch(f"{MODULE}.JobService") as MockJobService:
            MockJobService.update_job_payload = AsyncMock(return_value=MagicMock())

            remote_job_id = uuid4()
            await worker._persist_remote_job_id(job, payload, remote_job_id)

            assert payload.remote_job_id == remote_job_id
            MockJobService.update_job_payload.assert_awaited_once()
            args, _ = MockJobService.update_job_payload.call_args
            assert args[0] == job.id
            assert args[1].remote_job_id == remote_job_id
            assert args[1].snapshot_id == snapshot_id
