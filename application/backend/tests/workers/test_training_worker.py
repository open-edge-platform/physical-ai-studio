# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing as mp
import queue
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from schemas.base_job import JobStatus, JobType
from schemas.dataset import Snapshot
from schemas.job import LocalTrainJobPayload, RemoteTrainJobPayload, TrainingPrecision, TrainJobPayload
from schemas.model import Model
from services.snapshot_service import SnapshotService

if TYPE_CHECKING:
    from pathlib import Path


MODULE = "workers.training_worker"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_payload(
    *,
    compile_model: bool = True,
    precision: TrainingPrecision = TrainingPrecision.BF16_MIXED,
    lora_enabled: bool = False,
    lora_use_dora: bool = False,
) -> TrainJobPayload:
    return LocalTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="pi05" if lora_enabled else "act",
        model_name="test-model",
        max_epochs=5,
        batch_size=8,
        num_workers=0,
        auto_scale_batch_size=False,
        compile_model=compile_model,
        precision=precision,
        lora_enabled=lora_enabled,
        lora_use_dora=lora_use_dora,
    )


def _make_remote_payload(*, remote_trainer_id: UUID | None = None) -> RemoteTrainJobPayload:
    return RemoteTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="test-model",
        remote_trainer_id=remote_trainer_id or uuid4(),
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
    job.start_time = None
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
def cancel_event():
    """Each training process receives its own shared cancellation event."""
    return mp.Event()


@pytest.fixture
def worker(stop_event, cancel_event, event_queue):
    """Build a minimal TrainingWorker without allocating a process."""
    from workers.training_worker import TrainingWorker

    w = object.__new__(TrainingWorker)
    # Mirror BaseProcessWorker/TrainingWorker wiring: should_stop() reads the
    # private stop event; _should_interrupt() also reads the per-job event.
    w._stop_event = stop_event
    w._interrupt_event = mp.Event()
    w.cancel_event = cancel_event
    w.queue = event_queue
    return w


@pytest.fixture(autouse=True)
def session_scope():
    @asynccontextmanager
    async def _scope():
        yield MagicMock()

    with patch(f"{MODULE}.get_async_db_session_ctx", new=_scope):
        yield


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
            patch(f"{MODULE}.get_training_backend", AsyncMock(return_value=backend)),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService"),
        ):
            job_service = MockJobService.return_value
            job_service.update_job_status = AsyncMock(return_value=failed_job)
            job_service.update_job = AsyncMock(return_value=MagicMock())

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            backend.train.assert_awaited_once()
            job_service.update_job.assert_called_once()
            assert isinstance(job_service.update_job.call_args.kwargs["update"]["start_time"], datetime)
            failed_call = job_service.update_job_status.call_args_list[0]
            assert failed_call.kwargs["status"] == JobStatus.FAILED

    @pytest.mark.anyio
    async def test_remote_reattach_preserves_original_start_time(self, worker, tmp_path):
        """Reattaching after a restart must not restart the elapsed-time clock."""
        from services.training_backends import TrainingCanceledError

        payload = _make_remote_payload().model_copy(update={"remote_job_id": uuid4()})
        job = _make_job(payload)
        started = datetime(2026, 9, 23, 14, 0, tzinfo=UTC)
        job.start_time = started
        backend = MagicMock()
        backend.train = AsyncMock(side_effect=TrainingCanceledError("stopped"))
        dispatcher = MagicMock()
        dispatcher.is_alive.return_value = False

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.get_training_backend", AsyncMock(return_value=backend)),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as job_service_type,
        ):
            job_service = job_service_type.return_value
            job_service.update_job_status = AsyncMock(return_value=job)
            job_service.update_job = AsyncMock(return_value=job)
            await worker._train_model(job, _make_model(tmp_path), None, payload)

        assert job_service.update_job.call_args.kwargs["update"]["start_time"] == started

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
            patch(f"{MODULE}.get_training_backend", AsyncMock(return_value=backend)),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            job_service = MockJobService.return_value
            model_service = MockModelService.return_value
            job_service.update_job_status = AsyncMock(return_value=canceled_job)
            job_service.update_job = AsyncMock(return_value=MagicMock())
            model_service.create_model = AsyncMock()

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            model_service.create_model.assert_not_called()
            canceled_call = job_service.update_job_status.call_args_list[0]
            assert canceled_call.kwargs["status"] == JobStatus.CANCELED

    @pytest.mark.anyio
    async def test_interrupt_after_silent_stop_marks_canceled(self, worker, cancel_event, tmp_path):
        """A backend that stops cooperatively (no raise) while interrupted ends CANCELED."""
        payload = _make_payload(compile_model=False)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        cancel_event.set()

        backend = MagicMock()
        backend.train = AsyncMock()

        dispatcher = MagicMock()
        dispatcher.is_alive = MagicMock(return_value=False)

        canceled_job = MagicMock()
        canceled_job.id = job.id
        canceled_job.status = JobStatus.CANCELED

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.get_training_backend", AsyncMock(return_value=backend)),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            job_service = MockJobService.return_value
            model_service = MockModelService.return_value
            job_service.update_job_status = AsyncMock(return_value=canceled_job)
            job_service.update_job = AsyncMock(return_value=MagicMock())
            model_service.create_model = AsyncMock()

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            backend.train.assert_not_awaited()
            model_service.create_model.assert_not_called()
            assert job_service.update_job_status.call_args_list[0].kwargs["status"] == JobStatus.CANCELED

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
            patch(f"{MODULE}.get_training_backend", AsyncMock(return_value=backend)),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            job_service = MockJobService.return_value
            model_service = MockModelService.return_value
            job_service.update_job_status = AsyncMock(return_value=completed_job)
            job_service.update_job = AsyncMock(return_value=MagicMock())
            model_service.create_model = AsyncMock(return_value=model)

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            backend.train.assert_awaited_once()
            model_service.create_model.assert_awaited_once_with(model)
            assert job_service.update_job_status.call_args_list[0].kwargs["status"] == JobStatus.COMPLETED

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
            patch(f"{MODULE}.get_training_backend", AsyncMock(return_value=backend)),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            job_service = MockJobService.return_value
            model_service = MockModelService.return_value
            job_service.update_job_status = AsyncMock(return_value=MagicMock())
            job_service.update_job = AsyncMock(return_value=MagicMock())
            model_service.create_model = AsyncMock(return_value=model)

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
            patch(f"{MODULE}.get_training_backend", AsyncMock(return_value=backend)),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            job_service = MockJobService.return_value
            model_service = MockModelService.return_value
            job_service.update_job_status = AsyncMock(return_value=pending_job)
            job_service.update_job = AsyncMock(return_value=MagicMock())
            model_service.create_model = AsyncMock()

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            # The job is requeued (PENDING) so the next start reattaches; no model,
            # and it is never marked FAILED or CANCELED.
            model_service.create_model.assert_not_called()
            statuses = [c.kwargs["status"] for c in job_service.update_job_status.call_args_list]
            assert statuses == [JobStatus.PENDING]

    @pytest.mark.anyio
    @pytest.mark.parametrize(("cancel_requested", "signal_cancel"), [(False, False), (True, False), (False, True)])
    async def test_context_wires_reattach_fields_from_payload(self, worker, tmp_path, cancel_requested, signal_cancel):
        """The context carries the persisted remote_job_id and a suspend predicate."""
        payload = _make_remote_payload()
        payload.cancel_requested = cancel_requested
        remote_job_id = uuid4()
        payload.remote_job_id = remote_job_id
        if signal_cancel:
            worker.cancel_event.set()
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        captured = {}

        async def _capture(context):
            captured["context"] = context
            # Evaluate the shutdown predicate while training is active, before the
            # finally-block sets the interrupt event to stop the dispatcher.
            captured["suspend_during_train"] = context.should_suspend()
            captured["cancel_during_train"] = context.should_cancel_job()

        backend = MagicMock()
        backend.train = AsyncMock(side_effect=_capture)

        dispatcher = MagicMock()
        dispatcher.is_alive = MagicMock(return_value=False)

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.get_training_backend", AsyncMock(return_value=backend)),
            patch(f"{MODULE}.TrainingTrackingDispatcher", return_value=dispatcher),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
        ):
            job_service = MockJobService.return_value
            model_service = MockModelService.return_value
            job_service.update_job_status = AsyncMock(return_value=MagicMock())
            job_service.update_job = AsyncMock(return_value=MagicMock())
            model_service.create_model = AsyncMock(return_value=model)

            await worker._train_model(job, model, snapshot, payload, base_model=None)

        context = captured["context"]
        assert context.remote_job_id == remote_job_id
        # should_suspend mirrors the worker's global stop signal (shutdown), which
        # is distinct from a per-job cancel (interrupt_event). No stop was requested
        # during training, so it is False.
        assert captured["suspend_during_train"] is False
        assert captured["cancel_during_train"] is (cancel_requested or signal_cancel)
        assert context.on_remote_job_id is not None
        assert context.on_authentication_issue is not None

    @pytest.mark.anyio
    async def test_persist_remote_job_id_updates_payload(self, worker, tmp_path):
        """The persist callback retains the snapshot identity with the remote job id."""
        payload = _make_payload(compile_model=False)
        snapshot_id = uuid4()
        payload.snapshot_id = snapshot_id
        job = _make_job(payload)

        with patch(f"{MODULE}.JobService") as MockJobService:
            job_service = MockJobService.return_value
            job_service.update_job_payload = AsyncMock(return_value=MagicMock())

            remote_job_id = uuid4()
            await worker._persist_remote_job_id(job, payload, remote_job_id)

            assert payload.remote_job_id == remote_job_id
            job_service.update_job_payload.assert_awaited_once()
            args, _ = job_service.update_job_payload.call_args
            assert args[0] == job.id
            assert args[1].remote_job_id == remote_job_id
            assert args[1].snapshot_id == snapshot_id

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("lora_enabled", "lora_use_dora"),
        [(False, False), (True, False), (True, True)],
    )
    async def test_run_training_job_carries_lora_flags_into_model_properties(
        self, worker, tmp_path, lora_enabled, lora_use_dora
    ):
        """The model's `properties` are the only place LoRA/DoRA provenance survives."""
        payload = _make_payload(lora_enabled=lora_enabled, lora_use_dora=lora_use_dora)
        job = _make_job(payload)
        dataset = MagicMock()
        snapshot = _make_snapshot(tmp_path)

        captured = {}

        async def _capture(_job, model, _snapshot, _payload, base_model):
            captured["model"] = model

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.DatasetService") as MockDatasetService,
            patch(f"{MODULE}.SnapshotService") as MockSnapshotService,
            patch.object(worker, "_train_model", side_effect=_capture),
        ):
            MockDatasetService.return_value.get_dataset_by_id = AsyncMock(return_value=dataset)
            MockSnapshotService.generate_snapshot_folder_name = MagicMock(return_value="snap")
            MockSnapshotService.return_value.create_snapshot_for_dataset = AsyncMock(return_value=snapshot)

            await worker._run_training_job(job, payload)

        assert captured["model"].properties == {
            "lora_enabled": lora_enabled,
            "lora_use_dora": lora_use_dora,
            "snapflow_enabled": False,
        }
        assert captured["model"].lora_enabled is lora_enabled
        assert captured["model"].lora_use_dora is lora_use_dora


class TestAssignedJob:
    @pytest.mark.anyio
    async def test_concurrent_jobs_use_distinct_snapshot_paths(self, worker, tmp_path) -> None:
        payload = _make_payload()
        jobs = [_make_job(payload), _make_job(payload)]
        settings = _make_settings(tmp_path)
        settings.snapshot_dir = tmp_path / "snapshots"
        with (
            patch(f"{MODULE}.get_settings", return_value=settings),
            patch(f"{MODULE}.DatasetService") as dataset_service,
            patch(f"{MODULE}.SnapshotService") as snapshot_service,
            patch("services.snapshot_service.time.strftime", return_value="snapshot_same_second"),
            patch.object(worker, "_train_model", AsyncMock()),
        ):
            dataset_service.return_value.get_dataset_by_id = AsyncMock(return_value=MagicMock())
            snapshot_service.generate_snapshot_folder_name.side_effect = SnapshotService.generate_snapshot_folder_name
            create_snapshot = AsyncMock(return_value=_make_snapshot(tmp_path))
            snapshot_service.return_value.create_snapshot_for_dataset = create_snapshot
            for job in jobs:
                await worker._run_training_job(job, payload)

        assert [call.args for call in snapshot_service.generate_snapshot_folder_name.call_args_list] == [
            (job.id,) for job in jobs
        ]
        destinations = [call.kwargs["destination"] for call in create_snapshot.await_args_list]
        assert destinations == [settings.snapshot_dir / f"snapshot_same_second_{job.id}" for job in jobs]
        assert len(set(destinations)) == 2

    @pytest.mark.anyio
    @pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.CANCELED, JobStatus.COMPLETED])
    async def test_runs_only_assigned_claimed_job(self, worker, status) -> None:
        job = _make_job(_make_payload())
        job.status = status
        worker.job_id = job.id
        with (
            patch(
                f"{MODULE}.JobService.claim_pending_train_job",
                AsyncMock(return_value=job if status == JobStatus.PENDING else None),
            ) as claim_job,
            patch.object(worker, "_run_training_job", AsyncMock()) as run_job,
        ):
            await worker.run_loop()

        claim_job.assert_awaited_once_with(job.id)
        assert run_job.await_count == (1 if status == JobStatus.PENDING else 0)

    @pytest.mark.anyio
    async def test_shutdown_before_start_does_not_run_job(self, worker) -> None:
        worker._interrupt_event.set()
        with patch.object(worker, "_run_training_job", AsyncMock()) as run_job:
            await worker.run_loop()
        run_job.assert_not_awaited()


class TestSnapFlowProvenance:
    """The models list badges a distilled checkpoint from `Model.properties`.

    A model row is only persisted for a run that finished without being
    canceled, and the distillation boundary is validated to fall inside the
    epoch budget, so a completed SnapFlow job always produced a distilled
    checkpoint and the request is a sound source for the flag.
    """

    @staticmethod
    async def _built_model(worker, tmp_path, payload: TrainJobPayload) -> Model:
        job = _make_job(payload)
        train = AsyncMock()

        with (
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.DatasetService") as MockDatasetService,
            patch(f"{MODULE}.SnapshotService") as MockSnapshotService,
            patch.object(type(worker), "_train_model", train),
        ):
            MockDatasetService.return_value.get_dataset_by_id = AsyncMock()
            snapshot_service = MockSnapshotService.return_value
            snapshot_service.create_snapshot_for_dataset = AsyncMock(return_value=_make_snapshot(tmp_path))
            await worker._run_training_job(job, payload)

        assert train.await_args is not None
        return train.await_args.args[1]

    @pytest.mark.anyio
    async def test_a_flow_matching_run_is_not_marked_distilled(self, worker, tmp_path):
        model = await self._built_model(worker, tmp_path, _make_payload())

        assert model.snapflow_enabled is False

    @pytest.mark.anyio
    async def test_a_distillation_run_is_marked_on_the_model(self, worker, tmp_path):
        payload = LocalTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            policy="pi05",
            model_name="test-model",
            max_epochs=8,
            snapflow_enabled=True,
            snapflow_distill_epochs=3,
        )

        model = await self._built_model(worker, tmp_path, payload)

        assert model.snapflow_enabled is True
