# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing as mp
import queue
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Pre-import to break circular dependency: scheduler -> training_worker -> scheduler
import core.scheduler  # noqa: F401

from schemas.base_job import JobStatus, JobType
from schemas.dataset import Snapshot
from schemas.job import TrainJobPayload, TrainingPrecision
from schemas.model import Model
from services.event_processor import EventType


MODULE = "workers.training_worker"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_payload(
    *, compile_model: bool = True, precision: TrainingPrecision = TrainingPrecision.DEFAULT
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
    settings.supported_backends = []
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
    w._stop_event = stop_event
    w.interrupt_event = interrupt_event
    w.queue = event_queue
    return w


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompileFallback:
    """Tests for the compile-failure fallback mechanism in _train_model."""

    @pytest.mark.anyio
    async def test_fallback_retries_without_compile_on_failure(self, worker, event_queue, tmp_path):
        """When compile_model=True and trainer.fit raises, the worker should:
        1. Notify via JOB_UPDATE with a fallback message
        2. Re-create the policy with compile_model=False
        3. Retry trainer.fit successfully
        """
        payload = _make_payload(compile_model=True)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        first_policy = MagicMock()
        second_policy = MagicMock()
        setup_policy_returns = [first_policy, second_policy]

        first_trainer = MagicMock()
        first_trainer.fit = MagicMock(side_effect=RuntimeError("compile failed"))
        second_trainer = MagicMock()
        second_trainer.fit = MagicMock()
        trainer_returns = [first_trainer, second_trainer]

        updated_job = MagicMock()
        updated_job.id = job.id
        updated_job.status = JobStatus.RUNNING
        updated_job.message = "Model compilation failed, falling back to non-compiled training."

        completed_job = MagicMock()
        completed_job.id = job.id
        completed_job.status = JobStatus.COMPLETED

        with (
            patch(f"{MODULE}.setup_policy", side_effect=setup_policy_returns) as mock_setup,
            patch(f"{MODULE}.Trainer", side_effect=trainer_returns),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
            patch(f"{MODULE}.LeRobotDataModule"),
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.CSVLogger"),
            patch(f"{MODULE}.ModelCheckpoint"),
            patch(f"{MODULE}.TrainingTrackingDispatcher") as MockDispatcher,
            patch(f"{MODULE}.TrainingTrackingCallback"),
            patch(f"{MODULE}.TrainingLogCallback"),
            patch(f"{MODULE}.get_torch_device", return_value="cpu"),
            patch(f"{MODULE}.get_lightning_strategy", return_value="auto"),
        ):
            MockDispatcher.return_value = MagicMock()
            MockDispatcher.return_value.start = MagicMock()
            MockDispatcher.return_value.is_alive = MagicMock(return_value=False)

            MockJobService.update_job_status = AsyncMock(
                side_effect=[
                    MagicMock(),  # "Training started"
                    updated_job,  # compile fallback notification
                    completed_job,  # "Training finished"
                ]
            )
            MockModelService.create_model = AsyncMock(return_value=model)

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            # setup_policy called twice: first with compile, then without
            assert mock_setup.call_count == 2
            assert mock_setup.call_args_list[0].kwargs["compile_model"] is True
            assert mock_setup.call_args_list[1].kwargs["compile_model"] is False

            # First trainer.fit was called (and raised), second was called (and succeeded)
            first_trainer.fit.assert_called_once()
            second_trainer.fit.assert_called_once()

            # JobService.update_job_status called with fallback message
            fallback_call = MockJobService.update_job_status.call_args_list[1]
            assert fallback_call.kwargs["status"] == JobStatus.RUNNING
            assert "falling back" in fallback_call.kwargs["message"].lower()

            # Final status is COMPLETED
            completed_call = MockJobService.update_job_status.call_args_list[2]
            assert completed_call.kwargs["status"] == JobStatus.COMPLETED

            # Queue received JOB_UPDATE with fallback message
            events = []
            while not event_queue.empty():
                events.append(event_queue.get_nowait())

            job_updates = [e for e in events if e[0] == EventType.JOB_UPDATE]
            assert any("falling back" in str(getattr(e[1], "message", "")).lower() for e in job_updates)

    @pytest.mark.anyio
    async def test_no_fallback_when_compile_disabled(self, worker, event_queue, tmp_path):
        """When compile_model=False and trainer.fit raises, the error should propagate
        without any fallback attempt."""
        payload = _make_payload(compile_model=False)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        policy = MagicMock()
        trainer = MagicMock()
        trainer.fit = MagicMock(side_effect=RuntimeError("training failed"))

        failed_job = MagicMock()
        failed_job.id = job.id
        failed_job.status = JobStatus.FAILED

        with (
            patch(f"{MODULE}.setup_policy", return_value=policy) as mock_setup,
            patch(f"{MODULE}.Trainer", return_value=trainer),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
            patch(f"{MODULE}.LeRobotDataModule"),
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.CSVLogger"),
            patch(f"{MODULE}.ModelCheckpoint"),
            patch(f"{MODULE}.TrainingTrackingDispatcher") as MockDispatcher,
            patch(f"{MODULE}.TrainingTrackingCallback"),
            patch(f"{MODULE}.TrainingLogCallback"),
            patch(f"{MODULE}.get_torch_device", return_value="cpu"),
            patch(f"{MODULE}.get_lightning_strategy", return_value="auto"),
        ):
            MockDispatcher.return_value = MagicMock()
            MockDispatcher.return_value.start = MagicMock()
            MockDispatcher.return_value.is_alive = MagicMock(return_value=False)

            MockJobService.update_job_status = AsyncMock(
                side_effect=[
                    MagicMock(),  # "Training started"
                    failed_job,  # "Training failed"
                ]
            )

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            # setup_policy only called once — no fallback retry
            assert mock_setup.call_count == 1

            # trainer.fit only called once
            trainer.fit.assert_called_once()

            # Job ended as FAILED
            failed_call = MockJobService.update_job_status.call_args_list[1]
            assert failed_call.kwargs["status"] == JobStatus.FAILED

    @pytest.mark.anyio
    async def test_fallback_failure_still_marks_job_failed(self, worker, event_queue, tmp_path):
        """When compile_model=True, the first fit fails triggering fallback, but the
        second fit also fails — the job should end as FAILED."""
        payload = _make_payload(compile_model=True)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        first_policy = MagicMock()
        second_policy = MagicMock()

        first_trainer = MagicMock()
        first_trainer.fit = MagicMock(side_effect=RuntimeError("compile failed"))
        second_trainer = MagicMock()
        second_trainer.fit = MagicMock(side_effect=RuntimeError("also failed without compile"))
        trainer_returns = [first_trainer, second_trainer]

        updated_job = MagicMock()
        updated_job.id = job.id
        updated_job.message = "Model compilation failed, falling back to non-compiled training."

        failed_job = MagicMock()
        failed_job.id = job.id
        failed_job.status = JobStatus.FAILED

        with (
            patch(f"{MODULE}.setup_policy", side_effect=[first_policy, second_policy]) as mock_setup,
            patch(f"{MODULE}.Trainer", side_effect=trainer_returns),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService"),
            patch(f"{MODULE}.LeRobotDataModule"),
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.CSVLogger"),
            patch(f"{MODULE}.ModelCheckpoint"),
            patch(f"{MODULE}.TrainingTrackingDispatcher") as MockDispatcher,
            patch(f"{MODULE}.TrainingTrackingCallback"),
            patch(f"{MODULE}.TrainingLogCallback"),
            patch(f"{MODULE}.get_torch_device", return_value="cpu"),
            patch(f"{MODULE}.get_lightning_strategy", return_value="auto"),
        ):
            MockDispatcher.return_value = MagicMock()
            MockDispatcher.return_value.start = MagicMock()
            MockDispatcher.return_value.is_alive = MagicMock(return_value=False)

            MockJobService.update_job_status = AsyncMock(
                side_effect=[
                    MagicMock(),  # "Training started"
                    updated_job,  # compile fallback notification
                    failed_job,  # "Training failed"
                ]
            )

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            # setup_policy called twice (compile + fallback)
            assert mock_setup.call_count == 2

            # Both trainers had fit called
            first_trainer.fit.assert_called_once()
            second_trainer.fit.assert_called_once()

            # Job ended as FAILED
            failed_call = MockJobService.update_job_status.call_args_list[2]
            assert failed_call.kwargs["status"] == JobStatus.FAILED

    @pytest.mark.anyio
    async def test_successful_compile_no_fallback(self, worker, event_queue, tmp_path):
        """When compile_model=True and trainer.fit succeeds on the first try,
        no fallback should occur."""
        payload = _make_payload(compile_model=True)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        policy = MagicMock()
        trainer = MagicMock()
        trainer.fit = MagicMock()  # succeeds

        completed_job = MagicMock()
        completed_job.id = job.id
        completed_job.status = JobStatus.COMPLETED

        with (
            patch(f"{MODULE}.setup_policy", return_value=policy) as mock_setup,
            patch(f"{MODULE}.Trainer", return_value=trainer),
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
            patch(f"{MODULE}.LeRobotDataModule"),
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.CSVLogger"),
            patch(f"{MODULE}.ModelCheckpoint"),
            patch(f"{MODULE}.TrainingTrackingDispatcher") as MockDispatcher,
            patch(f"{MODULE}.TrainingTrackingCallback"),
            patch(f"{MODULE}.TrainingLogCallback"),
            patch(f"{MODULE}.get_torch_device", return_value="cpu"),
            patch(f"{MODULE}.get_lightning_strategy", return_value="auto"),
        ):
            MockDispatcher.return_value = MagicMock()
            MockDispatcher.return_value.start = MagicMock()
            MockDispatcher.return_value.is_alive = MagicMock(return_value=False)

            MockJobService.update_job_status = AsyncMock(
                side_effect=[
                    MagicMock(),  # "Training started"
                    completed_job,  # "Training finished"
                ]
            )
            MockModelService.create_model = AsyncMock(return_value=model)

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            # setup_policy called only once — no fallback
            assert mock_setup.call_count == 1
            assert mock_setup.call_args_list[0].kwargs["compile_model"] is True

            # trainer.fit called only once
            trainer.fit.assert_called_once()

            # No fallback message in update_job_status calls
            for call in MockJobService.update_job_status.call_args_list:
                msg = call.kwargs.get("message", "")
                assert "falling back" not in msg.lower()

    @pytest.mark.anyio
    async def test_precision_default_passes_none_to_trainer(self, worker, event_queue, tmp_path):
        """When precision is 'default', None should be passed to Trainer."""
        payload = _make_payload(compile_model=False, precision=TrainingPrecision.DEFAULT)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        policy = MagicMock()

        with (
            patch(f"{MODULE}.setup_policy", return_value=policy),
            patch(f"{MODULE}.Trainer") as MockTrainer,
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
            patch(f"{MODULE}.LeRobotDataModule"),
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.CSVLogger"),
            patch(f"{MODULE}.ModelCheckpoint"),
            patch(f"{MODULE}.TrainingTrackingDispatcher") as MockDispatcher,
            patch(f"{MODULE}.TrainingTrackingCallback"),
            patch(f"{MODULE}.TrainingLogCallback"),
            patch(f"{MODULE}.get_torch_device", return_value="cpu"),
            patch(f"{MODULE}.get_lightning_strategy", return_value="auto"),
        ):
            MockDispatcher.return_value = MagicMock()
            MockDispatcher.return_value.start = MagicMock()
            MockDispatcher.return_value.is_alive = MagicMock(return_value=False)

            trainer_instance = MagicMock()
            trainer_instance.fit = MagicMock()
            MockTrainer.return_value = trainer_instance

            completed_job = MagicMock()
            MockJobService.update_job_status = AsyncMock(side_effect=[MagicMock(), completed_job])
            MockModelService.create_model = AsyncMock(return_value=model)

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            # Trainer was called with precision=None
            assert MockTrainer.call_args.kwargs["precision"] is None

    @pytest.mark.anyio
    async def test_precision_bf16_passes_string_to_trainer(self, worker, event_queue, tmp_path):
        """When precision is 'bf16-mixed', the string should be passed to Trainer."""
        payload = _make_payload(compile_model=False, precision=TrainingPrecision.BF16_MIXED)
        model = _make_model(tmp_path)
        snapshot = _make_snapshot(tmp_path)
        job = _make_job(payload)

        policy = MagicMock()

        with (
            patch(f"{MODULE}.setup_policy", return_value=policy),
            patch(f"{MODULE}.Trainer") as MockTrainer,
            patch(f"{MODULE}.JobService") as MockJobService,
            patch(f"{MODULE}.ModelService") as MockModelService,
            patch(f"{MODULE}.LeRobotDataModule"),
            patch(f"{MODULE}.get_settings", return_value=_make_settings(tmp_path)),
            patch(f"{MODULE}.CSVLogger"),
            patch(f"{MODULE}.ModelCheckpoint"),
            patch(f"{MODULE}.TrainingTrackingDispatcher") as MockDispatcher,
            patch(f"{MODULE}.TrainingTrackingCallback"),
            patch(f"{MODULE}.TrainingLogCallback"),
            patch(f"{MODULE}.get_torch_device", return_value="cpu"),
            patch(f"{MODULE}.get_lightning_strategy", return_value="auto"),
        ):
            MockDispatcher.return_value = MagicMock()
            MockDispatcher.return_value.start = MagicMock()
            MockDispatcher.return_value.is_alive = MagicMock(return_value=False)

            trainer_instance = MagicMock()
            trainer_instance.fit = MagicMock()
            MockTrainer.return_value = trainer_instance

            completed_job = MagicMock()
            MockJobService.update_job_status = AsyncMock(side_effect=[MagicMock(), completed_job])
            MockModelService.create_model = AsyncMock(return_value=model)

            await worker._train_model(job, model, snapshot, payload, base_model=None)

            assert MockTrainer.call_args.kwargs["precision"] == "bf16-mixed"
