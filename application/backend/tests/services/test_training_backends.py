# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for training backend selection and the progress dispatcher."""

from __future__ import annotations

import multiprocessing as mp
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from loguru import logger

from schemas.job import TrainingTarget, TrainJobPayload
from services.training_backends import get_training_backend
from services.training_backends.local import LocalTrainingBackend
from services.training_backends.remote import SNAPSHOT_UPLOAD_PROGRESS, TRAINING_PROGRESS_END, RemoteTrainingBackend
from services.training_service import TrainingTrackingDispatcher


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.trainer.request_timeout_s = 5.0
    return settings


def _payload(target: TrainingTarget) -> TrainJobPayload:
    return TrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        training_target=target,
        remote_trainer_id=uuid4() if target is TrainingTarget.REMOTE else None,
        remote_trainer_url="https://trainer.test" if target is TrainingTarget.REMOTE else None,
        remote_trainer_name="gpu-box-1" if target is TrainingTarget.REMOTE else None,
    )


def test_get_training_backend_returns_local_for_local_job() -> None:
    backend = get_training_backend(_payload(TrainingTarget.LOCAL))
    assert isinstance(backend, LocalTrainingBackend)


def test_get_training_backend_returns_remote_for_remote_job() -> None:
    settings = _settings()
    with (
        patch("settings.get_settings", return_value=settings),
        patch("services.training_backends.remote.get_settings", return_value=settings),
    ):
        backend = get_training_backend(_payload(TrainingTarget.REMOTE))
    assert isinstance(backend, RemoteTrainingBackend)

    # The pinned trainer name reaches the backend, so its job logs are attributable.
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    try:
        backend._log.info("check")
    finally:
        logger.remove(sink_id)
    assert messages == ["[gpu-box-1] check"]


def test_dispatcher_report_enqueues_progress_tuple() -> None:
    dispatcher = TrainingTrackingDispatcher(uuid4(), mp.Queue(), mp.Event(), AsyncMock())

    dispatcher.report(42, message="halfway", extra_info={"train/loss_step": 0.1})

    assert dispatcher.queue.get(timeout=1) == (42, "halfway", {"train/loss_step": 0.1})


def test_dispatcher_report_defaults_message_and_extra_to_none() -> None:
    dispatcher = TrainingTrackingDispatcher(uuid4(), mp.Queue(), mp.Event(), AsyncMock())

    dispatcher.report(10)

    assert dispatcher.queue.get(timeout=1) == (10, None, None)


def test_remote_progress_maps_raw_0_100_into_training_window() -> None:
    """The trainer reports raw 0-100; the backend windows it exactly once."""
    to_local = RemoteTrainingBackend._to_local_progress

    assert to_local(0) == SNAPSHOT_UPLOAD_PROGRESS
    assert to_local(100) == TRAINING_PROGRESS_END
    span = TRAINING_PROGRESS_END - SNAPSHOT_UPLOAD_PROGRESS
    assert to_local(50) == SNAPSHOT_UPLOAD_PROGRESS + round(50 * span / 100)
    # Monotonic and clamped within the reserved window.
    assert all(SNAPSHOT_UPLOAD_PROGRESS <= to_local(p) <= TRAINING_PROGRESS_END for p in range(101))
