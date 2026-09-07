# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for training backend selection and the progress dispatcher."""

from __future__ import annotations

import multiprocessing as mp
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from loguru import logger

from schemas.hardware import DeviceType
from schemas.job import LocalTrainJobPayload, RemoteTrainJobPayload, SshTrainJobPayload, TrainingTarget, TrainJobPayload
from schemas.remote_server import RemoteServer
from services.training_backends import get_training_backend
from services.training_backends.local import LocalTrainingBackend
from services.training_backends.remote import SNAPSHOT_UPLOAD_PROGRESS, TRAINING_PROGRESS_END, RemoteTrainingBackend
from services.training_backends.ssh import SshTrainingBackend
from services.training_service import TrainingTrackingDispatcher


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.trainer.request_timeout_s = 5.0
    return settings


def _active_ssh_feature():
    from core.security import SshFeatureAvailability

    return SshFeatureAvailability(enabled=True, network_exposed=False)


def _payload(target: TrainingTarget) -> TrainJobPayload:
    if target is TrainingTarget.LOCAL:
        return LocalTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            policy="act",
            model_name="model",
        )
    if target is TrainingTarget.REMOTE:
        return RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            policy="act",
            model_name="model",
            remote_trainer_id=uuid4(),
            remote_trainer_url="https://trainer.test",
            remote_trainer_name="gpu-box-1",
        )
    return SshTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        remote_server_id=uuid4(),
    )


async def test_get_training_backend_returns_local_for_local_job() -> None:
    backend = await get_training_backend(_payload(TrainingTarget.LOCAL), uuid4())
    assert isinstance(backend, LocalTrainingBackend)


async def test_get_training_backend_returns_remote_for_remote_job() -> None:
    settings = _settings()
    with (
        patch("settings.get_settings", return_value=settings),
        patch("services.training_backends.remote.get_settings", return_value=settings),
    ):
        backend = await get_training_backend(_payload(TrainingTarget.REMOTE), uuid4())
    assert isinstance(backend, RemoteTrainingBackend)

    # The pinned trainer name reaches the backend, so its job logs are attributable.
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    try:
        backend._log.info("check")
    finally:
        logger.remove(sink_id)
    assert messages == ["[gpu-box-1] check"]


async def test_get_training_backend_returns_ssh_for_ssh_job() -> None:
    payload = _payload(TrainingTarget.SSH)
    assert payload.remote_server_id is not None
    server = RemoteServer(
        id=payload.remote_server_id,
        name="gpu-box-1",
        ssh_host_alias="gpu-box",
        device_type=DeviceType.CUDA,
    )
    with (
        patch("services.remote_server_service.RemoteServerService.get_remote_server", AsyncMock(return_value=server)),
        patch("core.security.get_ssh_feature_availability", return_value=_active_ssh_feature()),
    ):
        backend = await get_training_backend(payload, uuid4())
    assert isinstance(backend, SshTrainingBackend)
    assert backend._server is server


async def test_get_training_backend_raises_without_remote_server_id() -> None:
    payload = _payload(TrainingTarget.SSH).model_copy(update={"remote_server_id": None})
    with (
        patch("core.security.get_ssh_feature_availability", return_value=_active_ssh_feature()),
        pytest.raises(ValueError, match="remote server"),
    ):
        await get_training_backend(payload, uuid4())


async def test_get_training_backend_rejects_ssh_job_when_feature_inactive() -> None:
    """Defense in depth: a job persisted while active must not silently start if disabled since."""
    from core.security import SshFeatureAvailability
    from exceptions import SshFeatureDisabledError

    payload = _payload(TrainingTarget.SSH)
    inactive = SshFeatureAvailability(enabled=False, network_exposed=False)
    with (
        patch("core.security.get_ssh_feature_availability", return_value=inactive),
        pytest.raises(SshFeatureDisabledError),
    ):
        await get_training_backend(payload, uuid4())


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
