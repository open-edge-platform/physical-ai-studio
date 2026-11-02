# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Startup tests for the SSH remote-trainer feature's loopback-binding enforcement."""

from __future__ import annotations

import threading
from queue import Queue
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from loguru import logger

from core import lifecycle as lifecycle_module
from core.security import get_ssh_feature_availability
from schemas.remote_trainer import RemoteTrainer, SshConnection
from services.event_processor import EventProcessor, EventType

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _clear_caches() -> Generator[None]:
    get_ssh_feature_availability.cache_clear()
    yield
    get_ssh_feature_availability.cache_clear()


@pytest.fixture
def _stub_heavy_startup(monkeypatch):
    scheduler = MagicMock()
    scheduler.mp_stop_event = MagicMock()
    scheduler.event_queue = MagicMock()
    monkeypatch.setattr(lifecycle_module, "Scheduler", lambda: scheduler)
    monkeypatch.setattr(lifecycle_module, "EventProcessor", lambda queue: MagicMock())
    monkeypatch.setattr(lifecycle_module, "setup_logging", lambda: None)
    monkeypatch.setattr(lifecycle_module, "setup_uvicorn_logging", lambda: None)
    robot_manager = MagicMock()
    robot_manager.find_robots = AsyncMock()
    monkeypatch.setattr(lifecycle_module, "RobotConnectionManager", lambda: robot_manager)
    return scheduler


@pytest.mark.anyio
async def test_shutdown_keeps_event_processor_draining(monkeypatch, _stub_heavy_startup) -> None:
    scheduler = _stub_heavy_startup
    scheduler.event_queue = Queue(maxsize=1)
    monkeypatch.setattr(lifecycle_module, "EventProcessor", EventProcessor)
    drained = threading.Event()
    received = []
    main_thread = threading.get_ident()

    def shutdown() -> None:
        assert threading.get_ident() != main_thread
        for value in range(3):
            scheduler.event_queue.put((EventType.JOB_UPDATE, value), timeout=2)
        assert drained.wait(timeout=2), "Event loop stopped draining while workers shut down"

    def receive(event, value) -> None:
        received.append(value)
        if value == 2:
            drained.set()

    scheduler.shutdown.side_effect = shutdown
    app = FastAPI()
    async with lifecycle_module.lifespan(app):
        app.state.event_processor.subscribe([EventType.JOB_UPDATE], receive)

    assert received == [0, 1, 2]
    scheduler.shutdown.assert_called_once()


async def _run_startup_and_capture_logs(app: FastAPI) -> list[str]:
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        async with lifecycle_module.lifespan(app):
            pass
    finally:
        logger.remove(sink_id)
    return messages


@pytest.mark.anyio
async def test_startup_restores_persistent_ssh_trainers_as_well_as_tunnels() -> None:
    trainer = RemoteTrainer(id=uuid4(), name="gpu", connection=SshConnection(ssh_host_alias="gpu-box"))
    service = MagicMock()
    service.list_remote_trainers = AsyncMock(return_value=[trainer])
    db = MagicMock()
    db.__aenter__ = AsyncMock(return_value=MagicMock())
    db.__aexit__ = AsyncMock(return_value=False)
    with (
        patch.object(lifecycle_module, "get_async_db_session_ctx", return_value=db),
        patch.object(lifecycle_module, "get_ssh_feature_availability", return_value=MagicMock(active=True)),
        patch.object(lifecycle_module, "RemoteTrainerService", return_value=service) as service_class,
        patch.object(lifecycle_module.remote_trainer_tunnel_manager, "start_all", new_callable=AsyncMock),
    ):
        await lifecycle_module._start_remote_trainer_tunnels()

    service_class._start_persistent_trainer_in_background.assert_called_once_with(trainer, None)


@pytest.mark.anyio
async def test_ssh_feature_on_non_loopback_logs_critical_and_deactivates(monkeypatch, _stub_heavy_startup) -> None:
    monkeypatch.setenv("HOST", "0.0.0.0")
    get_ssh_feature_availability.cache_clear()
    app = FastAPI()
    messages = await _run_startup_and_capture_logs(app)
    assert app.state.ssh_feature_availability.network_exposed is True
    assert app.state.ssh_feature_availability.active is False
    assert any("SSH remote-trainer feature disabled at startup" in message for message in messages)


@pytest.mark.anyio
async def test_ssh_feature_on_loopback_logs_no_warning_and_stays_active(monkeypatch, _stub_heavy_startup) -> None:
    monkeypatch.setenv("HOST", "127.0.0.1")
    get_ssh_feature_availability.cache_clear()
    app = FastAPI()
    messages = await _run_startup_and_capture_logs(app)
    assert app.state.ssh_feature_availability.active is True
    assert not any("SSH remote-trainer feature disabled at startup" in message for message in messages)
