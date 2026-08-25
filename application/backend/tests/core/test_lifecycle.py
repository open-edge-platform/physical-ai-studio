# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Startup tests for the SSH remote-trainer feature's loopback-binding enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from loguru import logger

from core import lifecycle as lifecycle_module
from core.security import get_ssh_feature_availability

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
async def test_ssh_feature_enabled_on_non_loopback_logs_critical_and_deactivates(
    monkeypatch, _stub_heavy_startup
) -> None:
    monkeypatch.setenv("SSH_REMOTE_TRAINER_ENABLED", "true")
    monkeypatch.setenv("HOST", "0.0.0.0")
    get_ssh_feature_availability.cache_clear()
    app = FastAPI()
    messages = await _run_startup_and_capture_logs(app)
    assert app.state.ssh_feature_availability.enabled is True
    assert app.state.ssh_feature_availability.network_exposed is True
    assert app.state.ssh_feature_availability.active is False
    assert any("SSH remote-trainer feature disabled at startup" in message for message in messages)


@pytest.mark.anyio
async def test_ssh_feature_enabled_on_loopback_logs_no_warning_and_stays_active(
    monkeypatch, _stub_heavy_startup
) -> None:
    monkeypatch.setenv("SSH_REMOTE_TRAINER_ENABLED", "true")
    monkeypatch.setenv("HOST", "127.0.0.1")
    get_ssh_feature_availability.cache_clear()
    app = FastAPI()
    messages = await _run_startup_and_capture_logs(app)
    assert app.state.ssh_feature_availability.active is True
    assert not any("SSH remote-trainer feature disabled at startup" in message for message in messages)


@pytest.mark.anyio
async def test_ssh_feature_disabled_by_default_logs_no_warning(monkeypatch, _stub_heavy_startup) -> None:
    monkeypatch.delenv("SSH_REMOTE_TRAINER_ENABLED", raising=False)
    monkeypatch.setenv("HOST", "0.0.0.0")
    get_ssh_feature_availability.cache_clear()
    app = FastAPI()
    messages = await _run_startup_and_capture_logs(app)
    assert app.state.ssh_feature_availability.enabled is False
    assert app.state.ssh_feature_availability.active is False
    assert not any("SSH remote-trainer feature disabled at startup" in message for message in messages)
