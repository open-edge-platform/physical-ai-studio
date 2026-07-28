from __future__ import annotations

from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from pydantic import AnyHttpUrl

from schemas.remote_trainer import RemoteTrainer
from services import RemoteTrainerService

MODULE = "services.remote_trainer_service"


class _Response:
    def __init__(self, payload: object, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> object:
        return self._payload


class _Client:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = iter(responses)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def get(self, _url: str) -> _Response:
        return next(self._responses)


def _trainer() -> RemoteTrainer:
    return RemoteTrainer(id=uuid4(), name="trainer", url=AnyHttpUrl("https://trainer.test"))


@pytest.mark.anyio
async def test_check_remote_trainer_reports_healthy_devices() -> None:
    trainer = _trainer()
    client = _Client(
        [
            _Response({"status": "healthy"}),
            _Response(
                [
                    {"type": "cpu", "name": "CPU", "memory": None, "index": None},
                    {"type": "npu", "name": "NPU", "memory": 17179869184, "index": 0},
                    {"type": "xpu", "name": "Intel Arc", "memory": 17179869184, "index": 0},
                    {"type": "cuda", "name": "NVIDIA A100", "memory": 85899345920, "index": 0},
                ]
            ),
            _Response({"total_bytes": 1_000_000_000_000, "free_bytes": 600_000_000_000}),
        ]
    )

    with (
        patch.object(RemoteTrainerService, "get_remote_trainer", new=AsyncMock(return_value=trainer)),
        patch(f"{MODULE}.httpx.AsyncClient", return_value=client) as async_client,
    ):
        result = await RemoteTrainerService(MagicMock()).check_remote_trainer(trainer.id)

    assert result.remote_trainer_id == trainer.id
    assert result.status == "healthy"
    assert result.reason_code is None
    assert [(device.type, device.name, device.index) for device in result.devices] == [
        ("xpu", "Intel Arc", 0),
        ("cuda", "NVIDIA A100", 0),
    ]
    assert result.storage is not None
    assert result.storage.total_bytes == 1_000_000_000_000
    assert result.storage.free_bytes == 600_000_000_000
    async_client.assert_called_once_with(timeout=httpx.Timeout(5.0), follow_redirects=False, trust_env=False)


@pytest.mark.anyio
async def test_check_remote_trainer_tolerates_missing_storage_endpoint() -> None:
    trainer = _trainer()
    client = _Client(
        [
            _Response({"status": "healthy"}),
            _Response([{"type": "cuda", "name": "NVIDIA A100", "memory": 85899345920, "index": 0}]),
            _Response({}, httpx.HTTPStatusError("not found", request=None, response=None)),  # type: ignore[arg-type]
        ]
    )

    with (
        patch.object(RemoteTrainerService, "get_remote_trainer", new=AsyncMock(return_value=trainer)),
        patch(f"{MODULE}.httpx.AsyncClient", return_value=client),
    ):
        result = await RemoteTrainerService(MagicMock()).check_remote_trainer(trainer.id)

    assert result.status == "healthy"
    assert result.storage is None


@pytest.mark.anyio
async def test_check_remote_trainer_reports_timeout_without_upstream_details() -> None:
    trainer = _trainer()
    client = _Client([_Response({}, httpx.ReadTimeout("timed out"))])

    with (
        patch.object(RemoteTrainerService, "get_remote_trainer", new=AsyncMock(return_value=trainer)),
        patch(f"{MODULE}.httpx.AsyncClient", return_value=client),
    ):
        result = await RemoteTrainerService(MagicMock()).check_remote_trainer(trainer.id)

    assert result.status == "unreachable"
    assert result.reason_code == "timeout"
    assert result.devices == []


@pytest.mark.anyio
async def test_check_remote_trainer_reports_malformed_devices_as_degraded() -> None:
    trainer = _trainer()
    client = _Client([_Response({"status": "healthy"}), _Response({"not": "a list"})])

    with (
        patch.object(RemoteTrainerService, "get_remote_trainer", new=AsyncMock(return_value=trainer)),
        patch(f"{MODULE}.httpx.AsyncClient", return_value=client),
    ):
        result = await RemoteTrainerService(MagicMock()).check_remote_trainer(trainer.id)

    assert result.status == "degraded"
    assert result.reason_code == "invalid_devices_response"
