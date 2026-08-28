import asyncio
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID, uuid4

import httpx
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from repositories.remote_trainer_repo import RemoteTrainerRepository
from schemas.hardware import DeviceInfo, DeviceType, StorageInfo
from schemas.remote_trainer import (
    HealthStatus,
    RemoteTrainer,
    RemoteTrainerCreate,
    RemoteTrainerHealth,
    RemoteTrainerUpdate,
)

_HEALTH_CHECK_TIMEOUT_S = 5.0

# RemoteTrainerService is instantiated fresh per request (see
# api.dependencies.get_remote_trainer_service), so an instance-level cache
# can't prevent concurrent requests for the same trainer (multiple browser
# tabs/components polling at once) from each issuing their own /health,
# /devices, /storage round trip. Coalesce those into one in-flight probe per
# trainer, shared across requests via this module-level table.
_inflight_checks: dict[UUID, asyncio.Task[RemoteTrainerHealth]] = {}


class RemoteTrainerService:
    """Manage global direct remote trainer endpoint configurations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = RemoteTrainerRepository(session)

    async def list_remote_trainers(self) -> list[RemoteTrainer]:
        """Return configured endpoints ordered by their creation time."""
        return await self.repo.list_ordered()

    async def get_remote_trainer(self, remote_trainer_id: UUID) -> RemoteTrainer:
        """Return one configured endpoint or raise a not-found error."""
        remote_trainer = await self.repo.get_by_id(remote_trainer_id)
        if remote_trainer is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_TRAINER, str(remote_trainer_id))
        return remote_trainer

    async def check_remote_trainer(self, remote_trainer_id: UUID) -> RemoteTrainerHealth:
        """Check a configured trainer's liveness and available compute devices.

        Concurrent callers (multiple browser tabs, or the trainers table and a
        training dialog both polling the same trainer) share one in-flight
        probe instead of each triggering their own /health, /devices,
        /storage round trip against the trainer.
        """
        remote_trainer = await self.get_remote_trainer(remote_trainer_id)
        task = _inflight_checks.get(remote_trainer_id)
        if task is None:
            task = asyncio.ensure_future(self._probe_remote_trainer(remote_trainer_id, remote_trainer))
            _inflight_checks[remote_trainer_id] = task

            def _clear_inflight(done: asyncio.Task[RemoteTrainerHealth], trainer_id: UUID = remote_trainer_id) -> None:
                if _inflight_checks.get(trainer_id) is done:
                    del _inflight_checks[trainer_id]

            task.add_done_callback(_clear_inflight)
        return await task

    @staticmethod
    async def _probe_remote_trainer(remote_trainer_id: UUID, remote_trainer: RemoteTrainer) -> RemoteTrainerHealth:
        """Probe a trainer's /health, /devices, /storage once and build its health report."""
        checked_at = datetime.now(UTC)
        started = perf_counter()
        base_url = str(remote_trainer.url).rstrip("/")
        timeout = httpx.Timeout(_HEALTH_CHECK_TIMEOUT_S)
        status: HealthStatus = "healthy"
        reason_code: str | None = None
        devices: list[DeviceInfo] = []
        storage: StorageInfo | None = None

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False) as client:
                health_response = await client.get(f"{base_url}/health")
                health_response.raise_for_status()
                health_payload = health_response.json()
                if not isinstance(health_payload, dict) or health_payload.get("status") != "healthy":
                    status, reason_code = "degraded", "unhealthy"
                else:
                    devices_response = await client.get(f"{base_url}/devices")
                    devices_response.raise_for_status()
                    devices_payload = devices_response.json()
                    if not isinstance(devices_payload, list):
                        status, reason_code = "degraded", "invalid_devices_response"
                    else:
                        validated_devices = [DeviceInfo.model_validate(device) for device in devices_payload]
                        devices = [
                            device for device in validated_devices if device.type in {DeviceType.XPU, DeviceType.CUDA}
                        ]
                        storage = await RemoteTrainerService._fetch_storage(client, base_url)
        except httpx.TimeoutException:
            status, reason_code = "unreachable", "timeout"
        except httpx.HTTPStatusError:
            status, reason_code = "unreachable", "http_error"
        except httpx.HTTPError:
            status, reason_code = "unreachable", "connection_failed"
        except (ValidationError, ValueError):
            status, reason_code = "degraded", "invalid_devices_response"

        return RemoteTrainerHealth(
            remote_trainer_id=remote_trainer_id,
            status=status,
            checked_at=checked_at,
            latency_ms=round((perf_counter() - started) * 1000),
            devices=devices,
            storage=storage,
            reason_code=reason_code,
        )

    @staticmethod
    async def _fetch_storage(client: httpx.AsyncClient, base_url: str) -> StorageInfo | None:
        """Best-effort fetch of the trainer's available storage."""
        try:
            storage_response = await client.get(f"{base_url}/storage")
            storage_response.raise_for_status()
            return StorageInfo.model_validate(storage_response.json())
        except (httpx.HTTPError, ValidationError, ValueError):
            return None

    async def create_remote_trainer(self, config: RemoteTrainerCreate) -> RemoteTrainer:
        """Persist a direct trainer endpoint."""
        remote_trainer = RemoteTrainer(id=uuid4(), **config.model_dump())
        try:
            return await self.repo.save(remote_trainer)
        except IntegrityError as error:
            await self.session.rollback()
            raise ResourceAlreadyExistsError(
                "Remote trainer",
                "A trainer with this URL is already configured.",
            ) from error

    async def update_remote_trainer(self, remote_trainer_id: UUID, update: RemoteTrainerUpdate) -> RemoteTrainer:
        """Update a direct trainer endpoint."""
        remote_trainer = await self.repo.get_by_id(remote_trainer_id)
        if remote_trainer is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_TRAINER, str(remote_trainer_id))
        try:
            return await self.repo.update(remote_trainer, update.model_dump(exclude_none=True, exclude_unset=True))
        except IntegrityError as error:
            await self.session.rollback()
            raise ResourceAlreadyExistsError(
                "Remote trainer",
                "A trainer with this URL is already configured.",
            ) from error

    async def delete_remote_trainer(self, remote_trainer_id: UUID) -> None:
        """Delete a configured endpoint without changing already-submitted jobs."""
        if await self.repo.get_by_id(remote_trainer_id) is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_TRAINER, str(remote_trainer_id))
        await self.repo.delete_by_id(remote_trainer_id)
