from datetime import UTC, datetime
from time import perf_counter
from typing import Literal
from uuid import UUID, uuid4

import httpx
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from repositories.remote_trainer_repo import RemoteTrainerRepository
from schemas.hardware import DeviceInfo, DeviceType, StorageInfo
from schemas.remote_trainer import RemoteTrainer, RemoteTrainerCreate, RemoteTrainerHealth, RemoteTrainerUpdate

_HEALTH_CHECK_TIMEOUT_S = 5.0


class RemoteTrainerService:
    """Manage global direct remote trainer endpoint configurations."""

    @staticmethod
    async def list_remote_trainers() -> list[RemoteTrainer]:
        """Return configured endpoints ordered by their creation time."""
        async with get_async_db_session_ctx() as session:
            return await RemoteTrainerRepository(session).list_ordered()

    @staticmethod
    async def get_remote_trainer(remote_trainer_id: UUID) -> RemoteTrainer:
        """Return one configured endpoint or raise a not-found error."""
        async with get_async_db_session_ctx() as session:
            remote_trainer = await RemoteTrainerRepository(session).get_by_id(remote_trainer_id)
            if remote_trainer is None:
                raise ResourceNotFoundError(ResourceType.REMOTE_TRAINER, str(remote_trainer_id))
            return remote_trainer

    @classmethod
    async def check_remote_trainer(cls, remote_trainer_id: UUID) -> RemoteTrainerHealth:
        """Check a configured trainer's liveness and available compute devices."""
        remote_trainer = await cls.get_remote_trainer(remote_trainer_id)
        checked_at = datetime.now(UTC)
        started = perf_counter()
        base_url = str(remote_trainer.url).rstrip("/")
        timeout = httpx.Timeout(_HEALTH_CHECK_TIMEOUT_S)
        status: Literal["healthy", "degraded", "unreachable"] = "healthy"
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
                        storage = await cls._fetch_storage(client, base_url)
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

    @staticmethod
    async def create_remote_trainer(config: RemoteTrainerCreate) -> RemoteTrainer:
        """Persist a direct remote trainer endpoint."""
        remote_trainer = RemoteTrainer(id=uuid4(), **config.model_dump())
        async with get_async_db_session_ctx() as session:
            try:
                return await RemoteTrainerRepository(session).save(remote_trainer)
            except IntegrityError as error:
                await session.rollback()
                raise ResourceAlreadyExistsError(
                    "Remote trainer",
                    "A trainer with this URL is already configured.",
                ) from error

    @staticmethod
    async def update_remote_trainer(remote_trainer_id: UUID, update: RemoteTrainerUpdate) -> RemoteTrainer:
        """Update a direct remote trainer endpoint."""
        async with get_async_db_session_ctx() as session:
            repository = RemoteTrainerRepository(session)
            remote_trainer = await repository.get_by_id(remote_trainer_id)
            if remote_trainer is None:
                raise ResourceNotFoundError(ResourceType.REMOTE_TRAINER, str(remote_trainer_id))
            try:
                return await repository.update(remote_trainer, update.model_dump(exclude_none=True, exclude_unset=True))
            except IntegrityError as error:
                await session.rollback()
                raise ResourceAlreadyExistsError(
                    "Remote trainer",
                    "A trainer with this URL is already configured.",
                ) from error

    @staticmethod
    async def delete_remote_trainer(remote_trainer_id: UUID) -> None:
        """Delete a configured endpoint without changing already-submitted jobs."""
        async with get_async_db_session_ctx() as session:
            repository = RemoteTrainerRepository(session)
            if await repository.get_by_id(remote_trainer_id) is None:
                raise ResourceNotFoundError(ResourceType.REMOTE_TRAINER, str(remote_trainer_id))
            await repository.delete_by_id(remote_trainer_id)
