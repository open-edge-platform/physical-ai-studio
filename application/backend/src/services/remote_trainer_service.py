import asyncio
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID, uuid4

import httpx
from loguru import logger
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.security import get_ssh_feature_availability
from db.schema import JobDB
from exceptions import (
    ResourceAlreadyExistsError,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    SshFeatureDisabledError,
)
from repositories.remote_trainer_repo import RemoteTrainerRepository
from schemas.base_job import JobStatus, JobType
from schemas.hardware import DeviceInfo, DeviceType, StorageInfo
from schemas.remote_trainer import (
    HealthStatus,
    RemoteTrainer,
    RemoteTrainerConnectionMode,
    RemoteTrainerCreate,
    RemoteTrainerHealth,
    RemoteTrainerUpdate,
)
from services import remote_trainer_tunnel_manager
from services.ssh import persistent_trainer

_HEALTH_CHECK_TIMEOUT_S = 5.0

# RemoteTrainerService is instantiated fresh per request (see
# api.dependencies.get_remote_trainer_service), so an instance-level cache
# can't prevent concurrent requests for the same trainer (multiple browser
# tabs/components polling at once) from each issuing their own /health,
# /devices, /storage round trip. Coalesce those into one in-flight probe per
# trainer, shared across requests via this module-level table.
_inflight_checks: dict[UUID, asyncio.Task[RemoteTrainerHealth]] = {}

# Background trainer-container launches (see `_start_persistent_trainer_in_background`),
# kept referenced so asyncio never garbage-collects a task mid-flight.
_background_launches: dict[UUID, asyncio.Task[None]] = {}


# Fields whose column is nullable, so an explicit null in an update means "clear it"
# rather than "not provided".
_NULLABLE_UPDATE_FIELDS = frozenset({"ssh_host_alias", "ssh_connection", "ssh_remote_port", "ssh_local_port"})
_CONNECTION_FIELDS = {"connection_mode", "url", "ssh_host_alias", "ssh_connection", "ssh_remote_port", "ssh_local_port"}


class RemoteTrainerService:
    """Manage configured remote trainer endpoints."""

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
        """Probe a trainer's /health, /devices, /storage once and build its health report.

        If this trainer's persistent container is still being launched in the
        background (see `RemoteTrainerService._start_persistent_trainer_in_background`),
        nothing is listening on the tunnel yet - dialing it would just read as
        a generic "unreachable" failure. Reporting the actual in-progress phase
        instead skips a probe that cannot succeed and tells the user their save
        is working, not broken.
        """
        checked_at = datetime.now(UTC)
        launch_phase = persistent_trainer.get_launch_phase(remote_trainer_id)
        if launch_phase is not None:
            return RemoteTrainerHealth(
                remote_trainer_id=remote_trainer_id,
                status="starting",
                checked_at=checked_at,
                latency_ms=None,
                devices=[],
                storage=None,
                reason_code=launch_phase,
            )
        if launch_failure := persistent_trainer.get_launch_failure(remote_trainer_id):
            return RemoteTrainerHealth(
                remote_trainer_id=remote_trainer_id,
                status="degraded",
                checked_at=checked_at,
                latency_ms=None,
                devices=[],
                storage=None,
                reason_code=launch_failure,
            )
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

        status, reason_code = RemoteTrainerService._require_managed_accelerator(
            remote_trainer, status, reason_code, devices
        )

        if status != "unreachable":
            persistent_trainer.mark_reachable(remote_trainer_id)
        elif persistent_trainer.is_within_startup_grace_period(remote_trainer_id):
            # A trainer whose launch attempt began recently is given the
            # benefit of the doubt: it may simply still be pulling its image
            # or warming up its own health endpoint, not genuinely broken.
            status, reason_code = "starting", "Waiting for the trainer container to come online…"

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
    def _require_managed_accelerator(
        remote_trainer: RemoteTrainer,
        status: HealthStatus,
        reason_code: str | None,
        devices: list[DeviceInfo],
    ) -> tuple[HealthStatus, str | None]:
        if status == "healthy" and remote_trainer.connection_mode is RemoteTrainerConnectionMode.SSH and not devices:
            return "degraded", "container_accelerator_unavailable"
        return status, reason_code

    @staticmethod
    async def _fetch_storage(client: httpx.AsyncClient, base_url: str) -> StorageInfo | None:
        """Best-effort fetch of the trainer's available storage."""
        try:
            storage_response = await client.get(f"{base_url}/storage")
            storage_response.raise_for_status()
            return StorageInfo.model_validate(storage_response.json())
        except (httpx.HTTPError, ValidationError, ValueError):
            return None

    async def _ensure_unique_ssh_remote_port(self, remote_trainer: RemoteTrainer) -> None:
        """Reject two managed containers that would bind the same host port."""
        if remote_trainer.connection_mode is not RemoteTrainerConnectionMode.SSH:
            return
        for existing in await self.repo.list_ordered():
            if (
                existing.id != remote_trainer.id
                and existing.connection_mode is RemoteTrainerConnectionMode.SSH
                and existing.ssh_remote_port == remote_trainer.ssh_remote_port
                and existing.ssh_host_alias == remote_trainer.ssh_host_alias
                and existing.ssh_connection == remote_trainer.ssh_connection
            ):
                raise ResourceAlreadyExistsError(
                    "Remote trainer",
                    f"SSH host already has a trainer using remote port {remote_trainer.ssh_remote_port}.",
                )

    async def _require_no_active_jobs(self, remote_trainer_id: UUID) -> None:
        result = await self.session.execute(
            select(JobDB.id)
            .where(
                JobDB.type == JobType.TRAINING,
                JobDB.status.in_((JobStatus.PENDING, JobStatus.RUNNING)),
                func.json_extract(JobDB.payload, "$.remote_trainer_id") == str(remote_trainer_id),
            )
            .limit(1)
        )
        if result.scalar_one_or_none() is not None:
            raise ResourceInUseError(ResourceType.REMOTE_TRAINER, remote_trainer_id)

    @staticmethod
    async def _cancel_launch(remote_trainer_id: UUID) -> None:
        task = _background_launches.pop(remote_trainer_id, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _start_persistent_trainer_in_background(
        remote_trainer: RemoteTrainer, accepted_host_key_fingerprint: str | None
    ) -> None:
        """Launch the trainer container without blocking the save request on it.

        Resolving/pulling the trainer image and starting the container can take
        minutes (a multi-gigabyte pull on first use); a caller saving a trainer
        must not hang waiting for that. This fires the launch in the background
        and only logs a failure - `sync_tunnel` still runs (and is awaited)
        synchronously, since it is fast and is what the caller actually needs
        confirmed before the save response returns.
        """
        if remote_trainer.connection_mode is not RemoteTrainerConnectionMode.SSH:
            return

        async def _start() -> None:
            try:
                await persistent_trainer.start(remote_trainer, accepted_host_key_fingerprint)
            except Exception:
                logger.exception(
                    "Failed to start the persistent trainer container for '{}'; the SSH tunnel is open, but no "
                    "container is listening on the other end yet. Edit and save the trainer again to retry.",
                    remote_trainer.name,
                )

        task = asyncio.create_task(_start())
        _background_launches[remote_trainer.id] = task

        def _clear(done: asyncio.Task[None]) -> None:
            if _background_launches.get(remote_trainer.id) is done:
                _background_launches.pop(remote_trainer.id, None)

        task.add_done_callback(_clear)

    async def create_remote_trainer(
        self, config: RemoteTrainerCreate, accepted_host_key_fingerprint: str | None = None
    ) -> RemoteTrainer:
        """Persist a trainer endpoint, requiring SSH availability for managed trainers."""
        self._require_ssh_feature_if_tunneled(config.connection_mode)
        remote_trainer = RemoteTrainer(id=uuid4(), **config.model_dump())
        await self._ensure_unique_ssh_remote_port(remote_trainer)
        try:
            saved = await self.repo.save(remote_trainer)
        except IntegrityError as error:
            await self.session.rollback()
            raise ResourceAlreadyExistsError(
                "Remote trainer",
                "A trainer with this URL is already configured.",
            ) from error
        try:
            await remote_trainer_tunnel_manager.sync_tunnel(saved, accepted_host_key_fingerprint)
        except Exception:
            await self.repo.delete_by_id(saved.id)
            raise
        self._start_persistent_trainer_in_background(saved, accepted_host_key_fingerprint)
        return saved

    async def update_remote_trainer(
        self,
        remote_trainer_id: UUID,
        update: RemoteTrainerUpdate,
        accepted_host_key_fingerprint: str | None = None,
    ) -> RemoteTrainer:
        """Update a remote trainer endpoint."""
        remote_trainer = await self.repo.get_by_id(remote_trainer_id)
        if remote_trainer is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_TRAINER, str(remote_trainer_id))
        try:
            # exclude_unset (not exclude_none) so an explicit null clears an existing
            # tunnel field instead of being dropped as "not provided".
            data = update.model_dump(exclude_unset=True)
            data = {k: v for k, v in data.items() if v is not None or k in _NULLABLE_UPDATE_FIELDS}
            validated = RemoteTrainerCreate.model_validate(
                {
                    **remote_trainer.model_dump(),
                    **data,
                }
            )
            should_normalize_connection = (
                "connection_mode" in data
                or "url" in data
                or any(data.get(field) is not None for field in _CONNECTION_FIELDS - {"connection_mode", "url"})
            )
            if should_normalize_connection:
                self._require_ssh_feature_if_tunneled(validated.connection_mode)
                data.update(validated.model_dump(include=_CONNECTION_FIELDS))
            connection_changed = any(
                getattr(remote_trainer, field) != getattr(validated, field) for field in _CONNECTION_FIELDS
            )
            if connection_changed and (
                remote_trainer.connection_mode is RemoteTrainerConnectionMode.SSH
                or validated.connection_mode is RemoteTrainerConnectionMode.SSH
            ):
                await self._require_no_active_jobs(remote_trainer_id)
            await self._ensure_unique_ssh_remote_port(remote_trainer.model_copy(update=validated.model_dump()))
            saved = await self.repo.update(remote_trainer, data)
        except IntegrityError as error:
            await self.session.rollback()
            raise ResourceAlreadyExistsError(
                "Remote trainer",
                "A trainer with this URL is already configured.",
            ) from error
        try:
            await remote_trainer_tunnel_manager.sync_tunnel(saved, accepted_host_key_fingerprint)
        except Exception:
            await self.repo.update(saved, remote_trainer.model_dump(include={"name", *_CONNECTION_FIELDS}))
            raise
        if remote_trainer.connection_mode is RemoteTrainerConnectionMode.SSH:
            await self._cancel_launch(remote_trainer_id)
            host_changed = (
                remote_trainer.ssh_host_alias != saved.ssh_host_alias
                or remote_trainer.ssh_connection != saved.ssh_connection
            )
            if (
                host_changed
                or remote_trainer.ssh_remote_port != saved.ssh_remote_port
                or saved.connection_mode is RemoteTrainerConnectionMode.DIRECT
            ):
                await persistent_trainer.stop(
                    remote_trainer,
                    remove_volume=host_changed or saved.connection_mode is RemoteTrainerConnectionMode.DIRECT,
                )
        self._start_persistent_trainer_in_background(saved, accepted_host_key_fingerprint)
        return saved

    async def delete_remote_trainer(self, remote_trainer_id: UUID) -> None:
        """Delete a trainer unless its managed container has queued or running jobs."""
        remote_trainer = await self.repo.get_by_id(remote_trainer_id)
        if remote_trainer is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_TRAINER, str(remote_trainer_id))
        if remote_trainer.connection_mode is RemoteTrainerConnectionMode.SSH:
            await self._require_no_active_jobs(remote_trainer_id)
        await self._cancel_launch(remote_trainer_id)
        await persistent_trainer.stop(remote_trainer)
        await remote_trainer_tunnel_manager.stop_tunnel(remote_trainer_id)
        await self.repo.delete_by_id(remote_trainer_id)

    @staticmethod
    def _require_ssh_feature_if_tunneled(connection_mode: RemoteTrainerConnectionMode) -> None:
        """Fail closed if a caller is saving tunnel config the feature can't currently honor."""
        if connection_mode is RemoteTrainerConnectionMode.DIRECT:
            return
        availability = get_ssh_feature_availability()
        if not availability.active:
            raise SshFeatureDisabledError(availability.reason)
