# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Provision, reattach to, and tear down per-job SSH trainer containers.

This is the top-level orchestrator for SSH-provisioned remote training: it
resolves and verifies the trainer image, waits out a busy GPU, re-checks disk
against the job's real snapshot size, launches a least-privilege container,
opens the SSH tunnel to it, verifies readiness, and persists enough state
(:mod:`schemas.job_provisioning`) that a crashed studio can reattach instead of
losing track of a still-running container.

Wiring this into job dispatch (``get_training_backend`` / ``TrainingWorker``)
is a later step; this module is usable standalone against a
:class:`~repositories.job_provisioning_repo.JobProvisioningRepository` and a
:class:`~schemas.remote_server.RemoteServer`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import TYPE_CHECKING, Self

import httpx
from loguru import logger

from core.backend_instance import get_backend_instance_id
from exceptions import (
    SshConnectionError,
    SshHostAliasNotFoundError,
    SshHostKeyMismatchError,
    SshHostKeyUnknownError,
    TrainerContainerLaunchError,
    TrainerLibraryVersionMismatchError,
    TrainerProtocolVersionMismatchError,
    TrainerReadinessTimeoutError,
)
from schemas.hardware import DeviceType
from schemas.job_provisioning import JobProvisioning, JobProvisioningUpdate
from services.ssh import docker_ops
from services.ssh.docker_ops import LibraryVersionCheck, ResolvedImage
from services.ssh.transport import SshTransport, open_transport
from services.ssh.tunnel import SshTunnel
from services.training_backends.phase import PhaseKey
from settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import TracebackType
    from uuid import UUID

    from repositories.job_provisioning_repo import JobProvisioningRepository
    from schemas.remote_server import RemoteServer
    from settings import Settings

# Port the trainer image listens on inside the container. This launch path
# never sets `TRAINER_PORT`, so this must match `TrainerSettings.port`'s
# default and the Dockerfile's `EXPOSE`.
TRAINER_CONTAINER_PORT = 8001


class ReattachFailureReason(StrEnum):
    """Why `verify_reattach` could not confirm a persisted job's container.

    Each member drives a distinct recovery outcome (see
    `services.ssh.recovery`): some are safe to tear down because ownership was
    already established, some must never be touched because it wasn't, and some
    are transient and should simply be retried on the next normal pickup.
    """

    # The container is gone entirely. Nothing to tear down.
    CONTAINER_GONE = "container_gone"
    # Ownership (backend_instance_id + job id labels) never matched. This may be
    # another Studio installation's container - never touch it.
    OWNERSHIP_MISMATCH = "ownership_mismatch"
    # Ownership matched, but the running image disagrees with the persisted
    # digest. Ownership is established, so this one is safe to tear down.
    DIGEST_MISMATCH = "digest_mismatch"
    # Ownership and digest matched, but `/health` never answered in time. May
    # simply be slow to start; leave it for the next attempt.
    HEALTH_NEVER_READY = "health_never_ready"
    # The SSH host key could not be verified. Fails closed: never tear down a
    # container it was not possible to authenticate the host for.
    HOST_KEY_FAILURE = "host_key_failure"
    # The alias is no longer in the user's SSH config. Cannot connect at all.
    ALIAS_MISSING = "alias_missing"
    # Connected before, but the host or the tunnel could not be reached this
    # time. Transient; leave it for the next attempt.
    PORT_UNREACHABLE = "port_unreachable"
    # `docker inspect` itself failed for a reason other than the container
    # being absent (daemon unavailable, permission denied, ...). Ownership
    # could not be checked at all; transient, leave it for the next attempt.
    INSPECTION_FAILED = "inspection_failed"


@dataclass(frozen=True, slots=True)
class ReattachVerification:
    """Outcome of confirming a persisted job's container is still trustworthy.

    `verify_reattach` only classifies; it never tears a container down itself,
    so a caller that conflates "could not verify" with "verified and is wrong"
    can never destroy a container it simply failed to reach.
    """

    ok: bool
    reason: ReattachFailureReason | None = None
    detail: str | None = None
    # True once ownership (backend_instance_id + job id labels) was positively
    # confirmed, so a caller knows a subsequent teardown is provably this
    # installation's own container and not a guess.
    safe_to_teardown: bool = False


@dataclass(frozen=True)
class ProvisionedTrainer:
    """A running, tunnel-reachable trainer, and how to tear it down."""

    base_url: str
    container_name: str
    image: ResolvedImage
    library_version_check: LibraryVersionCheck
    _tunnel: SshTunnel
    _server_alias: str
    _stop_timeout_s: int
    _data_volume: str
    _settings: Settings

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.teardown()

    async def teardown(self) -> None:
        """Tear down the tunnel, then stop/remove the container and its data volume.

        Best-effort: a teardown must never itself raise and mask whatever
        caused it, so any failure here is logged, not propagated.
        """
        await self._tunnel.close()
        try:
            async with SshTransport(self._server_alias, self._settings) as transport:
                await docker_ops.stop_and_remove_container(transport, self.container_name, self._stop_timeout_s)
                await docker_ops.remove_volume(transport, self._data_volume)
        except Exception as error:
            logger.warning("Failed to tear down trainer container '{}': {}", self.container_name, error)


class SshProvisioningService:
    """Provisions, reattaches to, and sweeps SSH-provisioned trainer containers."""

    def __init__(
        self,
        repository: JobProvisioningRepository,
        settings: Settings | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings or get_settings()

    # PLR0913/PLR0915: one keyword-only parameter per tunable/callback and one
    # statement per orchestration step; splitting either up would obscure the
    # single, carefully-ordered happy path this method exists to document (see
    # the module and class docstrings).
    async def provision(  # noqa: PLR0913, PLR0915
        self,
        job_id: UUID,
        server: RemoteServer,
        *,
        protocol_version: int,
        snapshot_size_bytes: int,
        min_library_version: str | None = None,
        library_version_policy_name: str = "default",
        on_gpu_wait: Callable[[float], Awaitable[None]] | None = None,
        on_phase: Callable[[PhaseKey], None] | None = None,
    ) -> ProvisionedTrainer:
        """Provision a fresh trainer container for one job.

        On any failure, whatever was already started is torn down before the
        exception propagates - a caller never has to clean up a partially
        provisioned job itself.

        Args:
            job_id: The training job this container is provisioned for.
            server: The SSH-provisioned server to launch on.
            protocol_version: Studio's own compiled-in trainer protocol version.
            snapshot_size_bytes: The job's actual dataset snapshot size, for the
                pre-launch disk re-check.
            min_library_version: Minimum `physicalai-train` version policy;
                defaults to `settings.ssh_min_library_version`.
            library_version_policy_name: Name of the policy, for a stricter,
                fatal-below-minimum check (e.g. a specific model family).
            on_gpu_wait: Awaited while waiting out a busy GPU, so a caller can
                report a `waiting` phase state.
            on_phase: Called synchronously as provisioning enters each new
                stage (image verification, image pull, trainer start), so a
                caller can drive the SSH phase stepper. `connect` is reported
                by the caller before this method is invoked at all, since the
                job provisioning row is already persisted by then.

        Returns:
            A running, tunnel-reachable trainer.
        """
        settings = self._settings
        minimum_version = min_library_version or settings.ssh_min_library_version
        backend_instance_id = get_backend_instance_id()
        name = docker_ops.container_name(str(job_id))
        data_volume = docker_ops.data_volume_name(str(job_id))

        def _phase(key: PhaseKey) -> None:
            if on_phase is not None:
                on_phase(key)

        await self._repository.save(
            JobProvisioning(
                job_id=job_id,
                remote_server_id=server.id,
                ssh_host_alias=server.ssh_host_alias,
                container_name=name,
                backend_instance_id=backend_instance_id,
            )
        )

        tunnel: SshTunnel | None = None
        launched = False
        volume_created = False
        try:
            async with SshTransport(server.ssh_host_alias, settings) as transport:
                image = await docker_ops.resolve_protocol_image(
                    transport, server.device_type, protocol_version, settings
                )
                library_check = docker_ops.check_library_version(
                    image, minimum_version=minimum_version, policy_name=library_version_policy_name
                )
                if library_check.warning:
                    logger.warning(library_check.warning)
                _phase(PhaseKey.IMAGE_VERIFY)
                await docker_ops.verify_image_signature(transport, image, settings)
                await docker_ops.check_disk_for_job(transport, snapshot_size_bytes, server.name)
                _phase(PhaseKey.IMAGE_PULL)
                await docker_ops.pull_image(transport, image, settings)

                await self._repository.update_by_job_id(
                    job_id, JobProvisioningUpdate(image_ref=image.tag_reference, image_digest=image.digest)
                )

            await docker_ops.wait_for_gpu_free(
                lambda: open_transport(server.ssh_host_alias, settings),
                server.device_type,
                settings,
                server.name,
                on_wait=on_gpu_wait,
            )

            _phase(PhaseKey.TRAINER_START)
            async with SshTransport(server.ssh_host_alias, settings) as transport:
                render_gid = (
                    None
                    if server.device_type is DeviceType.CUDA
                    else await docker_ops.resolve_render_group_gid(transport)
                )
                labels = docker_ops.management_labels(
                    job_id=str(job_id),
                    server_id=str(server.id),
                    backend_instance_id=backend_instance_id,
                    image_digest=image.digest,
                )
                await docker_ops.create_data_volume(transport, data_volume, labels, server.name)
                volume_created = True
                argv = docker_ops.build_run_argv(
                    image_digest_ref=image.digest_reference,
                    device_type=server.device_type,
                    name=name,
                    labels=labels,
                    data_volume=data_volume,
                    remote_container_port=TRAINER_CONTAINER_PORT,
                    stop_timeout_s=settings.ssh_container_stop_timeout_s,
                    render_gid=render_gid,
                )
                container_id = await docker_ops.launch_container(transport, argv, server.name)
                launched = True

                published_port = await self._resolve_published_port(transport, name, server.name)

            tunnel = SshTunnel(
                lambda: open_transport(server.ssh_host_alias, settings),
                "127.0.0.1",
                published_port,
                settings,
            )
            await tunnel.open()
            base_url = f"http://127.0.0.1:{tunnel.local_port}"

            await self._repository.update_by_job_id(
                job_id,
                JobProvisioningUpdate(
                    container_id=container_id,
                    container_name=name,
                    remote_port=published_port,
                    local_tunnel_port=tunnel.local_port,
                    backend_instance_id=backend_instance_id,
                ),
            )

            health = await self._await_ready(base_url, server.name)
            reported_protocol = health.get("protocol_version")
            if not isinstance(reported_protocol, int) or reported_protocol != protocol_version:
                raise TrainerProtocolVersionMismatchError(
                    server.name,
                    protocol_version,
                    reported_protocol if isinstance(reported_protocol, int) else None,
                )
            health_library_version = health.get("library_version")
            label_version = library_check.reported_version
            if (
                isinstance(label_version, str)
                and isinstance(health_library_version, str)
                and health_library_version != label_version
            ):
                raise TrainerLibraryVersionMismatchError(str(label_version), str(health_library_version))

            await self._repository.update_by_job_id(
                job_id,
                JobProvisioningUpdate(
                    trainer_build_version=health.get("build_revision")
                    if isinstance(health.get("build_revision"), str)
                    else None,
                    trainer_protocol_version=reported_protocol,
                ),
            )

            if tunnel is None:
                raise RuntimeError("Unreachable: tunnel is always assigned before this point")
            return ProvisionedTrainer(
                base_url=base_url,
                container_name=name,
                image=image,
                library_version_check=library_check,
                _tunnel=tunnel,
                _server_alias=server.ssh_host_alias,
                _stop_timeout_s=settings.ssh_container_stop_timeout_s,
                _data_volume=data_volume,
                _settings=settings,
            )
        except BaseException:
            if tunnel is not None:
                await tunnel.close()
            if launched or volume_created:
                try:
                    async with SshTransport(server.ssh_host_alias, settings) as transport:
                        if launched:
                            await docker_ops.stop_and_remove_container(
                                transport, name, settings.ssh_container_stop_timeout_s
                            )
                        if volume_created:
                            await docker_ops.remove_volume(transport, data_volume)
                except Exception as cleanup_error:
                    logger.warning(
                        "Failed to clean up container '{}' after a failed provision: {}", name, cleanup_error
                    )
            raise

    async def _resolve_published_port(self, transport: SshTransport, name: str, server_name: str) -> int:
        """Resolve the real host port docker published for the trainer container.

        `--publish 127.0.0.1::<port>` in `build_run_argv` binds an OS-assigned
        ephemeral host port, not `TRAINER_CONTAINER_PORT` itself - that number
        only names the *container's* internal port. The tunnel must forward to
        the real published port, or it connects to whatever (usually nothing)
        is listening on the container port number on the host itself.

        Raises:
            TrainerContainerLaunchError: The published port could not be
                resolved.
        """
        published_port = await docker_ops.resolve_published_port(transport, name, TRAINER_CONTAINER_PORT)
        if published_port is None:
            raise TrainerContainerLaunchError(
                server_name, "container started but its published port could not be resolved"
            )
        return published_port

    async def _await_ready(self, base_url: str, server_name: str) -> dict:
        """Poll `/health` until it answers or the readiness budget is spent.

        Raises:
            TrainerReadinessTimeoutError: The trainer never answered, or its
                `/health` response reported no protocol version.
        """
        settings = self._settings
        started = monotonic()
        last_error: str | None = None
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            while monotonic() - started < settings.ssh_readiness_timeout_s:
                try:
                    response = await client.get(f"{base_url}/health")
                    response.raise_for_status()
                    payload = response.json()
                except (httpx.HTTPError, ValueError) as error:
                    last_error = str(error)
                else:
                    if isinstance(payload, dict):
                        if payload.get("protocol_version") is None:
                            raise TrainerReadinessTimeoutError(
                                server_name, "the trainer's /health reports no protocol version"
                            )
                        return payload
                    last_error = "malformed /health response"
                await asyncio.sleep(settings.ssh_readiness_poll_interval_s)
        raise TrainerReadinessTimeoutError(server_name, last_error)

    async def reattach(self, job_provisioning: JobProvisioning, server: RemoteServer) -> ProvisionedTrainer | None:
        """Reattach to a still-running container recorded for a job.

        Called on studio startup for jobs left `RUNNING`/`PENDING` by a
        previous process. Returns ``None`` when the container is no longer
        running, so the caller can fall back to failing the job instead of
        reattaching to nothing.
        """
        settings = self._settings
        name = job_provisioning.container_name
        if name is None:
            return None

        async with SshTransport(server.ssh_host_alias, settings) as transport:
            inspect = await transport.run_command(["docker", "inspect", "--format", "{{.State.Running}}", name])
            if not inspect.ok or inspect.first_line().lower() != "true":
                return None

        tunnel = SshTunnel(
            lambda: open_transport(server.ssh_host_alias, settings),
            "127.0.0.1",
            job_provisioning.remote_port or TRAINER_CONTAINER_PORT,
            settings,
        )
        await tunnel.open()
        base_url = f"http://127.0.0.1:{tunnel.local_port}"

        await self._repository.update_by_job_id(
            job_provisioning.job_id, JobProvisioningUpdate(local_tunnel_port=tunnel.local_port)
        )

        return ProvisionedTrainer(
            base_url=base_url,
            container_name=name,
            image=ResolvedImage(
                tag_reference=job_provisioning.image_ref or "",
                digest_reference="",
                digest=job_provisioning.image_digest or "",
                library_version=None,
            ),
            library_version_check=LibraryVersionCheck(
                reported_version=None, minimum_version=settings.ssh_min_library_version
            ),
            _tunnel=tunnel,
            _server_alias=server.ssh_host_alias,
            _stop_timeout_s=settings.ssh_container_stop_timeout_s,
            _data_volume=docker_ops.data_volume_name(str(job_provisioning.job_id)),
            _settings=settings,
        )

    # PLR0911: one return per classified outcome (see `ReattachFailureReason`) -
    # a lookup table would obscure which branch establishes ownership and which
    # never connects at all, and that distinction is what a caller relies on to
    # decide what is safe to clean up.
    async def verify_reattach(  # noqa: PLR0911
        self, job_provisioning: JobProvisioning, server: RemoteServer
    ) -> ReattachVerification:
        """Confirm a persisted job's container is still trustworthy, without keeping a tunnel open.

        Called once at studio startup, before `reattach` opens the tunnel the
        training backend actually trains through - this only classifies
        whether that later reattach should be trusted to proceed. A throwaway
        tunnel is opened just long enough to confirm `/health` answers, then
        closed immediately: leaving a tunnel open here would idle it for
        however long the job takes to be picked back up.

        Never tears a container down itself: every failure branch below is a
        pure classification, so the caller decides what is safe to clean up.
        """
        name = job_provisioning.container_name
        if name is None:
            return ReattachVerification(
                ok=False, reason=ReattachFailureReason.CONTAINER_GONE, detail="no container was ever launched"
            )

        settings = self._settings
        try:
            async with SshTransport(server.ssh_host_alias, settings) as transport:
                inspection = await docker_ops.inspect_container(transport, name)
                if inspection is None or not inspection.running:
                    return ReattachVerification(ok=False, reason=ReattachFailureReason.CONTAINER_GONE)

                backend_instance_id = get_backend_instance_id()
                owner = inspection.labels.get(docker_ops.INSTANCE_LABEL)
                job_label = inspection.labels.get(docker_ops.JOB_LABEL)
                if owner != backend_instance_id or job_label != str(job_provisioning.job_id):
                    return ReattachVerification(
                        ok=False,
                        reason=ReattachFailureReason.OWNERSHIP_MISMATCH,
                        detail="the running container's ownership labels do not match this installation and job",
                    )

                digest_label = inspection.labels.get(docker_ops.IMAGE_DIGEST_LABEL)
                if job_provisioning.image_digest and digest_label != job_provisioning.image_digest:
                    detail = (
                        f"running image digest '{digest_label}' does not match persisted "
                        f"'{job_provisioning.image_digest}'"
                        if digest_label
                        else "running container carries no image digest label to compare against the persisted digest"
                    )
                    return ReattachVerification(
                        ok=False,
                        reason=ReattachFailureReason.DIGEST_MISMATCH,
                        detail=detail,
                        safe_to_teardown=True,
                    )
        except SshHostAliasNotFoundError as error:
            return ReattachVerification(ok=False, reason=ReattachFailureReason.ALIAS_MISSING, detail=str(error))
        except (SshHostKeyUnknownError, SshHostKeyMismatchError) as error:
            return ReattachVerification(ok=False, reason=ReattachFailureReason.HOST_KEY_FAILURE, detail=str(error))
        except SshConnectionError as error:
            return ReattachVerification(ok=False, reason=ReattachFailureReason.PORT_UNREACHABLE, detail=str(error))
        except docker_ops.ContainerInspectionError as error:
            return ReattachVerification(ok=False, reason=ReattachFailureReason.INSPECTION_FAILED, detail=str(error))

        # Ownership and digest are confirmed; open a throwaway tunnel just long
        # enough to confirm `/health` answers, then close it. The real reattach
        # (opened by `SshTrainingBackend` when the job is next picked up) gets
        # its own, longer-lived tunnel.
        tunnel = SshTunnel(
            lambda: open_transport(server.ssh_host_alias, settings),
            "127.0.0.1",
            job_provisioning.remote_port or TRAINER_CONTAINER_PORT,
            settings,
        )
        try:
            await tunnel.open()
            await self._await_ready(f"http://127.0.0.1:{tunnel.local_port}", server.name)
        except TrainerReadinessTimeoutError as error:
            return ReattachVerification(
                ok=False, reason=ReattachFailureReason.HEALTH_NEVER_READY, detail=str(error), safe_to_teardown=True
            )
        except SshConnectionError as error:
            return ReattachVerification(ok=False, reason=ReattachFailureReason.PORT_UNREACHABLE, detail=str(error))
        finally:
            await tunnel.close()

        return ReattachVerification(ok=True, safe_to_teardown=True)

    async def teardown(self, job_id: UUID, server: RemoteServer) -> None:
        """Tear down a job's container by id, without an open tunnel.

        Used for cancellation/cleanup paths that only have the persisted
        provisioning row, not a live `ProvisionedTrainer`.
        """
        job_provisioning = await self._repository.get_by_job_id(job_id)
        if job_provisioning is None or job_provisioning.container_name is None:
            return
        async with SshTransport(server.ssh_host_alias, self._settings) as transport:
            await docker_ops.stop_and_remove_container(
                transport, job_provisioning.container_name, self._settings.ssh_container_stop_timeout_s
            )
            await docker_ops.remove_volume(transport, docker_ops.data_volume_name(str(job_id)))
        await self._repository.delete_by_job_id(job_id)

    async def sweep_orphans(self, server: RemoteServer, active_job_ids: set[UUID]) -> list[str]:
        """Remove containers and data volumes this installation owns but no job claims.

        Requires every management label to match *and* the resource's
        `backend_instance_id` label to match this installation's own id *and*
        its job id to be absent from `active_job_ids` - a job that is still
        `PENDING`/`RUNNING` has an active reattach claim and must never be
        touched, even if this sweep runs concurrently with its own startup
        reattach.

        Returns:
            Names of the containers and volumes removed.
        """
        backend_instance_id = get_backend_instance_id()
        removed: list[str] = []
        async with SshTransport(server.ssh_host_alias, self._settings) as transport:
            active_ids = {str(job_id) for job_id in active_job_ids}
            containers = await docker_ops.list_managed_containers(transport, backend_instance_id)
            for container in containers:
                if container.job_id is not None and container.job_id in active_ids:
                    continue
                await docker_ops.stop_and_remove_container(
                    transport, container.container_id, self._settings.ssh_container_stop_timeout_s
                )
                removed.append(container.name)
            volumes = await docker_ops.list_managed_volumes(transport, backend_instance_id)
            for volume in volumes:
                if volume.job_id is not None and volume.job_id in active_ids:
                    continue
                await docker_ops.remove_volume(transport, volume.name)
                removed.append(volume.name)
        return removed
