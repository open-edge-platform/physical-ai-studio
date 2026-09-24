"""Long-lived trainer containers for SSH-tunneled remote trainers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

from core.backend_instance import get_backend_instance_id
from schemas.hardware import DeviceType
from schemas.remote_trainer import RemoteTrainer
from services.ssh import docker_ops
from services.ssh.connection import AliasTarget, DirectTarget
from services.ssh.docker_ops import verify_image_signature
from services.ssh.trainer_image import DEFAULT_PROTOCOL_VERSION, resolve_render_group_gid
from services.ssh.transport import SshTransport
from settings import get_settings

if TYPE_CHECKING:
    from uuid import UUID

# Human-readable phase a trainer's background launch (see `remote_trainer_service
# ._start_persistent_trainer_in_background`) is currently in, keyed by trainer id.
# Read by `RemoteTrainerService.check_remote_trainer` so a health check made while
# the container is still being connected to/pulled/started reports "starting"
# instead of dialing a trainer URL nothing is listening on yet and reading that
# as "unreachable". Cleared as soon as `start()` returns, one way or the other -
# whether launching itself is still in progress is judged separately, from
# `_launch_started_at` below.
_launch_phase: dict[UUID, str] = {}

# When the most recent launch attempt for a trainer began, kept (not cleared)
# whether that attempt is still running, already succeeded, or already failed.
# A freshly-launched container can take a while to pull its image and start,
# and a few seconds more once running before its own health endpoint answers -
# `is_within_startup_grace_period` uses this so a health probe that is merely
# too early reads as "starting", not "unreachable", without having to still
# find the launch coroutine literally in flight. Cleared once
# `mark_reachable` confirms the trainer is actually answering.
_launch_started_at: dict[UUID, datetime] = {}

# Stable reasons safe to show in the UI when Studio cannot satisfy its own
# managed-container prerequisites.
_launch_failure: dict[UUID, str] = {}

# How long after a launch attempt begins an otherwise-unreachable trainer is
# still given the benefit of the doubt. Generous enough to cover a slow,
# multi-gigabyte first pull; past this, "unreachable" is reported as what it
# actually is - something a user should look into, not silently wait out.
_STARTUP_GRACE_PERIOD: Final = timedelta(minutes=15)


def get_launch_phase(remote_trainer_id: UUID) -> str | None:
    """Return the in-progress launch phase for a trainer, or ``None`` if it isn't launching."""
    return _launch_phase.get(remote_trainer_id)


def get_launch_failure(remote_trainer_id: UUID) -> str | None:
    """Return a managed-container prerequisite failure, if one occurred."""
    return _launch_failure.get(remote_trainer_id)


def is_within_startup_grace_period(remote_trainer_id: UUID) -> bool:
    """True while a launch is running, or recently began and hasn't been confirmed reachable yet."""
    if remote_trainer_id in _launch_phase:
        return True
    started_at = _launch_started_at.get(remote_trainer_id)
    return started_at is not None and datetime.now(UTC) - started_at < _STARTUP_GRACE_PERIOD


def mark_reachable(remote_trainer_id: UUID) -> None:
    """Clear the startup grace window once a health probe confirms the trainer is reachable."""
    _launch_started_at.pop(remote_trainer_id, None)
    _launch_failure.pop(remote_trainer_id, None)


def _container_name(trainer_id: object) -> str:
    return f"physicalai-trainer-{trainer_id}"


def _ssh_target(remote_trainer: RemoteTrainer) -> AliasTarget | DirectTarget:
    if remote_trainer.ssh_host_alias:
        return AliasTarget(remote_trainer.ssh_host_alias)
    connection = remote_trainer.ssh_connection
    if connection is None:
        raise ValueError("SSH trainer has no configured host")
    return DirectTarget(connection.hostname, connection.port, connection.user, connection.identity_file)


async def start(remote_trainer: RemoteTrainer, accepted_host_key_fingerprint: str | None = None) -> None:
    """Ensure an SSH trainer has a running container."""
    if remote_trainer.connection_mode.value != "ssh":
        return
    settings = get_settings()
    target = _ssh_target(remote_trainer)
    remote_port = remote_trainer.ssh_remote_port
    if remote_port is None:
        raise ValueError("SSH trainer has no configured remote port")
    name = _container_name(remote_trainer.id)
    _launch_started_at[remote_trainer.id] = datetime.now(UTC)
    _launch_failure.pop(remote_trainer.id, None)
    _launch_phase[remote_trainer.id] = f"Connecting to '{target.name}'…"
    try:
        async with SshTransport(
            target, settings, accepted_host_key_fingerprint=accepted_host_key_fingerprint
        ) as transport:
            _launch_phase[remote_trainer.id] = "Checking Docker…"
            docker = await transport.run_command(["docker", "version", "--format", "{{.Server.Version}}"])
            if not docker.ok:
                _launch_failure[remote_trainer.id] = "docker_unavailable"
                raise ValueError("Docker is not available on the SSH host")
            existing = await docker_ops.inspect_container(transport, name)
            if existing and existing.running:
                return
            if existing:
                await docker_ops.stop_and_remove_container(transport, name, settings.ssh_container_stop_timeout_s)
            backend_instance_id = get_backend_instance_id()
            _launch_phase[remote_trainer.id] = "Detecting accelerator…"
            cuda = await transport.run_command(["nvidia-smi", "-L"])
            xpu = await transport.run_command(["xpu-smi", "discovery"])

            device = DeviceType.CUDA if cuda.ok else DeviceType.XPU if xpu.ok else None
            if device is None:
                _launch_failure[remote_trainer.id] = "accelerator_unavailable"
                raise ValueError("No supported CUDA or XPU accelerator found on SSH trainer")
            _launch_phase[remote_trainer.id] = "Resolving trainer image…"
            image = await docker_ops.resolve_protocol_image(transport, device, DEFAULT_PROTOCOL_VERSION, settings)
            _launch_phase[remote_trainer.id] = "Verifying trainer image…"
            await verify_image_signature(image, settings)
            _launch_phase[remote_trainer.id] = "Pulling trainer image…"
            await docker_ops.pull_image(transport, image, settings)
            labels = docker_ops.management_labels(
                job_id=str(remote_trainer.id),
                server_id=str(remote_trainer.id),
                backend_instance_id=backend_instance_id,
                image_digest=image.digest,
            )
            volume = docker_ops.data_volume_name(str(remote_trainer.id))
            await docker_ops.create_data_volume(transport, volume, labels, remote_trainer.name)
            argv = docker_ops.build_run_argv(
                image_digest_ref=image.digest_reference,
                device_type=device,
                name=name,
                labels=labels,
                data_volume=volume,
                remote_container_port=8001,
                stop_timeout_s=settings.ssh_container_stop_timeout_s,
                render_gid=(None if device is DeviceType.CUDA else await resolve_render_group_gid(transport)),
                shm_size_gb=settings.ssh_trainer_shm_size_gb,
            )
            # The tunnel needs a stable loopback port for the reusable trainer container.
            argv[argv.index("127.0.0.1::8001")] = f"127.0.0.1:{remote_port}:8001"
            _launch_phase[remote_trainer.id] = "Starting trainer container…"
            # A leftover container from a previous attempt for this trainer (crash,
            # retried save, deleted-then-recreated trainer) can still hold this
            # exact port; `docker run` refuses to bind it a second time. Only ever
            # removes this trainer's own managed containers - see
            # `remove_stale_containers_on_port`.
            await docker_ops.remove_stale_containers_on_port(
                transport, remote_port, backend_instance_id, str(remote_trainer.id)
            )
            await docker_ops.launch_container(transport, argv, remote_trainer.name)
    finally:
        _launch_phase.pop(remote_trainer.id, None)


async def stop(remote_trainer: RemoteTrainer, *, remove_volume: bool = True) -> None:
    """Remove the container, retaining its volume when reconfiguring a trainer."""
    if remote_trainer.connection_mode.value != "ssh":
        return
    settings = get_settings()
    async with SshTransport(_ssh_target(remote_trainer), settings) as transport:
        await docker_ops.stop_and_remove_container(
            transport, _container_name(remote_trainer.id), settings.ssh_container_stop_timeout_s
        )
        if remove_volume:
            await docker_ops.remove_volume(transport, docker_ops.data_volume_name(str(remote_trainer.id)))
    _launch_phase.pop(remote_trainer.id, None)
    _launch_started_at.pop(remote_trainer.id, None)
    _launch_failure.pop(remote_trainer.id, None)
