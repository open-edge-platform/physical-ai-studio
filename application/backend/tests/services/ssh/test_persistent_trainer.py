"""Tests for the persistent SSH-tunneled trainer container launch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from schemas.hardware import DeviceType
from schemas.remote_trainer import RemoteTrainer, RemoteTrainerConnectionMode
from services.ssh import persistent_trainer
from services.ssh.docker_ops import ContainerInspection, ResolvedImage

MODULE = "services.ssh.persistent_trainer"

_IMAGE = ResolvedImage(
    tag_reference="ghcr.io/x/physicalai-trainer-cuda:protocol-1",
    digest_reference="ghcr.io/x/physicalai-trainer-cuda@sha256:" + "a" * 64,
    digest="sha256:" + "a" * 64,
    library_version="1.0.0",
)


@pytest.fixture(autouse=True)
def _mock_signature_verification():
    with patch(f"{MODULE}.verify_image_signature", new_callable=AsyncMock) as verify:
        yield verify


def _ssh_trainer() -> RemoteTrainer:
    return RemoteTrainer(
        id=uuid4(),
        name="gpu-trainer",
        url="http://127.0.0.1:8001",
        connection_mode=RemoteTrainerConnectionMode.SSH,
        ssh_host_alias="gpu-box",
        ssh_remote_port=8001,
        ssh_local_port=8001,
    )


def _fake_transport_cm() -> MagicMock:
    transport = AsyncMock()
    transport.run_command = AsyncMock(return_value=MagicMock(ok=True))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=transport)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


async def test_get_launch_phase_is_none_before_and_after_a_successful_start() -> None:
    trainer = _ssh_trainer()
    assert persistent_trainer.get_launch_phase(trainer.id) is None

    with (
        patch(f"{MODULE}.SshTransport", return_value=_fake_transport_cm()),
        patch(f"{MODULE}.get_backend_instance_id", return_value="this-instance"),
        patch(f"{MODULE}.docker_ops") as docker_ops_module,
    ):
        docker_ops_module.inspect_container = AsyncMock(return_value=None)
        docker_ops_module.resolve_protocol_image = AsyncMock(return_value=_IMAGE)
        docker_ops_module.pull_image = AsyncMock()
        docker_ops_module.management_labels = MagicMock(return_value={})
        docker_ops_module.data_volume_name = MagicMock(return_value="vol")
        docker_ops_module.create_data_volume = AsyncMock()
        docker_ops_module.build_run_argv = MagicMock(return_value=["docker", "run", "127.0.0.1::8001", "img"])
        docker_ops_module.remove_stale_containers_on_port = AsyncMock(return_value=[])
        docker_ops_module.launch_container = AsyncMock(return_value="container-id")

        await persistent_trainer.start(trainer)

    assert persistent_trainer.get_launch_phase(trainer.id) is None
    docker_ops_module.launch_container.assert_awaited_once()


async def test_unverified_image_is_never_pulled_or_launched() -> None:
    trainer = _ssh_trainer()
    with (
        patch(f"{MODULE}.SshTransport", return_value=_fake_transport_cm()),
        patch(f"{MODULE}.docker_ops") as docker_ops_module,
        patch(f"{MODULE}.verify_image_signature", new_callable=AsyncMock, side_effect=RuntimeError("unsigned")),
        pytest.raises(RuntimeError, match="unsigned"),
    ):
        docker_ops_module.inspect_container = AsyncMock(return_value=None)
        docker_ops_module.resolve_protocol_image = AsyncMock(return_value=_IMAGE)
        docker_ops_module.pull_image = AsyncMock()
        docker_ops_module.launch_container = AsyncMock()
        await persistent_trainer.start(trainer)

    docker_ops_module.pull_image.assert_not_awaited()
    docker_ops_module.launch_container.assert_not_awaited()


async def test_get_launch_phase_reports_the_current_phase_while_running() -> None:
    """A caller reading `get_launch_phase` mid-launch sees a human phase, not `None`."""
    trainer = _ssh_trainer()
    observed_phase_during_pull: str | None = None

    async def _pull_image(*_args: object, **_kwargs: object) -> None:
        nonlocal observed_phase_during_pull
        observed_phase_during_pull = persistent_trainer.get_launch_phase(trainer.id)

    with (
        patch(f"{MODULE}.SshTransport", return_value=_fake_transport_cm()),
        patch(f"{MODULE}.get_backend_instance_id", return_value="this-instance"),
        patch(f"{MODULE}.docker_ops") as docker_ops_module,
    ):
        docker_ops_module.inspect_container = AsyncMock(return_value=None)
        docker_ops_module.resolve_protocol_image = AsyncMock(return_value=_IMAGE)
        docker_ops_module.pull_image = AsyncMock(side_effect=_pull_image)
        docker_ops_module.management_labels = MagicMock(return_value={})
        docker_ops_module.data_volume_name = MagicMock(return_value="vol")
        docker_ops_module.create_data_volume = AsyncMock()
        docker_ops_module.build_run_argv = MagicMock(return_value=["docker", "run", "127.0.0.1::8001", "img"])
        docker_ops_module.remove_stale_containers_on_port = AsyncMock(return_value=[])
        docker_ops_module.launch_container = AsyncMock(return_value="container-id")

        await persistent_trainer.start(trainer)

    assert observed_phase_during_pull is not None
    assert "pull" in observed_phase_during_pull.lower()


async def test_get_launch_phase_is_cleared_even_when_the_launch_fails() -> None:
    """A failed launch must not leave a trainer stuck reporting 'starting' forever."""
    trainer = _ssh_trainer()

    with (
        patch(f"{MODULE}.SshTransport", return_value=_fake_transport_cm()),
        patch(f"{MODULE}.get_backend_instance_id", return_value="this-instance"),
        patch(f"{MODULE}.docker_ops") as docker_ops_module,
        pytest.raises(RuntimeError),
    ):
        docker_ops_module.inspect_container = AsyncMock(return_value=None)
        docker_ops_module.resolve_protocol_image = AsyncMock(side_effect=RuntimeError("boom"))

        await persistent_trainer.start(trainer)

    assert persistent_trainer.get_launch_phase(trainer.id) is None


async def test_start_records_missing_docker_for_the_health_ui() -> None:
    trainer = _ssh_trainer()
    transport = _fake_transport_cm()
    transport.__aenter__.return_value.run_command = AsyncMock(return_value=MagicMock(ok=False))

    with patch(f"{MODULE}.SshTransport", return_value=transport), pytest.raises(ValueError, match="Docker"):
        await persistent_trainer.start(trainer)

    assert persistent_trainer.get_launch_failure(trainer.id) == "docker_unavailable"


async def test_start_records_missing_accelerator_for_the_health_ui() -> None:
    trainer = _ssh_trainer()
    transport = _fake_transport_cm()
    transport.__aenter__.return_value.run_command = AsyncMock(return_value=MagicMock(ok=False))
    transport.__aenter__.return_value.run_command.side_effect = [
        MagicMock(ok=True),
        MagicMock(ok=False),
        MagicMock(ok=False),
    ]

    with (
        patch(f"{MODULE}.SshTransport", return_value=transport),
        patch(f"{MODULE}.docker_ops.inspect_container", new=AsyncMock(return_value=None)),
        pytest.raises(ValueError, match="accelerator"),
    ):
        await persistent_trainer.start(trainer)

    assert persistent_trainer.get_launch_failure(trainer.id) == "accelerator_unavailable"


async def test_start_is_a_noop_for_a_direct_url_trainer() -> None:
    trainer = RemoteTrainer(id=uuid4(), name="direct", url="https://trainer.test")

    with patch(f"{MODULE}.SshTransport") as ssh_transport:
        await persistent_trainer.start(trainer)

    ssh_transport.assert_not_called()
    assert persistent_trainer.get_launch_phase(trainer.id) is None


async def test_start_skips_launch_when_the_container_is_running() -> None:
    trainer = _ssh_trainer()

    with (
        patch(f"{MODULE}.SshTransport", return_value=_fake_transport_cm()),
        patch(f"{MODULE}.docker_ops") as docker_ops_module,
    ):
        docker_ops_module.inspect_container = AsyncMock(return_value=ContainerInspection(running=True, labels={}))
        docker_ops_module.launch_container = AsyncMock()

        await persistent_trainer.start(trainer)

    docker_ops_module.launch_container.assert_not_awaited()
    assert persistent_trainer.get_launch_phase(trainer.id) is None


async def test_start_recreates_a_stopped_container() -> None:
    trainer = _ssh_trainer()

    with (
        patch(f"{MODULE}.SshTransport", return_value=_fake_transport_cm()),
        patch(f"{MODULE}.get_backend_instance_id", return_value="this-instance"),
        patch(f"{MODULE}.docker_ops") as docker_ops_module,
    ):
        docker_ops_module.inspect_container = AsyncMock(return_value=ContainerInspection(running=False, labels={}))
        docker_ops_module.stop_and_remove_container = AsyncMock()
        docker_ops_module.resolve_protocol_image = AsyncMock(return_value=_IMAGE)
        docker_ops_module.pull_image = AsyncMock()
        docker_ops_module.management_labels = MagicMock(return_value={})
        docker_ops_module.data_volume_name = MagicMock(return_value="vol")
        docker_ops_module.create_data_volume = AsyncMock()
        docker_ops_module.build_run_argv = MagicMock(return_value=["docker", "run", "127.0.0.1::8001", "img"])
        docker_ops_module.remove_stale_containers_on_port = AsyncMock(return_value=[])
        docker_ops_module.launch_container = AsyncMock(return_value="container-id")

        await persistent_trainer.start(trainer)

    docker_ops_module.stop_and_remove_container.assert_awaited_once()
    docker_ops_module.launch_container.assert_awaited_once()


async def test_device_type_defaults_to_cuda_when_nvidia_smi_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _ssh_trainer()
    seen_device_types: list[DeviceType] = []

    async def _resolve_protocol_image(_transport, device_type, *_args, **_kwargs):
        seen_device_types.append(device_type)
        return _IMAGE

    with (
        patch(f"{MODULE}.SshTransport", return_value=_fake_transport_cm()),
        patch(f"{MODULE}.get_backend_instance_id", return_value="this-instance"),
        patch(f"{MODULE}.docker_ops") as docker_ops_module,
    ):
        docker_ops_module.inspect_container = AsyncMock(return_value=None)
        docker_ops_module.resolve_protocol_image = AsyncMock(side_effect=_resolve_protocol_image)
        docker_ops_module.pull_image = AsyncMock()
        docker_ops_module.management_labels = MagicMock(return_value={})
        docker_ops_module.data_volume_name = MagicMock(return_value="vol")
        docker_ops_module.create_data_volume = AsyncMock()
        docker_ops_module.build_run_argv = MagicMock(return_value=["docker", "run", "127.0.0.1::8001", "img"])
        docker_ops_module.remove_stale_containers_on_port = AsyncMock(return_value=[])
        docker_ops_module.launch_container = AsyncMock(return_value="container-id")

        await persistent_trainer.start(trainer)

    assert seen_device_types == [DeviceType.CUDA]


async def test_is_within_startup_grace_period_true_while_actively_launching() -> None:
    trainer = _ssh_trainer()
    observed_during_pull: bool | None = None

    async def _observe(*_args: object, **_kwargs: object) -> ResolvedImage:
        nonlocal observed_during_pull
        observed_during_pull = persistent_trainer.is_within_startup_grace_period(trainer.id)
        return _IMAGE

    with (
        patch(f"{MODULE}.SshTransport", return_value=_fake_transport_cm()),
        patch(f"{MODULE}.get_backend_instance_id", return_value="this-instance"),
        patch(f"{MODULE}.docker_ops") as docker_ops_module,
    ):
        docker_ops_module.inspect_container = AsyncMock(return_value=None)
        docker_ops_module.resolve_protocol_image = AsyncMock(side_effect=_observe)
        docker_ops_module.pull_image = AsyncMock()
        docker_ops_module.management_labels = MagicMock(return_value={})
        docker_ops_module.data_volume_name = MagicMock(return_value="vol")
        docker_ops_module.create_data_volume = AsyncMock()
        docker_ops_module.build_run_argv = MagicMock(return_value=["docker", "run", "127.0.0.1::8001", "img"])
        docker_ops_module.remove_stale_containers_on_port = AsyncMock(return_value=[])
        docker_ops_module.launch_container = AsyncMock(return_value="container-id")

        await persistent_trainer.start(trainer)

    assert observed_during_pull is True


async def test_is_within_startup_grace_period_true_shortly_after_a_failed_launch() -> None:
    """A launch that already failed still reads as 'starting' for the grace window."""
    trainer = _ssh_trainer()
    assert persistent_trainer.is_within_startup_grace_period(trainer.id) is False

    with (
        patch(f"{MODULE}.SshTransport", return_value=_fake_transport_cm()),
        patch(f"{MODULE}.get_backend_instance_id", return_value="this-instance"),
        patch(f"{MODULE}.docker_ops") as docker_ops_module,
        pytest.raises(RuntimeError),
    ):
        docker_ops_module.inspect_container = AsyncMock(return_value=None)
        docker_ops_module.resolve_protocol_image = AsyncMock(side_effect=RuntimeError("boom"))
        await persistent_trainer.start(trainer)

    # The phase itself clears immediately (no longer *actively* launching)...
    assert persistent_trainer.get_launch_phase(trainer.id) is None
    # ...but the grace period from when the attempt began still applies.
    assert persistent_trainer.is_within_startup_grace_period(trainer.id) is True


async def test_is_within_startup_grace_period_false_once_never_launched() -> None:
    trainer = _ssh_trainer()
    assert persistent_trainer.is_within_startup_grace_period(trainer.id) is False


async def test_mark_reachable_ends_the_grace_period() -> None:
    trainer = _ssh_trainer()

    with (
        patch(f"{MODULE}.SshTransport", return_value=_fake_transport_cm()),
        patch(f"{MODULE}.get_backend_instance_id", return_value="this-instance"),
        patch(f"{MODULE}.docker_ops") as docker_ops_module,
    ):
        docker_ops_module.inspect_container = AsyncMock(return_value=None)
        docker_ops_module.resolve_protocol_image = AsyncMock(return_value=_IMAGE)
        docker_ops_module.pull_image = AsyncMock()
        docker_ops_module.management_labels = MagicMock(return_value={})
        docker_ops_module.data_volume_name = MagicMock(return_value="vol")
        docker_ops_module.create_data_volume = AsyncMock()
        docker_ops_module.build_run_argv = MagicMock(return_value=["docker", "run", "127.0.0.1::8001", "img"])
        docker_ops_module.remove_stale_containers_on_port = AsyncMock(return_value=[])
        docker_ops_module.launch_container = AsyncMock(return_value="container-id")

        await persistent_trainer.start(trainer)

    assert persistent_trainer.is_within_startup_grace_period(trainer.id) is True

    persistent_trainer.mark_reachable(trainer.id)

    assert persistent_trainer.is_within_startup_grace_period(trainer.id) is False
