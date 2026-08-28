# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SSH provisioning orchestrator.

Every remote/docker interaction is faked: `SshTransport` and `SshTunnel` are
monkeypatched with in-memory doubles, and `docker_ops` calls hit a scripted
fake transport rather than a real host. This proves the *orchestration* -
ordering, persistence, and cleanup-on-failure - independent of the individual
docker_ops/transport unit tests.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Self
from uuid import uuid4

import pytest

from exceptions import (
    SshConnectionError,
    SshHostAliasNotFoundError,
    SshHostKeyMismatchError,
    TrainerContainerLaunchError,
    TrainerImageResolutionError,
    TrainerImageVerificationError,
    TrainerProtocolVersionMismatchError,
    TrainerReadinessTimeoutError,
)
from schemas.hardware import DeviceType
from schemas.job_provisioning import JobProvisioning, JobProvisioningUpdate
from schemas.remote_server import RemoteServer
from services.ssh import docker_ops
from services.ssh import provisioning as provisioning_module
from services.ssh.docker_ops import JOB_LABEL, LIBRARY_VERSION_LABEL, MANAGED_LABEL, LibraryVersionCheck, ResolvedImage
from services.ssh.preflight import PROTOCOL_LABEL
from services.ssh.provisioning import ReattachFailureReason, SshProvisioningService
from services.ssh.transport import CommandResult
from services.training_backends.phase import PhaseKey
from settings import Settings

if TYPE_CHECKING:
    from types import TracebackType

_REGISTRY = "ghcr.io/open-edge-platform"
_PROTOCOL_VERSION = 1
_TAG_REF = f"{_REGISTRY}/physicalai-trainer-cuda:protocol-1"
_DIGEST = "sha256:" + "b" * 64


def _ok(stdout: str = "") -> CommandResult:
    return CommandResult(argv=(), command="", exit_status=0, stdout=stdout)


def _fail(stderr: str = "failed") -> CommandResult:
    return CommandResult(argv=(), command="", exit_status=1, stderr=stderr)


def _server() -> RemoteServer:
    return RemoteServer(id=uuid4(), name="Lab GPU box", ssh_host_alias="gpu-box", device_type=DeviceType.CUDA)


def _inspection_script(*, running: bool, labels: dict[str, str]) -> dict[str, CommandResult]:
    """Script `docker inspect --format ...` for the two distinct calls `inspect_container` makes.

    `_healthy_script()`'s single catch-all `"docker inspect"` key would match
    both calls with the same reply, so these tests build the two format-scoped
    keys directly instead.
    """
    return {
        "docker inspect --format {{.State.Running}}": _ok(f"{str(running).lower()}\n"),
        "docker inspect --format {{json .Config.Labels}}": _ok(json.dumps(labels)),
    }


def _healthy_script(container_id: str = "abc123", published_port: int = 54321) -> dict[str, CommandResult]:
    return {
        f"docker buildx imagetools inspect {_TAG_REF} --format {{{{json .Manifest.Digest}}}}": _ok(json.dumps(_DIGEST)),
        f"docker buildx imagetools inspect {_TAG_REF} --format {{{{json .Image.Config.Labels}}}}": _ok(
            json.dumps({PROTOCOL_LABEL: "1", LIBRARY_VERSION_LABEL: "1.0.0"})
        ),
        "cosign version": _ok("v2.4.1"),
        "cosign verify": _ok("Verified OK"),
        "df -B1 -P /var/lib/docker": _ok(
            "Filesystem 1B-blocks Used Available Capacity Mounted on\n"
            "/dev/sda1 900000000000 100000000000 85899345920 12% /var/lib/docker\n"
        ),
        "docker pull": _ok("Status: Downloaded"),
        "nvidia-smi --query-compute-apps": _ok("\n"),  # GPU free
        "docker volume create": _ok(""),
        "docker run": _ok(f"{container_id}\n"),
        "docker port": _ok(f"127.0.0.1:{published_port}\n"),
        "docker stop": _ok(""),
        "docker rm": _ok(""),
        "docker volume rm": _ok(""),
        "docker volume ls": _ok(""),
        "docker inspect": _ok("true\n"),
    }


class FakeSshTransport:
    """Stand-in for `SshTransport` as an async context manager."""

    def __init__(self, script: dict[str, CommandResult], recorder: list[tuple[str, ...]]) -> None:
        self.script = script
        self.recorder = recorder

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        return None

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def run_command(self, argv, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
        self.recorder.append(tuple(argv))
        joined = " ".join(argv)
        for prefix, result in self.script.items():
            if joined.startswith(prefix):
                return result
        return _fail(f"unscripted command: {joined}")


class FakeSshTunnel:
    """Stand-in for `SshTunnel` that never actually opens a socket."""

    instances: list[FakeSshTunnel] = []

    def __init__(self, open_transport, remote_host: str, remote_port: int, settings) -> None:
        self._local_port = 40000 + len(FakeSshTunnel.instances)
        self.closed = False
        FakeSshTunnel.instances.append(self)

    @property
    def local_port(self) -> int:
        return self._local_port

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class FakeProvisioningRepository:
    """In-memory stand-in for `JobProvisioningRepository`."""

    def __init__(self) -> None:
        self._rows: dict[str, JobProvisioning] = {}

    async def save(self, item: JobProvisioning) -> JobProvisioning:
        self._rows[str(item.job_id)] = item
        return item

    async def get_by_job_id(self, job_id) -> JobProvisioning | None:
        return self._rows.get(str(job_id))

    async def update_by_job_id(self, job_id, update: JobProvisioningUpdate) -> JobProvisioning:
        current = self._rows[str(job_id)]
        changes = update.model_dump(exclude_unset=True, exclude_none=True)
        merged = current.model_copy(update=changes)
        self._rows[str(job_id)] = merged
        return merged

    async def delete_by_job_id(self, job_id) -> None:
        self._rows.pop(str(job_id), None)


@pytest.fixture(autouse=True)
def _patch_transport_and_tunnel(monkeypatch):
    """Route every SshTransport()/SshTunnel() construction to fakes."""
    recorder: list[tuple[str, ...]] = []
    script_holder: dict[str, dict[str, CommandResult]] = {"script": {}}

    def fake_transport_ctor(alias, settings=None):
        return FakeSshTransport(script_holder["script"], recorder)

    def fake_open_transport(alias, settings=None):
        return FakeSshTransport(script_holder["script"], recorder)

    monkeypatch.setattr(provisioning_module, "SshTransport", fake_transport_ctor)
    monkeypatch.setattr(provisioning_module, "open_transport", fake_open_transport)
    monkeypatch.setattr(provisioning_module, "SshTunnel", FakeSshTunnel)
    FakeSshTunnel.instances.clear()

    yield recorder, script_holder


def _set_script(fixture, script: dict[str, CommandResult]) -> None:
    _, holder = fixture
    holder["script"] = script


def _commands(fixture) -> list[tuple[str, ...]]:
    recorder, _ = fixture
    return recorder


@pytest.fixture
def repository() -> FakeProvisioningRepository:
    return FakeProvisioningRepository()


async def test_provision_happy_path(_patch_transport_and_tunnel, repository) -> None:
    _set_script(_patch_transport_and_tunnel, _healthy_script())
    service = SshProvisioningService(repository)
    server = _server()
    job_id = uuid4()

    async def fake_ready(base_url, server_name):
        return {"protocol_version": _PROTOCOL_VERSION, "library_version": "1.0.0", "build_revision": "abc"}

    service._await_ready = fake_ready  # type: ignore[method-assign]

    trainer = await service.provision(job_id, server, protocol_version=_PROTOCOL_VERSION, snapshot_size_bytes=1024)

    assert trainer.base_url.startswith("http://127.0.0.1:")
    assert trainer.container_name == docker_ops.container_name(str(job_id))

    persisted = await repository.get_by_job_id(job_id)
    assert persisted.container_id == "abc123"
    assert persisted.image_digest == _DIGEST
    assert persisted.trainer_protocol_version == _PROTOCOL_VERSION

    await trainer.teardown()
    commands = _commands(_patch_transport_and_tunnel)
    assert any(cmd[:2] == ("docker", "stop") for cmd in commands)
    assert any(cmd[:3] == ("docker", "volume", "rm") for cmd in commands)


async def test_provision_fails_with_no_fallback_tag(_patch_transport_and_tunnel, repository) -> None:
    _set_script(_patch_transport_and_tunnel, {})  # every docker call fails
    service = SshProvisioningService(repository)

    with pytest.raises(TrainerImageResolutionError):
        await service.provision(uuid4(), _server(), protocol_version=_PROTOCOL_VERSION, snapshot_size_bytes=1)

    assert not any(cmd[:2] == ("docker", "run") for cmd in _commands(_patch_transport_and_tunnel))


async def test_provision_fails_closed_on_signature_verification(_patch_transport_and_tunnel, repository) -> None:
    script = _healthy_script()
    script["cosign verify"] = _fail("no matching signatures")
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    with pytest.raises(TrainerImageVerificationError):
        await service.provision(uuid4(), _server(), protocol_version=_PROTOCOL_VERSION, snapshot_size_bytes=1)

    # Signature verification fails before any pull.
    assert not any(cmd[:2] == ("docker", "pull") for cmd in _commands(_patch_transport_and_tunnel))


async def test_provision_waits_out_a_busy_gpu(_patch_transport_and_tunnel, repository, monkeypatch) -> None:
    _set_script(_patch_transport_and_tunnel, _healthy_script())
    service = SshProvisioningService(repository)
    waited: list[float] = []

    async def fake_ready(base_url, server_name):
        return {"protocol_version": _PROTOCOL_VERSION, "library_version": "1.0.0"}

    service._await_ready = fake_ready  # type: ignore[method-assign]

    async def patched_wait(open_transport, device_type, settings, server_name, *, on_wait=None, **kwargs):
        if on_wait is not None:
            await on_wait(1.0)

    monkeypatch.setattr(docker_ops, "wait_for_gpu_free", patched_wait)

    async def on_gpu_wait(elapsed: float) -> None:
        waited.append(elapsed)

    await service.provision(
        uuid4(),
        _server(),
        protocol_version=_PROTOCOL_VERSION,
        snapshot_size_bytes=1,
        on_gpu_wait=on_gpu_wait,
    )

    assert waited == [1.0]


async def test_provision_reports_phases_in_verify_pull_start_order(_patch_transport_and_tunnel, repository) -> None:
    """`on_phase` fires image_verify before image_pull before trainer_start,
    matching the order the image is actually verified then pulled (never
    pulling an unverified image) then the container is launched."""
    _set_script(_patch_transport_and_tunnel, _healthy_script())
    service = SshProvisioningService(repository)
    reported: list[PhaseKey] = []

    async def fake_ready(base_url, server_name):
        return {"protocol_version": _PROTOCOL_VERSION, "library_version": "1.0.0"}

    service._await_ready = fake_ready  # type: ignore[method-assign]

    await service.provision(
        uuid4(),
        _server(),
        protocol_version=_PROTOCOL_VERSION,
        snapshot_size_bytes=1,
        on_phase=reported.append,
    )

    assert reported == [PhaseKey.IMAGE_VERIFY, PhaseKey.IMAGE_PULL, PhaseKey.TRAINER_START]


async def test_provision_cleans_up_container_on_protocol_mismatch(_patch_transport_and_tunnel, repository) -> None:
    _set_script(_patch_transport_and_tunnel, _healthy_script())
    service = SshProvisioningService(repository)

    async def mismatched_ready(base_url, server_name):
        return {"protocol_version": 99, "library_version": "1.0.0"}

    service._await_ready = mismatched_ready  # type: ignore[method-assign]

    with pytest.raises(TrainerProtocolVersionMismatchError):
        await service.provision(uuid4(), _server(), protocol_version=_PROTOCOL_VERSION, snapshot_size_bytes=1)

    assert any(cmd[:2] == ("docker", "stop") for cmd in _commands(_patch_transport_and_tunnel))
    assert FakeSshTunnel.instances[-1].closed


async def test_provision_cleans_up_on_readiness_timeout(_patch_transport_and_tunnel, repository) -> None:
    _set_script(_patch_transport_and_tunnel, _healthy_script())
    service = SshProvisioningService(repository)

    async def never_ready(base_url, server_name):
        raise TrainerReadinessTimeoutError(server_name, "no response")

    service._await_ready = never_ready  # type: ignore[method-assign]

    with pytest.raises(TrainerReadinessTimeoutError):
        await service.provision(uuid4(), _server(), protocol_version=_PROTOCOL_VERSION, snapshot_size_bytes=1)

    assert any(cmd[:2] == ("docker", "stop") for cmd in _commands(_patch_transport_and_tunnel))


async def test_provision_launch_failure_needs_no_cleanup(_patch_transport_and_tunnel, repository) -> None:
    script = _healthy_script()
    script["docker run"] = _fail("port already allocated")
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    with pytest.raises(TrainerContainerLaunchError):
        await service.provision(uuid4(), _server(), protocol_version=_PROTOCOL_VERSION, snapshot_size_bytes=1)

    # docker run never produced a container id, so there is nothing to stop,
    # but the already-created data volume is still removed.
    commands = _commands(_patch_transport_and_tunnel)
    assert not any(cmd[:2] == ("docker", "stop") for cmd in commands)
    assert any(cmd[:3] == ("docker", "volume", "rm") for cmd in commands)


async def test_reattach_returns_none_when_container_not_running(_patch_transport_and_tunnel, repository) -> None:
    script = _healthy_script()
    script["docker inspect"] = _ok("false\n")
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)
    server = _server()
    job_id = uuid4()
    provisioning_row = JobProvisioning(
        job_id=job_id,
        remote_server_id=server.id,
        ssh_host_alias=server.ssh_host_alias,
        container_name=docker_ops.container_name(str(job_id)),
        remote_port=8080,
    )
    await repository.save(provisioning_row)

    result = await service.reattach(provisioning_row, server)

    assert result is None


async def test_reattach_reopens_tunnel_when_container_running(_patch_transport_and_tunnel, repository) -> None:
    _set_script(_patch_transport_and_tunnel, _healthy_script())
    service = SshProvisioningService(repository)
    server = _server()
    job_id = uuid4()
    provisioning_row = JobProvisioning(
        job_id=job_id,
        remote_server_id=server.id,
        ssh_host_alias=server.ssh_host_alias,
        container_name=docker_ops.container_name(str(job_id)),
        remote_port=8080,
    )
    await repository.save(provisioning_row)

    result = await service.reattach(provisioning_row, server)

    assert result is not None
    assert result.base_url.startswith("http://127.0.0.1:")


async def test_sweep_orphans_never_touches_active_or_foreign_containers(
    _patch_transport_and_tunnel, repository
) -> None:
    server = _server()
    active_job_id = uuid4()
    orphan_job_id = uuid4()
    ps_lines = "\n".join(
        json.dumps(
            {
                "ID": f"container-{job_id}",
                "Names": docker_ops.container_name(str(job_id)),
                "Labels": f"{MANAGED_LABEL}=true,{JOB_LABEL}={job_id}",
            }
        )
        for job_id in (active_job_id, orphan_job_id)
    )
    script = _healthy_script()
    script["docker ps"] = _ok(ps_lines + "\n")
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    removed = await service.sweep_orphans(server, active_job_ids={active_job_id})

    assert removed == [docker_ops.container_name(str(orphan_job_id))]


async def test_sweep_orphans_removes_orphaned_volumes_but_not_active(_patch_transport_and_tunnel, repository) -> None:
    server = _server()
    active_job_id = uuid4()
    orphan_job_id = uuid4()
    script = _healthy_script()
    script["docker ps"] = _ok("")  # no orphaned containers this sweep
    volume_lines = "\n".join(
        json.dumps(
            {
                "Name": docker_ops.data_volume_name(str(job_id)),
                "Labels": f"{MANAGED_LABEL}=true,{JOB_LABEL}={job_id}",
            }
        )
        for job_id in (active_job_id, orphan_job_id)
    )
    script["docker volume ls"] = _ok(volume_lines + "\n")
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    removed = await service.sweep_orphans(server, active_job_ids={active_job_id})

    assert removed == [docker_ops.data_volume_name(str(orphan_job_id))]


async def test_provisioned_trainer_teardown_uses_provisioning_settings(monkeypatch) -> None:
    """Teardown must reconnect with the same settings used to provision, not `get_settings()`."""
    settings = Settings()
    captured: dict[str, object] = {}

    class _Transport:
        def __init__(self, alias, settings=None):
            captured["alias"] = alias
            captured["settings"] = settings

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def run_command(self, argv, timeout=None):  # noqa: ASYNC109
            return _ok("")

    class _Tunnel:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(provisioning_module, "SshTransport", _Transport)
    trainer = provisioning_module.ProvisionedTrainer(
        base_url="http://127.0.0.1:1",
        container_name="physicalai-trainer-job1",
        image=ResolvedImage(tag_reference="t", digest_reference="d", digest="sha256:" + "a" * 64, library_version=None),
        library_version_check=LibraryVersionCheck(reported_version=None, minimum_version="0.1.0"),
        _tunnel=_Tunnel(),
        _server_alias="gpu-box",
        _stop_timeout_s=30,
        _data_volume="physicalai-trainer-data-job1",
        _settings=settings,
    )

    await trainer.teardown()

    assert captured["alias"] == "gpu-box"
    assert captured["settings"] is settings


# --------------------------------------------------------------------------- #
# verify_reattach                                                             #
# --------------------------------------------------------------------------- #


def _provisioning_row(server: RemoteServer, job_id, *, image_digest: str | None = _DIGEST) -> JobProvisioning:
    return JobProvisioning(
        job_id=job_id,
        remote_server_id=server.id,
        ssh_host_alias=server.ssh_host_alias,
        container_name=docker_ops.container_name(str(job_id)),
        remote_port=8080,
        image_digest=image_digest,
    )


async def test_verify_reattach_confirms_a_healthy_owned_container(
    _patch_transport_and_tunnel, repository, monkeypatch
) -> None:
    monkeypatch.setattr(provisioning_module, "get_backend_instance_id", lambda: "this-instance")
    server = _server()
    job_id = uuid4()
    row = _provisioning_row(server, job_id)
    await repository.save(row)
    script = _healthy_script()
    del script["docker inspect"]
    script.update(
        _inspection_script(
            running=True,
            labels={
                docker_ops.INSTANCE_LABEL: "this-instance",
                docker_ops.JOB_LABEL: str(job_id),
                docker_ops.IMAGE_DIGEST_LABEL: _DIGEST,
            },
        )
    )
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    async def fake_ready(base_url, server_name):
        return {"protocol_version": _PROTOCOL_VERSION}

    service._await_ready = fake_ready  # type: ignore[method-assign]

    result = await service.verify_reattach(row, server)

    assert result.ok
    assert result.reason is None
    assert result.safe_to_teardown


async def test_verify_reattach_reports_container_gone_when_no_container_was_ever_launched(
    _patch_transport_and_tunnel, repository
) -> None:
    server = _server()
    job_id = uuid4()
    row = JobProvisioning(
        job_id=job_id, remote_server_id=server.id, ssh_host_alias=server.ssh_host_alias, container_name=None
    )
    await repository.save(row)
    service = SshProvisioningService(repository)

    result = await service.verify_reattach(row, server)

    assert not result.ok
    assert result.reason is provisioning_module.ReattachFailureReason.CONTAINER_GONE
    assert not result.safe_to_teardown


async def test_verify_reattach_reports_container_gone_when_not_running(_patch_transport_and_tunnel, repository) -> None:
    server = _server()
    job_id = uuid4()
    row = _provisioning_row(server, job_id)
    await repository.save(row)
    script = _healthy_script()
    del script["docker inspect"]
    script.update(_inspection_script(running=False, labels={}))
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    result = await service.verify_reattach(row, server)

    assert not result.ok
    assert result.reason is ReattachFailureReason.CONTAINER_GONE
    assert not result.safe_to_teardown


async def test_verify_reattach_reports_ownership_mismatch_without_teardown(
    _patch_transport_and_tunnel, repository, monkeypatch
) -> None:
    monkeypatch.setattr(provisioning_module, "get_backend_instance_id", lambda: "this-instance")
    server = _server()
    job_id = uuid4()
    row = _provisioning_row(server, job_id)
    await repository.save(row)
    script = _healthy_script()
    del script["docker inspect"]
    script.update(
        _inspection_script(
            running=True,
            labels={docker_ops.INSTANCE_LABEL: "some-other-instance", docker_ops.JOB_LABEL: str(job_id)},
        )
    )
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    result = await service.verify_reattach(row, server)

    assert not result.ok
    assert result.reason is ReattachFailureReason.OWNERSHIP_MISMATCH
    # Ownership was never established, so a caller must never tear this down -
    # it may belong to another Studio installation.
    assert not result.safe_to_teardown


async def test_verify_reattach_reports_digest_mismatch_and_allows_teardown(
    _patch_transport_and_tunnel, repository, monkeypatch
) -> None:
    monkeypatch.setattr(provisioning_module, "get_backend_instance_id", lambda: "this-instance")
    server = _server()
    job_id = uuid4()
    row = _provisioning_row(server, job_id, image_digest=_DIGEST)
    await repository.save(row)
    script = _healthy_script()
    del script["docker inspect"]
    script.update(
        _inspection_script(
            running=True,
            labels={
                docker_ops.INSTANCE_LABEL: "this-instance",
                docker_ops.JOB_LABEL: str(job_id),
                docker_ops.IMAGE_DIGEST_LABEL: "sha256:" + "f" * 64,
            },
        )
    )
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    result = await service.verify_reattach(row, server)

    assert not result.ok
    assert result.reason is ReattachFailureReason.DIGEST_MISMATCH
    # Ownership was confirmed, so this one is provably ours to reclaim.
    assert result.safe_to_teardown


async def test_verify_reattach_reports_digest_mismatch_when_running_container_has_no_digest_label(
    _patch_transport_and_tunnel, repository, monkeypatch
) -> None:
    """A persisted digest with no running label to compare against fails closed.

    A container launched by an older Studio build (before `IMAGE_DIGEST_LABEL`
    existed) must never be silently treated as a confirmed match just because
    there is nothing to compare against.
    """
    monkeypatch.setattr(provisioning_module, "get_backend_instance_id", lambda: "this-instance")
    server = _server()
    job_id = uuid4()
    row = _provisioning_row(server, job_id, image_digest=_DIGEST)
    await repository.save(row)
    script = _healthy_script()
    del script["docker inspect"]
    script.update(
        _inspection_script(
            running=True,
            labels={docker_ops.INSTANCE_LABEL: "this-instance", docker_ops.JOB_LABEL: str(job_id)},
        )
    )
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    result = await service.verify_reattach(row, server)

    assert not result.ok
    assert result.reason is ReattachFailureReason.DIGEST_MISMATCH
    assert result.safe_to_teardown


async def test_verify_reattach_reports_inspection_failed_when_docker_inspect_errors(
    _patch_transport_and_tunnel, repository
) -> None:
    """An operational `docker inspect` failure must never be conflated with the
    container being confirmed gone."""
    server = _server()
    job_id = uuid4()
    row = _provisioning_row(server, job_id)
    await repository.save(row)
    script = _healthy_script()
    del script["docker inspect"]
    script["docker inspect --format {{.State.Running}}"] = _fail("Cannot connect to the Docker daemon")
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    result = await service.verify_reattach(row, server)

    assert not result.ok
    assert result.reason is ReattachFailureReason.INSPECTION_FAILED
    assert not result.safe_to_teardown


async def test_verify_reattach_reports_health_never_ready_as_transient(
    _patch_transport_and_tunnel, repository, monkeypatch
) -> None:
    monkeypatch.setattr(provisioning_module, "get_backend_instance_id", lambda: "this-instance")
    server = _server()
    job_id = uuid4()
    row = _provisioning_row(server, job_id)
    await repository.save(row)
    script = _healthy_script()
    del script["docker inspect"]
    script.update(
        _inspection_script(
            running=True,
            labels={
                docker_ops.INSTANCE_LABEL: "this-instance",
                docker_ops.JOB_LABEL: str(job_id),
                docker_ops.IMAGE_DIGEST_LABEL: _DIGEST,
            },
        )
    )
    _set_script(_patch_transport_and_tunnel, script)
    service = SshProvisioningService(repository)

    async def never_ready(base_url, server_name):
        raise TrainerReadinessTimeoutError(server_name, "no response")

    service._await_ready = never_ready  # type: ignore[method-assign]

    result = await service.verify_reattach(row, server)

    assert not result.ok
    assert result.reason is ReattachFailureReason.HEALTH_NEVER_READY
    assert result.safe_to_teardown


async def test_verify_reattach_reports_host_key_failure_without_teardown(
    _patch_transport_and_tunnel, repository, monkeypatch
) -> None:
    server = _server()
    job_id = uuid4()
    row = _provisioning_row(server, job_id)
    await repository.save(row)

    class _RaisingTransport:
        def __init__(self, alias, settings=None) -> None:
            self._alias = alias

        async def __aenter__(self):
            raise SshHostKeyMismatchError(self._alias)

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(provisioning_module, "SshTransport", _RaisingTransport)
    service = SshProvisioningService(repository)

    result = await service.verify_reattach(row, server)

    assert not result.ok
    assert result.reason is ReattachFailureReason.HOST_KEY_FAILURE
    assert not result.safe_to_teardown


async def test_verify_reattach_reports_alias_missing_without_teardown(
    _patch_transport_and_tunnel, repository, monkeypatch
) -> None:
    server = _server()
    job_id = uuid4()
    row = _provisioning_row(server, job_id)
    await repository.save(row)

    class _RaisingTransport:
        def __init__(self, alias, settings=None) -> None:
            self._alias = alias

        async def __aenter__(self):
            raise SshHostAliasNotFoundError(self._alias)

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(provisioning_module, "SshTransport", _RaisingTransport)
    service = SshProvisioningService(repository)

    result = await service.verify_reattach(row, server)

    assert not result.ok
    assert result.reason is ReattachFailureReason.ALIAS_MISSING
    assert not result.safe_to_teardown


async def test_verify_reattach_reports_port_unreachable_when_tunnel_fails(
    _patch_transport_and_tunnel, repository, monkeypatch
) -> None:
    monkeypatch.setattr(provisioning_module, "get_backend_instance_id", lambda: "this-instance")
    server = _server()
    job_id = uuid4()
    row = _provisioning_row(server, job_id)
    await repository.save(row)
    script = _healthy_script()
    del script["docker inspect"]
    script.update(
        _inspection_script(
            running=True,
            labels={
                docker_ops.INSTANCE_LABEL: "this-instance",
                docker_ops.JOB_LABEL: str(job_id),
                docker_ops.IMAGE_DIGEST_LABEL: _DIGEST,
            },
        )
    )
    _set_script(_patch_transport_and_tunnel, script)

    class _FailingTunnel(FakeSshTunnel):
        async def open(self) -> None:
            raise SshConnectionError(server.ssh_host_alias, reason="unreachable")

    monkeypatch.setattr(provisioning_module, "SshTunnel", _FailingTunnel)
    service = SshProvisioningService(repository)

    result = await service.verify_reattach(row, server)

    assert not result.ok
    assert result.reason is ReattachFailureReason.PORT_UNREACHABLE
    assert not result.safe_to_teardown
