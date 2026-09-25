# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Docker image resolution and container lifecycle."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from exceptions import (
    TrainerContainerLaunchError,
    TrainerImagePullError,
    TrainerImageResolutionError,
    TrainerImageVerificationError,
)
from schemas.hardware import DeviceType
from services.ssh import docker_ops, sigstore_verify
from services.ssh.docker_ops import LIBRARY_VERSION_LABEL, ResolvedImage
from services.ssh.trainer_image import PROTOCOL_LABEL
from services.ssh.transport import CommandResult
from settings import Settings

_REGISTRY = "ghcr.io/open-edge-platform"
_CUDA_TAG_REF = f"{_REGISTRY}/physicalai-trainer-cuda:protocol-1"
_DIGEST = "sha256:" + "a" * 64


def _ok(stdout: str = "") -> CommandResult:
    return CommandResult(argv=(), command="", exit_status=0, stdout=stdout)


def _fail(stderr: str = "command failed") -> CommandResult:
    return CommandResult(argv=(), command="", exit_status=1, stderr=stderr)


class FakeTransport:
    """Records every command and answers from a prefix-matched script."""

    def __init__(self, script: dict[str, CommandResult] | None = None) -> None:
        self.script = script or {}
        self.commands: list[tuple[str, ...]] = []

    async def run_command(self, argv, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
        self.commands.append(tuple(argv))
        joined = " ".join(argv)
        for prefix, result in self.script.items():
            if joined.startswith(prefix):
                return result
        return _fail(f"unscripted command: {joined}")

    def ran(self, fragment: str) -> bool:
        return any(fragment in " ".join(argv) for argv in self.commands)


@pytest.fixture
def settings() -> Settings:
    return Settings(TRAINER_IMAGE_REGISTRY=_REGISTRY)


async def test_verify_image_signature_pins_digest_and_fails_closed(settings: Settings) -> None:
    image = ResolvedImage(_CUDA_TAG_REF, f"{_REGISTRY}/physicalai-trainer-cuda@{_DIGEST}", _DIGEST, None)
    with patch.object(sigstore_verify, "verify_signature", new_callable=AsyncMock) as verify:
        await docker_ops.verify_image_signature(image, settings)
        verify.assert_awaited_once_with(
            image.digest_reference,
            identity_regexp=settings.cosign_certificate_identity_regexp,
            oidc_issuer=settings.cosign_oidc_issuer,
        )
        verify.side_effect = sigstore_verify.SignatureUnavailableError("offline")
        with pytest.raises(TrainerImageVerificationError):
            await docker_ops.verify_image_signature(image, settings)


# --------------------------------------------------------------------------- #
# resolve_protocol_image                                                      #
# --------------------------------------------------------------------------- #


async def test_resolve_protocol_image_returns_digest_and_labels(settings) -> None:
    labels = {PROTOCOL_LABEL: "1", LIBRARY_VERSION_LABEL: "0.5.0"}
    transport = FakeTransport(
        {
            f"docker buildx imagetools inspect {_CUDA_TAG_REF} --format {{{{json .Manifest.Digest}}}}": _ok(
                json.dumps(_DIGEST)
            ),
            f"docker buildx imagetools inspect {_CUDA_TAG_REF} --format {{{{json .Image.Config.Labels}}}}": _ok(
                json.dumps(labels)
            ),
        }
    )

    resolved = await docker_ops.resolve_protocol_image(transport, DeviceType.CUDA, 1, settings)

    assert resolved.tag_reference == _CUDA_TAG_REF
    assert resolved.digest == _DIGEST
    assert resolved.digest_reference == f"{_REGISTRY}/physicalai-trainer-cuda@{_DIGEST}"
    assert resolved.library_version == "0.5.0"


async def test_resolve_protocol_image_has_no_fallback_tag(settings) -> None:
    """An unresolved protocol tag must fail without falling back to `latest`."""
    transport = FakeTransport({})  # every command fails: unscripted

    with pytest.raises(TrainerImageResolutionError):
        await docker_ops.resolve_protocol_image(transport, DeviceType.CUDA, 1, settings)

    assert not transport.ran("latest")


async def test_resolve_protocol_image_rejects_missing_protocol_label(settings) -> None:
    transport = FakeTransport(
        {
            "docker buildx imagetools inspect": _ok(json.dumps(_DIGEST)),
        }
    )
    # Second call (labels) also matches the same broad prefix and returns the digest
    # payload, which is not a dict, so labels resolve to {} and the protocol label
    # check below is exercised.

    with pytest.raises(TrainerImageResolutionError):
        await docker_ops.resolve_protocol_image(transport, DeviceType.CUDA, 1, settings)


async def test_resolve_protocol_image_rejects_mismatched_protocol_label(settings) -> None:
    """A `protocol-1` tag whose manifest actually advertises protocol 2 must fail.

    A mis-tagged image, or a tag moved to the wrong manifest, must never
    proceed to a pull/run just because *some* protocol label is present.
    """
    labels = {PROTOCOL_LABEL: "2", LIBRARY_VERSION_LABEL: "0.5.0"}
    transport = FakeTransport(
        {
            f"docker buildx imagetools inspect {_CUDA_TAG_REF} --format {{{{json .Manifest.Digest}}}}": _ok(
                json.dumps(_DIGEST)
            ),
            f"docker buildx imagetools inspect {_CUDA_TAG_REF} --format {{{{json .Image.Config.Labels}}}}": _ok(
                json.dumps(labels)
            ),
        }
    )

    with pytest.raises(TrainerImageResolutionError):
        await docker_ops.resolve_protocol_image(transport, DeviceType.CUDA, 1, settings)


async def test_resolve_protocol_image_rejects_unparseable_protocol_label(settings) -> None:
    labels = {PROTOCOL_LABEL: "not-a-number"}
    transport = FakeTransport(
        {
            f"docker buildx imagetools inspect {_CUDA_TAG_REF} --format {{{{json .Manifest.Digest}}}}": _ok(
                json.dumps(_DIGEST)
            ),
            f"docker buildx imagetools inspect {_CUDA_TAG_REF} --format {{{{json .Image.Config.Labels}}}}": _ok(
                json.dumps(labels)
            ),
        }
    )

    with pytest.raises(TrainerImageResolutionError):
        await docker_ops.resolve_protocol_image(transport, DeviceType.CUDA, 1, settings)


# --------------------------------------------------------------------------- #
# Container launch                                                            #
# --------------------------------------------------------------------------- #


def _image() -> ResolvedImage:
    return ResolvedImage(
        tag_reference=_CUDA_TAG_REF,
        digest_reference=f"{_REGISTRY}/physicalai-trainer-cuda@{_DIGEST}",
        digest=_DIGEST,
        library_version="0.5.0",
    )


def test_build_run_argv_security_properties() -> None:
    argv = docker_ops.build_run_argv(
        image_digest_ref=f"{_REGISTRY}/physicalai-trainer-cuda@{_DIGEST}",
        device_type=DeviceType.CUDA,
        name="physicalai-trainer-abc",
        labels={"a": "b"},
        data_volume="physicalai-trainer-data-abc",
        remote_container_port=8080,
        stop_timeout_s=30,
    )

    assert argv[-1] == f"{_REGISTRY}/physicalai-trainer-cuda@{_DIGEST}"
    assert "--restart=no" in argv
    assert "127.0.0.1::8080" in argv
    assert "--privileged" not in argv
    assert "ALL" in argv and "--cap-drop" in argv
    assert "--stop-timeout=30" in argv
    assert "--shm-size=32g" in argv  # Docker's 64 MiB default exhausts PyTorch worker queues.
    assert not any(":latest" in part or part.endswith(":protocol-1") for part in argv)


def test_build_run_argv_allows_tuning_shared_memory() -> None:
    argv = docker_ops.build_run_argv(
        image_digest_ref=f"{_REGISTRY}/physicalai-trainer-cuda@{_DIGEST}",
        device_type=DeviceType.CUDA,
        name="physicalai-trainer-abc",
        labels={},
        data_volume="physicalai-trainer-data-abc",
        remote_container_port=8080,
        stop_timeout_s=30,
        shm_size_gb=8,
    )

    assert "--shm-size=8g" in argv


def test_build_run_argv_mounts_disk_backed_data_volume_not_tmpfs() -> None:
    """The trainer's storage dir is a named volume (disk), never a RAM tmpfs."""
    argv = docker_ops.build_run_argv(
        image_digest_ref=f"{_REGISTRY}/physicalai-trainer-cuda@{_DIGEST}",
        device_type=DeviceType.CUDA,
        name="physicalai-trainer-abc",
        labels={"a": "b"},
        data_volume="physicalai-trainer-data-abc",
        remote_container_port=8080,
        stop_timeout_s=30,
    )

    assert "type=volume,src=physicalai-trainer-data-abc,dst=/var/lib/physicalai-trainer" in argv
    assert any(part.startswith("/tmp:size=2g") for part in argv)
    assert not any("size=64g" in part for part in argv)


def test_build_run_argv_xpu_adds_group_add_for_render_gid() -> None:
    # Without --group-add the container's fixed non-root user cannot open the
    # render node even though --device /dev/dri passes it through, and
    # torch.xpu.is_available() silently reports zero devices.
    argv = docker_ops.build_run_argv(
        image_digest_ref=f"{_REGISTRY}/physicalai-trainer-xpu@{_DIGEST}",
        device_type=DeviceType.XPU,
        name="physicalai-trainer-abc",
        labels={"a": "b"},
        data_volume="physicalai-trainer-data-abc",
        remote_container_port=8080,
        stop_timeout_s=30,
        render_gid="44",
    )

    assert "/dev/dri" in argv
    assert "--group-add" in argv
    assert "44" in argv


def test_build_run_argv_xpu_without_render_gid_omits_group_add() -> None:
    argv = docker_ops.build_run_argv(
        image_digest_ref=f"{_REGISTRY}/physicalai-trainer-xpu@{_DIGEST}",
        device_type=DeviceType.XPU,
        name="physicalai-trainer-abc",
        labels={"a": "b"},
        data_volume="physicalai-trainer-data-abc",
        remote_container_port=8080,
        stop_timeout_s=30,
    )

    assert "--group-add" not in argv


async def test_pull_image_raises_on_failure(settings) -> None:
    transport = FakeTransport({"docker pull": _fail("no space left on device")})
    with pytest.raises(TrainerImagePullError):
        await docker_ops.pull_image(transport, _image(), settings)


async def test_pull_image_skips_pull_when_digest_already_present_locally(settings) -> None:
    """A cached image digest does not need another pull."""
    image = _image()
    transport = FakeTransport(
        {
            f"docker image inspect {image.digest_reference}": _ok(),
            "docker pull": _fail("should never be called when already cached"),
        }
    )

    await docker_ops.pull_image(transport, image, settings)

    assert not transport.ran("docker pull")


async def test_pull_image_pulls_when_not_present_locally(settings) -> None:
    image = _image()
    transport = FakeTransport(
        {
            f"docker image inspect {image.digest_reference}": _fail("No such image"),
            f"docker pull {image.digest_reference}": _ok(),
        }
    )

    await docker_ops.pull_image(transport, image, settings)

    assert transport.ran(f"docker pull {image.digest_reference}")


class _SequencedTransport(FakeTransport):
    """A `FakeTransport` where named commands answer from a queue, in order.

    Needed to simulate a background pull that is still running on the first
    check and has finished by a later one - a fixed `script` dict can only
    ever answer a given prefix the same way every time.
    """

    def __init__(
        self, script: dict[str, CommandResult] | None = None, sequences: dict[str, list[CommandResult]] | None = None
    ) -> None:
        super().__init__(script)
        self._sequences = {prefix: list(results) for prefix, results in (sequences or {}).items()}

    async def run_command(self, argv, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
        joined = " ".join(argv)
        for prefix, queue in self._sequences.items():
            if joined.startswith(prefix) and queue:
                self.commands.append(tuple(argv))
                return queue.pop(0)
        return await super().run_command(argv, timeout=timeout)


async def _no_sleep(_seconds: float) -> None:
    return None


async def test_pull_image_waits_for_in_progress_background_pull_then_skips(settings, monkeypatch) -> None:
    """Wait for an in-progress pull before fetching the same digest."""
    image = _image()
    monkeypatch.setattr(docker_ops.asyncio, "sleep", _no_sleep)
    transport = _SequencedTransport(
        {"docker pull": _fail("should never be called; the background pull already finished")},
        sequences={
            f"docker image inspect {image.digest_reference}": [_fail("No such image"), _ok()],
            "sh -c test -f": [_ok(), _fail("no pull in progress")],
        },
    )

    await docker_ops.pull_image(transport, image, settings)

    assert not transport.ran("docker pull")


async def test_pull_image_falls_back_to_direct_pull_when_background_pull_never_finishes(settings, monkeypatch) -> None:
    """A stalled background pull cannot block a direct digest pull indefinitely."""
    image = _image()
    monkeypatch.setattr(docker_ops.asyncio, "sleep", _no_sleep)
    stalled_settings = settings.model_copy(update={"ssh_image_pull_timeout_s": 0})
    transport = _SequencedTransport(
        {
            f"docker image inspect {image.digest_reference}": _fail("No such image"),
            f"docker pull {image.digest_reference}": _ok(),
        },
        sequences={"sh -c test -f": [_ok()]},
    )

    await docker_ops.pull_image(transport, image, stalled_settings)

    assert transport.ran(f"docker pull {image.digest_reference}")


# --------------------------------------------------------------------------- #
# prune_stale_images                                                          #
# --------------------------------------------------------------------------- #


async def test_prune_stale_images_removes_other_digests_of_same_repository(settings) -> None:
    image = _image()
    repository = f"{_REGISTRY}/physicalai-trainer-cuda"
    old_digest = "sha256:" + "b" * 64
    transport = FakeTransport(
        {
            f"docker images {repository} --digests": _ok(f"old-id {old_digest}\ncurrent-id {image.digest}\n"),
            "docker rmi old-id": _ok(),
            "docker rmi current-id": _fail("should never remove the image just pulled"),
        }
    )

    removed = await docker_ops.prune_stale_images(transport, image)

    assert removed == ["old-id"]
    assert transport.ran("docker rmi old-id")
    assert not transport.ran("docker rmi current-id")


async def test_prune_stale_images_keeps_image_still_in_use(settings) -> None:
    """A stale digest still referenced by a container (this installation's own
    still-running job, or another Studio installation's) must not be forced
    out from under it - `docker rmi` refusing is tolerated, not retried.
    """
    image = _image()
    repository = f"{_REGISTRY}/physicalai-trainer-cuda"
    old_digest = "sha256:" + "b" * 64
    in_use_error = (
        "Error response from daemon: conflict: unable to delete (must be forced) "
        "- image is being used by running container abc123"
    )
    transport = FakeTransport(
        {
            f"docker images {repository} --digests": _ok(f"old-id {old_digest}\n"),
            "docker rmi old-id": _fail(in_use_error),
        }
    )

    removed = await docker_ops.prune_stale_images(transport, image)

    assert removed == []


async def test_prune_stale_images_tolerates_already_removed_image(settings) -> None:
    """A stale ID can already be gone by the time this call gets to it - another
    job on the same remote server pruning the same digest concurrently, or an
    operator removing it by hand. `docker rmi` reporting "no such image" is a
    race won by someone else, not a failure worth logging.
    """
    image = _image()
    repository = f"{_REGISTRY}/physicalai-trainer-cuda"
    old_digest = "sha256:" + "b" * 64
    already_gone_error = f"Error response from daemon: No such image: old-id{old_digest}"
    transport = FakeTransport(
        {
            f"docker images {repository} --digests": _ok(f"old-id {old_digest}\n"),
            "docker rmi old-id": _fail(already_gone_error),
        }
    )

    removed = await docker_ops.prune_stale_images(transport, image)

    assert removed == []


async def test_prune_stale_images_tolerates_listing_failure(settings) -> None:
    image = _image()
    transport = FakeTransport({})  # every command fails: unscripted

    removed = await docker_ops.prune_stale_images(transport, image)

    assert removed == []


async def test_pull_image_prunes_stale_images_after_pulling(settings) -> None:
    image = _image()
    repository = f"{_REGISTRY}/physicalai-trainer-cuda"
    old_digest = "sha256:" + "b" * 64
    transport = FakeTransport(
        {
            f"docker image inspect {image.digest_reference}": _fail("No such image"),
            f"docker pull {image.digest_reference}": _ok(),
            f"docker images {repository} --digests": _ok(f"old-id {old_digest}\n"),
            "docker rmi old-id": _ok(),
        }
    )

    await docker_ops.pull_image(transport, image, settings)

    assert transport.ran("docker rmi old-id")


async def test_launch_container_returns_container_id() -> None:
    transport = FakeTransport({"docker run": _ok("abc123\n")})
    container_id = await docker_ops.launch_container(transport, ["docker", "run"], "gpu-box")
    assert container_id == "abc123"


async def test_launch_container_raises_on_failure() -> None:
    transport = FakeTransport({"docker run": _fail("port already allocated")})
    with pytest.raises(TrainerContainerLaunchError):
        await docker_ops.launch_container(transport, ["docker", "run"], "gpu-box")


# --------------------------------------------------------------------------- #
# Data volume lifecycle                                                       #
# --------------------------------------------------------------------------- #


def test_data_volume_name_is_deterministic() -> None:
    assert docker_ops.data_volume_name("abc") == "physicalai-trainer-data-abc"


async def test_create_data_volume_labels_the_volume() -> None:
    transport = FakeTransport({"docker volume create": _ok("")})

    await docker_ops.create_data_volume(transport, "physicalai-trainer-data-abc", {"a": "b"}, "gpu-box")

    assert transport.ran("--label=a=b")
    assert transport.ran("physicalai-trainer-data-abc")


async def test_create_data_volume_tolerates_existing_volume() -> None:
    transport = FakeTransport({"docker volume create": _fail("volume with name already exists")})

    await docker_ops.create_data_volume(transport, "physicalai-trainer-data-abc", {"a": "b"}, "gpu-box")


async def test_create_data_volume_raises_on_other_failure() -> None:
    transport = FakeTransport({"docker volume create": _fail("permission denied")})

    with pytest.raises(TrainerContainerLaunchError):
        await docker_ops.create_data_volume(transport, "physicalai-trainer-data-abc", {"a": "b"}, "gpu-box")


async def test_remove_volume_tolerates_missing_volume() -> None:
    transport = FakeTransport({"docker volume rm": _fail("no such volume")})

    await docker_ops.remove_volume(transport, "physicalai-trainer-data-abc")

    assert transport.ran("docker volume rm physicalai-trainer-data-abc")


# --------------------------------------------------------------------------- #
# management_labels / inspect_container (reattach support)                    #
# --------------------------------------------------------------------------- #


def test_management_labels_records_the_launched_image_digest() -> None:
    labels = docker_ops.management_labels(
        job_id="job1", server_id="server1", backend_instance_id="instance-1", image_digest=_DIGEST
    )

    assert labels[docker_ops.IMAGE_DIGEST_LABEL] == _DIGEST
    assert labels[docker_ops.JOB_LABEL] == "job1"
    assert labels[docker_ops.INSTANCE_LABEL] == "instance-1"


async def test_inspect_container_returns_none_when_container_is_gone() -> None:
    transport = FakeTransport({"docker inspect --format {{.State.Running}}": _fail("No such container")})

    result = await docker_ops.inspect_container(transport, "physicalai-trainer-job1")

    assert result is None


async def test_inspect_container_raises_when_the_inspect_command_itself_fails() -> None:
    """An operational failure (daemon down, permission denied, ...) must never be
    conflated with the container legitimately not existing."""
    transport = FakeTransport(
        {"docker inspect --format {{.State.Running}}": _fail("Cannot connect to the Docker daemon")}
    )

    with pytest.raises(docker_ops.ContainerInspectionError):
        await docker_ops.inspect_container(transport, "physicalai-trainer-job1")


async def test_inspect_container_raises_when_the_labels_call_fails() -> None:
    """The container was confirmed present by the first call; a second-call
    failure is still an operational error, not evidence of an empty label set
    (which would read as an ownership mismatch rather than "couldn't tell")."""
    transport = FakeTransport(
        {
            "docker inspect --format {{.State.Running}}": _ok("true\n"),
            "docker inspect --format {{json .Config.Labels}}": _fail("Cannot connect to the Docker daemon"),
        }
    )

    with pytest.raises(docker_ops.ContainerInspectionError):
        await docker_ops.inspect_container(transport, "physicalai-trainer-job1")


async def test_inspect_container_reports_running_state_and_labels() -> None:
    labels = {docker_ops.INSTANCE_LABEL: "instance-1", docker_ops.JOB_LABEL: "job1"}
    transport = FakeTransport(
        {
            "docker inspect --format {{.State.Running}}": _ok("true\n"),
            "docker inspect --format {{json .Config.Labels}}": _ok(json.dumps(labels)),
        }
    )

    result = await docker_ops.inspect_container(transport, "physicalai-trainer-job1")

    assert result is not None
    assert result.running is True
    assert result.labels == labels


async def test_inspect_container_reports_stopped_state() -> None:
    transport = FakeTransport(
        {
            "docker inspect --format {{.State.Running}}": _ok("false\n"),
            "docker inspect --format {{json .Config.Labels}}": _ok("{}"),
        }
    )

    result = await docker_ops.inspect_container(transport, "physicalai-trainer-job1")

    assert result is not None
    assert result.running is False


# --------------------------------------------------------------------------- #
# remove_stale_containers_on_port                                             #
# --------------------------------------------------------------------------- #


class _PortConflictTransport(FakeTransport):
    """Answers `docker ps`/`inspect`/`rm` per container id, not just by prefix.

    Regression fixture for a persistent SSH trainer whose fixed remote port
    was left held by a stale container: a real `docker ps --filter publish=`
    followed by per-id `docker inspect` calls needs distinct responses per id,
    which the plain prefix-matching `FakeTransport` can't express.
    """

    def __init__(self, ids_on_port: list[str], labels_by_id: dict[str, dict[str, str]]) -> None:
        super().__init__()
        self._ids_on_port = ids_on_port
        self._labels_by_id = labels_by_id
        self.removed: list[str] = []

    async def run_command(self, argv, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
        self.commands.append(tuple(argv))
        joined = " ".join(argv)
        if joined.startswith("docker ps -a --filter publish="):
            return _ok("\n".join(self._ids_on_port))
        if "--format {{.State.Running}}" in joined:
            return _ok("false\n")
        if "--format {{json .Config.Labels}}" in joined:
            container_id = argv[-1]
            return _ok(json.dumps(self._labels_by_id.get(container_id, {})))
        if joined.startswith("docker rm -f"):
            self.removed.append(argv[-1])
            return _ok()
        return _fail(f"unscripted command: {joined}")


async def test_remove_stale_containers_on_port_only_removes_the_same_trainers_container() -> None:
    owned_labels = {
        docker_ops.MANAGED_LABEL: "true",
        docker_ops.INSTANCE_LABEL: "this-instance",
        docker_ops.SERVER_LABEL: "trainer-id",
    }
    foreign_labels = {**owned_labels, docker_ops.INSTANCE_LABEL: "other-instance"}
    other_trainer_labels = {**owned_labels, docker_ops.SERVER_LABEL: "another-trainer"}
    transport = _PortConflictTransport(
        ids_on_port=["owned-id", "foreign-id", "other-trainer-id", "unmanaged-id"],
        labels_by_id={
            "owned-id": owned_labels,
            "foreign-id": foreign_labels,
            "other-trainer-id": other_trainer_labels,
            "unmanaged-id": {},
        },
    )

    removed = await docker_ops.remove_stale_containers_on_port(transport, 8001, "this-instance", "trainer-id")

    assert removed == ["owned-id"]
    assert transport.removed == ["owned-id"]


async def test_remove_stale_containers_on_port_is_a_noop_when_nothing_holds_the_port() -> None:
    transport = _PortConflictTransport(ids_on_port=[], labels_by_id={})

    removed = await docker_ops.remove_stale_containers_on_port(transport, 8001, "this-instance", "trainer-id")

    assert removed == []
    assert transport.removed == []
