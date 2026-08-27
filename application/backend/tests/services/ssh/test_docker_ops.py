# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Docker image resolution, verification, GPU-busy waiting, and container lifecycle."""

from __future__ import annotations

import json

import pytest

from exceptions import (
    GpuBusyTimeoutError,
    RemoteDiskSpaceError,
    TrainerContainerLaunchError,
    TrainerImagePullError,
    TrainerImageResolutionError,
    TrainerImageVerificationError,
    TrainerLibraryVersionError,
)
from schemas.hardware import DeviceType
from services.ssh import docker_ops
from services.ssh.docker_ops import LIBRARY_VERSION_LABEL, ResolvedImage
from services.ssh.preflight import PROTOCOL_LABEL
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
    """Unlike Tier 1's advisory preflight, there is no `latest` fallback here."""
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
# verify_image_signature                                                      #
# --------------------------------------------------------------------------- #


def _image() -> ResolvedImage:
    return ResolvedImage(
        tag_reference=_CUDA_TAG_REF,
        digest_reference=f"{_REGISTRY}/physicalai-trainer-cuda@{_DIGEST}",
        digest=_DIGEST,
        library_version="0.5.0",
    )


async def test_verify_image_signature_fails_closed_when_cosign_unavailable(settings) -> None:
    transport = FakeTransport({"cosign version": _fail()})

    with pytest.raises(TrainerImageVerificationError):
        await docker_ops.verify_image_signature(transport, _image(), settings)


async def test_verify_image_signature_fails_closed_on_failed_verification(settings) -> None:
    transport = FakeTransport({"cosign version": _ok("v2.4.1"), "cosign verify": _fail("no matching signatures")})

    with pytest.raises(TrainerImageVerificationError):
        await docker_ops.verify_image_signature(transport, _image(), settings)


async def test_verify_image_signature_allows_missing_cosign_when_not_required(settings) -> None:
    """`SSH_REQUIRE_COSIGN_VERIFICATION=false` downgrades a missing `cosign` to a warning."""
    settings = Settings(TRAINER_IMAGE_REGISTRY=_REGISTRY, SSH_REQUIRE_COSIGN_VERIFICATION=False)
    transport = FakeTransport({"cosign version": _fail()})

    await docker_ops.verify_image_signature(transport, _image(), settings)


async def test_verify_image_signature_still_fails_closed_on_bad_signature_when_not_required(settings) -> None:
    """Opting out of requiring `cosign` never excuses a signature that fails verification."""
    settings = Settings(TRAINER_IMAGE_REGISTRY=_REGISTRY, SSH_REQUIRE_COSIGN_VERIFICATION=False)
    transport = FakeTransport({"cosign version": _ok("v2.4.1"), "cosign verify": _fail("no matching signatures")})

    with pytest.raises(TrainerImageVerificationError):
        await docker_ops.verify_image_signature(transport, _image(), settings)


async def test_verify_image_signature_passes_and_pins_identity(settings) -> None:
    transport = FakeTransport({"cosign version": _ok("v2.4.1"), "cosign verify": _ok("Verified OK")})

    await docker_ops.verify_image_signature(transport, _image(), settings)

    assert any(
        "--certificate-identity-regexp" in " ".join(argv) and settings.cosign_oidc_issuer in " ".join(argv)
        for argv in transport.commands
    )


# --------------------------------------------------------------------------- #
# check_library_version                                                       #
# --------------------------------------------------------------------------- #


def test_check_library_version_no_label_is_silent() -> None:
    image = ResolvedImage(tag_reference="t", digest_reference="d", digest=_DIGEST, library_version=None)
    result = docker_ops.check_library_version(image, minimum_version="1.0.0")
    assert result.warning is None
    assert result.reported_version is None


def test_check_library_version_no_label_fails_closed_for_a_named_policy() -> None:
    """A strict, non-default policy must not be bypassable by omitting the label."""
    image = ResolvedImage(tag_reference="t", digest_reference="d", digest=_DIGEST, library_version=None)
    with pytest.raises(TrainerLibraryVersionError):
        docker_ops.check_library_version(image, minimum_version="1.0.0", policy_name="pi05")


def test_check_library_version_older_is_a_warning_not_a_failure() -> None:
    image = ResolvedImage(tag_reference="t", digest_reference="d", digest=_DIGEST, library_version="0.9.0")
    result = docker_ops.check_library_version(image, minimum_version="1.0.0")
    assert result.warning is not None


def test_check_library_version_equal_or_newer_is_silent() -> None:
    image = ResolvedImage(tag_reference="t", digest_reference="d", digest=_DIGEST, library_version="1.2.0")
    result = docker_ops.check_library_version(image, minimum_version="1.0.0")
    assert result.warning is None


def test_check_library_version_below_named_policy_minimum_fails() -> None:
    image = ResolvedImage(tag_reference="t", digest_reference="d", digest=_DIGEST, library_version="0.9.0")
    with pytest.raises(TrainerLibraryVersionError):
        docker_ops.check_library_version(image, minimum_version="1.0.0", policy_name="pi05")


def test_check_library_version_unparseable_label_is_treated_as_unreported() -> None:
    """An unparseable label (e.g. the Dockerfile's `unknown` default) must warn, not raise.

    `reported_version` must come back `None`, not the raw unparseable string:
    `SshProvisioningService.provision()` treats any non-`None` `reported_version`
    as authoritative against `/health`'s real version, and would otherwise raise
    `TrainerLibraryVersionMismatchError` on every image whose label isn't a
    valid PEP 440 version string.
    """
    image = ResolvedImage(tag_reference="t", digest_reference="d", digest=_DIGEST, library_version="unknown")
    result = docker_ops.check_library_version(image, minimum_version="1.0.0")
    assert result.reported_version is None
    assert result.warning is not None


def test_check_library_version_invalid_minimum_version_fails_fast() -> None:
    """A misconfigured minimum is config, not untrusted image data: it must raise."""
    image = ResolvedImage(tag_reference="t", digest_reference="d", digest=_DIGEST, library_version="1.0.0")
    with pytest.raises(ValueError):
        docker_ops.check_library_version(image, minimum_version="not-a-version")


# --------------------------------------------------------------------------- #
# GPU-busy wait                                                               #
# --------------------------------------------------------------------------- #


async def test_wait_for_gpu_free_returns_once_free(settings) -> None:
    calls = iter([_ok("1234\n"), _ok("\n")])  # busy once, then free

    class _Sequenced(FakeTransport):
        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def run_command(self, argv, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
            return next(calls)

    waits: list[float] = []

    async def on_wait(elapsed: float) -> None:
        waits.append(elapsed)

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    clock_values = iter([0.0, 0.0, 5.0])

    def fake_clock() -> float:
        return next(clock_values, 5.0)

    await docker_ops.wait_for_gpu_free(
        lambda: _Sequenced(),
        DeviceType.CUDA,
        settings,
        "gpu-box",
        on_wait=on_wait,
        clock=fake_clock,
        sleep=fake_sleep,
    )

    assert waits == [0.0]
    assert sleeps  # backed off at least once


async def test_wait_for_gpu_free_gives_up_after_timeout(settings) -> None:
    settings = settings.model_copy(update={"ssh_gpu_wait_giveup_s": 1.0, "ssh_gpu_wait_initial_backoff_s": 0.01})

    class _AlwaysBusy(FakeTransport):
        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def run_command(self, argv, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
            return _ok("1234\n")

    async def fake_sleep(seconds: float) -> None:
        return None

    clock_values = iter([0.0, 0.5, 2.0])

    def fake_clock() -> float:
        return next(clock_values, 2.0)

    with pytest.raises(GpuBusyTimeoutError):
        await docker_ops.wait_for_gpu_free(
            lambda: _AlwaysBusy(),
            DeviceType.CUDA,
            settings,
            "gpu-box",
            clock=fake_clock,
            sleep=fake_sleep,
        )


# --------------------------------------------------------------------------- #
# Disk re-check                                                               #
# --------------------------------------------------------------------------- #

_DF_HEADER = "Filesystem 1B-blocks Used Available Capacity Mounted on\n"
_PLENTY_OF_DISK = f"{_DF_HEADER}/dev/sda1 900000000000 100000000000 85899345920 12% /var/lib/docker\n"
_ALMOST_NO_DISK = f"{_DF_HEADER}/dev/sda1 900000000000 890000000000 1073741824 99% /var/lib/docker\n"


async def test_check_disk_for_job_passes_with_enough_free_space() -> None:
    transport = FakeTransport({"df -B1 -P /var/lib/docker": _ok(_PLENTY_OF_DISK)})
    await docker_ops.check_disk_for_job(transport, required_bytes=10 * 1024**3, server_name="gpu-box")


async def test_check_disk_for_job_fails_against_actual_snapshot_size() -> None:
    transport = FakeTransport({"df -B1 -P /var/lib/docker": _ok(_ALMOST_NO_DISK)})
    with pytest.raises(RemoteDiskSpaceError):
        await docker_ops.check_disk_for_job(transport, required_bytes=500 * 1024**3, server_name="gpu-box")


# --------------------------------------------------------------------------- #
# Container launch                                                            #
# --------------------------------------------------------------------------- #


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
    assert not any(":latest" in part or part.endswith(":protocol-1") for part in argv)


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


async def test_launch_container_returns_container_id() -> None:
    transport = FakeTransport({"docker run": _ok("abc123\n")})
    container_id = await docker_ops.launch_container(transport, ["docker", "run"], "gpu-box")
    assert container_id == "abc123"


async def test_launch_container_raises_on_failure() -> None:
    transport = FakeTransport({"docker run": _fail("port already allocated")})
    with pytest.raises(TrainerContainerLaunchError):
        await docker_ops.launch_container(transport, ["docker", "run"], "gpu-box")


async def test_resolve_published_port_parses_docker_port_output() -> None:
    transport = FakeTransport({"docker port": _ok("127.0.0.1:54321\n")})
    port = await docker_ops.resolve_published_port(transport, "physicalai-trainer-abc", 8080)
    assert port == 54321


# --------------------------------------------------------------------------- #
# Orphan sweep listing                                                        #
# --------------------------------------------------------------------------- #


async def test_list_managed_containers_parses_docker_ps_json_lines() -> None:
    line = json.dumps(
        {
            "ID": "abc123",
            "Names": "physicalai-trainer-job1",
            "Labels": f"{docker_ops.MANAGED_LABEL}=true,{docker_ops.JOB_LABEL}=job1",
        }
    )
    transport = FakeTransport({"docker ps": _ok(line + "\n")})

    containers = await docker_ops.list_managed_containers(transport, backend_instance_id="instance-1")

    assert len(containers) == 1
    assert containers[0].container_id == "abc123"
    assert containers[0].job_id == "job1"


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


async def test_list_managed_volumes_parses_docker_volume_ls_json_lines() -> None:
    line = json.dumps(
        {
            "Name": "physicalai-trainer-data-job1",
            "Labels": f"{docker_ops.MANAGED_LABEL}=true,{docker_ops.JOB_LABEL}=job1",
        }
    )
    transport = FakeTransport({"docker volume ls": _ok(line + "\n")})

    volumes = await docker_ops.list_managed_volumes(transport, backend_instance_id="instance-1")

    assert len(volumes) == 1
    assert volumes[0].name == "physicalai-trainer-data-job1"
    assert volumes[0].job_id == "job1"
