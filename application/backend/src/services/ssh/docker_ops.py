# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Docker image resolution, verification, GPU-busy waiting, and container lifecycle.

Everything in this module runs a command over an already-connected
:class:`~services.ssh.transport.SshTransport`. No function here opens or closes
a connection: :mod:`services.ssh.provisioning` owns connection lifetime, so a
single SSH connection can span image resolution, launch, and later teardown.

Image resolution and verification never pull a layer. ``docker buildx
imagetools inspect`` reads the registry manifest (and, via ``--format``, the
image config's labels) over the registry API, the same class of call Tier 1's
preflight uses for its own registry-reachability probe. The label carrying the
`physicalai-train` version is read from this same call, before any pull, so a
version-policy rejection never costs a multi-gigabyte transfer.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from loguru import logger
from packaging.version import InvalidVersion, Version

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
from services.ssh.preflight import (  # noqa: F401 - resolve_render_group_gid re-exported for provisioning.py
    PROTOCOL_LABEL,
    protocol_tag,
    resolve_render_group_gid,
    trainer_image_ref,
)
from services.ssh.transport import SshTransport
from settings import Settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# Label carrying the `physicalai-train` version baked into the image, read
# from the registry manifest before any pull.
LIBRARY_VERSION_LABEL: Final = "org.open-edge-platform.physicalai.trainer.library-version"

# Management labels. `MANAGED_LABEL` alone never proves ownership: two Studio
# installations can target the same remote server, so the orphan sweep also
# requires `INSTANCE_LABEL` to match this installation's own backend_instance_id
# (see `core.backend_instance`).
MANAGED_LABEL: Final = "org.open-edge-platform.physicalai.managed"
JOB_LABEL: Final = "org.open-edge-platform.physicalai.job-id"
SERVER_LABEL: Final = "org.open-edge-platform.physicalai.server-id"
INSTANCE_LABEL: Final = "org.open-edge-platform.physicalai.backend-instance-id"

_CONTAINER_NAME_PREFIX: Final = "physicalai-trainer-"

# Named volume backing the trainer's writable storage directory
# (`/var/lib/physicalai-trainer`). Disk-backed, unlike the `/tmp` tmpfs, so
# uploaded datasets and model artifacts consume disk rather than RAM.
_DATA_VOLUME_NAME_PREFIX: Final = "physicalai-trainer-data-"

# Fraction of device memory in use above which an accelerator counts as busy.
# Mirrors the (advisory) preflight heuristic for XPU, which has no
# per-process attribution comparable to `nvidia-smi --query-compute-apps`.
_GPU_BUSY_MEMORY_FRACTION: Final = 0.3

_DF_AVAILABLE_COLUMN: Final = 3
_DF_MIN_COLUMNS: Final = 5

# Fixed non-root uid/gid the trainer image runs as (Dockerfile.trainer's
# `TRAINER_UID`/`TRAINER_GID` build args, both default to 10001). Used for both
# `--user` and the `--tmpfs` mount ownership below: without matching `uid=`/
# `gid=` mount options, `--read-only` + `--tmpfs` mounts default to root
# ownership, and the trainer crashes on startup unable to create its own
# storage subdirectories.
_TRAINER_UID: Final = 10001
_TRAINER_GID: Final = 10001


@dataclass(frozen=True, slots=True)
class ResolvedImage:
    """A trainer image resolved to an immutable digest, not yet pulled.

    Attributes:
        tag_reference: The `protocol-<N>` tag reference that was resolved.
        digest_reference: `<repository>@<digest>` - the reference every
            subsequent pull/run/verify call uses. Never the mutable tag.
        digest: The resolved manifest digest, e.g. `sha256:...`.
        library_version: The `physicalai-train` version reported by the
            registry manifest's label, or ``None`` if the image carries none.
    """

    tag_reference: str
    digest_reference: str
    digest: str
    library_version: str | None


@dataclass(frozen=True, slots=True)
class LibraryVersionCheck:
    """Outcome of comparing a resolved image's library version against policy."""

    reported_version: str | None
    minimum_version: str
    warning: str | None = None


class ManagedContainer:
    """One container carrying Studio's management labels, as reported by `docker ps`."""

    __slots__ = ("container_id", "job_id", "labels", "name")

    def __init__(self, container_id: str, name: str, labels: dict[str, str]) -> None:
        self.container_id = container_id
        self.name = name
        self.labels = labels
        self.job_id = labels.get(JOB_LABEL)


class ManagedVolume:
    """One data volume carrying Studio's management labels, as reported by `docker volume ls`."""

    __slots__ = ("job_id", "name")

    def __init__(self, name: str, job_id: str | None) -> None:
        self.name = name
        self.job_id = job_id


def container_name(job_id: str) -> str:
    """Return the deterministic container name for one job.

    Deterministic so a reattach after a studio restart can find the container
    by name without having persisted anything beyond the job id itself.
    """
    return f"{_CONTAINER_NAME_PREFIX}{job_id}"


def data_volume_name(job_id: str) -> str:
    """Return the deterministic data-volume name for one job.

    Deterministic for the same reason `container_name` is: teardown and the
    orphan sweep can find the volume by name without persisting anything beyond
    the job id itself. Job ids are UUIDs, so names never collide between
    installations, but the sweep still requires the volume's management labels
    to match this installation (see `list_managed_volumes`).
    """
    return f"{_DATA_VOLUME_NAME_PREFIX}{job_id}"


def management_labels(*, job_id: str, server_id: str, backend_instance_id: str) -> dict[str, str]:
    """Return the labels every Studio-launched trainer container carries.

    `INSTANCE_LABEL` is the ownership marker the orphan sweep requires in
    addition to `MANAGED_LABEL`: a remote server can be shared by more than one
    Studio installation, and sweeping must never touch a container it merely
    recognizes the shape of.
    """
    return {
        MANAGED_LABEL: "true",
        JOB_LABEL: job_id,
        SERVER_LABEL: server_id,
        INSTANCE_LABEL: backend_instance_id,
    }


def _parse_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


async def resolve_protocol_image(
    transport: SshTransport,
    device_type: DeviceType,
    protocol_version: int,
    settings: Settings,
) -> ResolvedImage:
    """Resolve the device-specific `protocol-<N>` tag to an immutable digest.

    There is deliberately no fallback tag (unlike Tier 1's advisory preflight
    check, which may warn and fall back to `latest`): a job that cannot verify
    the exact protocol it will run against must fail rather than guess.

    Reads the manifest digest and the image config's labels (including
    `LIBRARY_VERSION_LABEL` and `PROTOCOL_LABEL`) via `docker buildx imagetools
    inspect --format`, which reads the registry manifest over the registry API
    and pulls no layer.

    Args:
        transport: An open transport to the remote server.
        device_type: The server's configured accelerator.
        protocol_version: Studio's own compiled-in trainer protocol version.
        settings: Application settings (registry location).

    Returns:
        The resolved image, its digest, and its declared library version.

    Raises:
        TrainerImageResolutionError: The tag does not resolve, or its manifest
            carries no protocol label.
    """
    tag_ref = trainer_image_ref(settings.trainer_image_registry, device_type, protocol_tag(protocol_version))

    digest_result = await transport.run_command(
        ["docker", "buildx", "imagetools", "inspect", tag_ref, "--format", "{{json .Manifest.Digest}}"]
    )
    digest = _parse_json(digest_result.first_line()) if digest_result.ok else None
    if not digest_result.ok or not isinstance(digest, str) or not digest:
        raise TrainerImageResolutionError(tag_ref, protocol_version, detail=digest_result.first_line() or None)

    labels_result = await transport.run_command(
        ["docker", "buildx", "imagetools", "inspect", tag_ref, "--format", "{{json .Image.Config.Labels}}"]
    )
    labels = _parse_json(labels_result.stdout) if labels_result.ok else None
    labels = labels if isinstance(labels, dict) else {}

    reported_protocol = labels.get(PROTOCOL_LABEL)
    if reported_protocol is None:
        raise TrainerImageResolutionError(
            tag_ref, protocol_version, detail="the resolved image advertises no trainer protocol version"
        )
    try:
        reported_protocol_int = int(str(reported_protocol))
    except ValueError:
        raise TrainerImageResolutionError(
            tag_ref,
            protocol_version,
            detail=f"the resolved image advertises an unparseable protocol version: {reported_protocol!r}",
        ) from None
    if reported_protocol_int != protocol_version:
        raise TrainerImageResolutionError(
            tag_ref,
            protocol_version,
            detail=(
                f"the resolved image advertises protocol {reported_protocol_int}, "
                f"but '{tag_ref}' was resolved for protocol {protocol_version}"
            ),
        )

    repository = tag_ref.rsplit(":", 1)[0]
    raw_library_version = labels.get(LIBRARY_VERSION_LABEL)
    return ResolvedImage(
        tag_reference=tag_ref,
        digest_reference=f"{repository}@{digest}",
        digest=digest,
        library_version=raw_library_version if isinstance(raw_library_version, str) else None,
    )


async def verify_image_signature(transport: SshTransport, image: ResolvedImage, settings: Settings) -> None:
    """Verify the resolved image's signature, pinned to Studio's release identity.

    Fails closed by default: both a failed `cosign verify` and `cosign` being
    unavailable on the remote host raise. This is the opposite policy from
    Tier 1's advisory preflight signature check, which is defense-in-depth on
    top of the publish-time signature and never blocks a save.

    `cosign` being unavailable can be downgraded to a non-blocking warning by
    setting `settings.ssh_require_cosign_verification` to `False` (e.g. for a
    host where installing `cosign` is not viable). A failed `cosign verify`
    always raises regardless of that setting: it means the image's signature
    does not match Studio's expected identity, not that a tool is missing.

    Args:
        transport: An open transport to the remote server.
        image: The digest-resolved image to verify.
        settings: Application settings (pinned certificate identity/issuer,
            and whether `cosign` availability is required).

    Raises:
        TrainerImageVerificationError: `cosign` is unavailable and required,
            or verification failed.
    """
    available = await transport.run_command(["cosign", "version"])
    if not available.ok:
        if not settings.ssh_require_cosign_verification:
            logger.warning(
                "cosign is not available on the remote host; proceeding without signature "
                "verification for '{}' because SSH_REQUIRE_COSIGN_VERIFICATION is disabled.",
                image.digest_reference,
            )
            return
        raise TrainerImageVerificationError(image.digest_reference, "cosign is not available on the remote host")

    verified = await transport.run_command(
        [
            "cosign",
            "verify",
            image.digest_reference,
            "--certificate-identity-regexp",
            settings.cosign_certificate_identity_regexp,
            "--certificate-oidc-issuer",
            settings.cosign_oidc_issuer,
        ]
    )
    if not verified.ok:
        raise TrainerImageVerificationError(image.digest_reference, verified.first_line() or "signature not verified")


def check_library_version(
    image: ResolvedImage,
    *,
    minimum_version: str,
    policy_name: str = "default",
) -> LibraryVersionCheck:
    """Range-check a resolved image's library version against a minimum policy.

    Older than the minimum is a non-fatal warning (the job proceeds); equal or
    newer is silent; a version this policy documents as required but the image
    does not meet is fatal, raised before any pull. A non-default policy also
    fails closed when the image carries no version label at all: an image
    cannot satisfy a required minimum it declines to report, so a missing
    label is not the same as compliance.

    Args:
        image: The digest-resolved image, carrying the registry label.
        minimum_version: The minimum `physicalai-train` version this policy
            requires.
        policy_name: Name surfaced in the failure message (e.g. a model
            family), for a policy stricter than Studio's own default minimum.

    Returns:
        The check outcome. `.warning` is set when the image is older than
        `minimum_version` but the policy is Studio's own non-strict default.

    Raises:
        TrainerLibraryVersionError: The image's version does not meet
            `minimum_version` and `policy_name` names a required minimum, or
            `policy_name` is non-default and the image carries no version
            label at all.
        ValueError: `minimum_version` is not a valid PEP 440 version.
    """
    reported = image.library_version
    if reported is None:
        if policy_name != "default":
            raise TrainerLibraryVersionError(policy_name, minimum_version, "unknown (no version label)")
        return LibraryVersionCheck(reported_version=None, minimum_version=minimum_version)

    try:
        minimum_parsed = Version(minimum_version)
    except InvalidVersion:
        raise ValueError(f"minimum_version '{minimum_version}' is not a valid PEP 440 version") from None

    try:
        reported_parsed = Version(reported)
    except InvalidVersion:
        return LibraryVersionCheck(
            reported_version=None,
            minimum_version=minimum_version,
            warning=f"Could not parse trainer library version '{reported}'; proceeding without a version check.",
        )

    if reported_parsed >= minimum_parsed:
        return LibraryVersionCheck(reported_version=reported, minimum_version=minimum_version)

    if policy_name != "default":
        raise TrainerLibraryVersionError(policy_name, minimum_version, reported)

    return LibraryVersionCheck(
        reported_version=reported,
        minimum_version=minimum_version,
        warning=f"Trainer image reports physicalai-train version '{reported}', older than '{minimum_version}'.",
    )


def _first_number_pair(stdout: str) -> tuple[float, float] | None:
    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", stdout)]
    if len(numbers) < 2 or numbers[1] <= 0:
        return None
    return numbers[0], numbers[1]


async def _cuda_gpu_busy(transport: SshTransport) -> bool | None:
    """Return whether a CUDA GPU is busy, or ``None`` if occupancy is unknown."""
    result = await transport.run_command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    if not result.ok:
        return None
    return bool([line for line in result.stdout.splitlines() if line.strip()])


async def _xpu_gpu_busy(transport: SshTransport) -> bool | None:
    """Return whether an XPU is busy by a memory-utilization heuristic, or ``None``."""
    result = await transport.run_command(["xpu-smi", "stats", "-d", "0"])
    if not result.ok:
        return None
    pair = _first_number_pair(result.stdout)
    if pair is None:
        return None
    used, total = pair
    return (used / total) >= _GPU_BUSY_MEMORY_FRACTION


async def is_gpu_busy(transport: SshTransport, device_type: DeviceType) -> bool | None:
    """Return whether the remote accelerator is currently busy.

    Returns ``None`` when occupancy could not be determined - callers should
    treat that as "not known to be busy" rather than blocking on a monitoring
    tool that happens to be absent.
    """
    if device_type is DeviceType.CUDA:
        return await _cuda_gpu_busy(transport)
    return await _xpu_gpu_busy(transport)


async def wait_for_gpu_free(
    open_transport: Callable[[], SshTransport],
    device_type: DeviceType,
    settings: Settings,
    server_name: str,
    *,
    on_wait: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    """Wait until the remote GPU reports free, with exponential backoff.

    Opens a fresh transport for each probe (rather than holding one connection
    for the whole wait) so a long wait does not hold a scarce per-server
    connection slot the whole time; probes still respect the alias's shared
    connect throttle.

    Args:
        open_transport: Returns a new, unconnected transport for one probe.
        device_type: The server's configured accelerator.
        settings: Application settings (backoff bounds, give-up timeout).
        server_name: Used only in the raised error message.
        on_wait: Awaited once per non-final wait, with the elapsed seconds so
            far - the caller uses this to report a `waiting` phase state.
        clock: Injected for deterministic tests.
        sleep: Injected for deterministic tests; defaults to `asyncio.sleep`.

    Raises:
        GpuBusyTimeoutError: The GPU stayed busy past
            `settings.ssh_gpu_wait_giveup_s`.
    """
    if sleep is None:
        import asyncio

        sleep = asyncio.sleep

    started = clock()
    backoff = settings.ssh_gpu_wait_initial_backoff_s
    while True:
        transport = open_transport()
        try:
            await transport.connect()
            busy = await is_gpu_busy(transport, device_type)
        finally:
            await transport.close()

        if not busy:
            return

        elapsed = clock() - started
        if elapsed >= settings.ssh_gpu_wait_giveup_s:
            raise GpuBusyTimeoutError(server_name, elapsed)

        if on_wait is not None:
            await on_wait(elapsed)

        await sleep(min(backoff, settings.ssh_gpu_wait_giveup_s - elapsed))
        backoff = min(backoff * 2, settings.ssh_gpu_wait_max_backoff_s)


def _parse_free_bytes(stdout: str) -> int | None:
    """Parse available bytes out of `df -B1 -P` output."""
    for line in stdout.splitlines()[1:]:
        columns = line.split()
        if len(columns) < _DF_MIN_COLUMNS:
            continue
        try:
            return int(columns[_DF_AVAILABLE_COLUMN])
        except ValueError:
            continue
    return None


async def check_disk_for_job(transport: SshTransport, required_bytes: int, server_name: str) -> None:
    """Re-check free disk against this job's actual snapshot size.

    A server's save-time preflight only checks a nominal minimum; a specific
    job's dataset can exceed it, so provisioning re-checks against the real
    size before pulling or launching anything.

    Raises:
        RemoteDiskSpaceError: Free space is below `required_bytes`.
    """
    result = await transport.run_command(["df", "-B1", "-P", "/var/lib/docker"])
    if not result.ok:
        result = await transport.run_command(["df", "-B1", "-P", "/"])

    free_bytes = _parse_free_bytes(result.stdout) if result.ok else None
    if free_bytes is None:
        # Unable to measure: fail closed rather than silently skipping the check.
        raise RemoteDiskSpaceError(server_name, free_bytes=0, required_bytes=required_bytes)
    if free_bytes < required_bytes:
        raise RemoteDiskSpaceError(server_name, free_bytes=free_bytes, required_bytes=required_bytes)


def _device_run_args(device_type: DeviceType, render_gid: str | None = None) -> list[str]:
    """Return the `docker run` flags that expose the accelerator.

    `render_gid`, when known, is added via `--group-add` so the container's
    fixed non-root user (`--user`, set to `_TRAINER_UID:_TRAINER_GID` below)
    can actually read/write the render node - `--device /dev/dri` alone passes
    the node through but its group ownership still gates access, and without
    this the container's `torch.xpu.is_available()` silently reports zero
    devices.
    """
    if device_type is DeviceType.CUDA:
        return ["--gpus", "all"]
    args = ["--device", "/dev/dri"]
    if render_gid:
        args.extend(["--group-add", render_gid])
    return args


def build_run_argv(  # noqa: PLR0913 - each flag is an independent run/security property
    *,
    image_digest_ref: str,
    device_type: DeviceType,
    name: str,
    labels: dict[str, str],
    data_volume: str,
    remote_container_port: int,
    stop_timeout_s: int,
    render_gid: str | None = None,
) -> list[str]:
    """Build the least-privilege `docker run` command for one trainer container.

    * Launches by digest (`image_digest_ref`), never a mutable tag.
    * `-p 127.0.0.1::<port>` publishes an OS-assigned ephemeral host port bound
      only to loopback - never reachable from another host on the network.
    * `--restart=no` - a crashed container must surface as a failed job, not
      restart silently out from under a job that already moved on.
    * Non-root, every Linux capability dropped, no `--privileged`, and only the
      device nodes the configured accelerator needs are passed through.
    * `--read-only` root filesystem, one bounded `--tmpfs` for `/tmp` scratch,
      and a single job-scoped data volume mounted at the trainer's storage
      directory (`/var/lib/physicalai-trainer`) for datasets and model
      artifacts - disk-backed so large uploads consume disk, not RAM.

    `render_gid`, from `resolve_render_group_gid`, is required for a working
    XPU container: without it `--device /dev/dri` passes the render node
    through but the fixed non-root user cannot open it, and
    `torch.xpu.is_available()` reports zero devices. Ignored for CUDA.

    The `/tmp` tmpfs carries explicit `uid=`/`gid=` mount options matching
    `--user`: an unqualified `--tmpfs` mounts root-owned, and the trainer's
    fixed non-root user cannot write it. The data volume needs no such options:
    Docker initializes a fresh volume with the image's directory ownership
    (`trainer:trainer` from `Dockerfile.trainer`).
    """
    tmpfs_owner = f"uid={_TRAINER_UID},gid={_TRAINER_GID}"
    return [
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        *(f"--label={key}={value}" for key, value in labels.items()),
        "--restart=no",
        f"--stop-timeout={stop_timeout_s}",
        "--publish",
        f"127.0.0.1::{remote_container_port}",
        "--user",
        f"{_TRAINER_UID}:{_TRAINER_GID}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        f"/tmp:size=2g,{tmpfs_owner}",  # noqa: S108  # nosec B108 - a `docker run` mount spec, not a local temp-file access
        "--mount",
        f"type=volume,src={data_volume},dst=/var/lib/physicalai-trainer",
        *_device_run_args(device_type, render_gid),
        image_digest_ref,
    ]


async def pull_image(transport: SshTransport, image: ResolvedImage, settings: Settings) -> None:
    """Pull the resolved image by digest.

    Uses `settings.ssh_image_pull_timeout_s` rather than the default
    `ssh_command_timeout_s`: that budget is sized for cheap probes
    (`docker version`, `nvidia-smi`), and is far too short for a multi-gigabyte
    image transfer. A pull that is still legitimately in progress when the
    short budget elapses would otherwise surface as a spurious pull failure,
    with only the partial progress output as its (misleading) detail.

    Raises:
        TrainerImagePullError: The pull failed, or did not finish within
            `settings.ssh_image_pull_timeout_s`.
    """
    result = await transport.run_command(
        ["docker", "pull", image.digest_reference], timeout=settings.ssh_image_pull_timeout_s
    )
    if not result.ok:
        raise TrainerImagePullError(image.digest_reference, detail=result.stderr or result.stdout or None)


async def launch_container(transport: SshTransport, argv: list[str], server_name: str) -> str:
    """Run `docker run --detach ...` and return the started container id.

    Raises:
        TrainerContainerLaunchError: The container could not be started.
    """
    result = await transport.run_command(argv)
    container_id = result.first_line()
    if not result.ok or not container_id:
        raise TrainerContainerLaunchError(server_name, detail=result.stderr or result.stdout or None)
    return container_id


async def resolve_published_port(transport: SshTransport, name: str, remote_container_port: int) -> int | None:
    """Return the host port Docker assigned for `remote_container_port`, or ``None``."""
    result = await transport.run_command(["docker", "port", name, str(remote_container_port)])
    if not result.ok:
        return None
    # `docker port` prints e.g. "127.0.0.1:54321".
    line = result.first_line()
    _, _, port_text = line.rpartition(":")
    try:
        return int(port_text)
    except ValueError:
        return None


async def stop_and_remove_container(transport: SshTransport, name_or_id: str, stop_timeout_s: int) -> None:
    """Stop and remove a container by name or id. Best-effort past `docker stop`.

    Tolerates the container already being gone (a previous teardown attempt
    that partially succeeded is a normal path here, not an error).
    """
    await transport.run_command(["docker", "stop", "--time", str(stop_timeout_s), name_or_id])
    remove = await transport.run_command(["docker", "rm", "--force", name_or_id])
    if not remove.ok and "No such container" not in (remove.stderr or ""):
        logger.warning("docker rm reported a failure for container '{}': {}", name_or_id, remove.stderr)


async def create_data_volume(transport: SshTransport, name: str, labels: dict[str, str], server_name: str) -> None:
    """Create the job's data volume with management labels, tolerating an existing one.

    Created explicitly rather than letting `docker run --mount` auto-create it,
    so the volume carries the management labels the orphan sweep relies on to
    attribute it to this installation. `docker run` then mounts the already
    labeled volume.

    Raises:
        TrainerContainerLaunchError: The volume could not be created.
    """
    label_args = [f"--label={key}={value}" for key, value in labels.items()]
    result = await transport.run_command(["docker", "volume", "create", *label_args, name])
    if result.ok or "already exists" in (result.stderr or "").lower():
        return
    raise TrainerContainerLaunchError(server_name, detail=result.stderr or result.stdout or None)


async def remove_volume(transport: SshTransport, name: str) -> None:
    """Remove a data volume by name. Best-effort, tolerating a missing volume."""
    result = await transport.run_command(["docker", "volume", "rm", name])
    if not result.ok and "no such volume" not in (result.stderr or "").lower():
        logger.warning("docker volume rm reported a failure for volume '{}': {}", name, result.stderr)


def _parse_label_map(raw_labels: str) -> dict[str, str]:
    """Parse a Docker `Labels` string (`k1=v1,k2=v2`) into a mapping."""
    labels: dict[str, str] = {}
    for entry in raw_labels.split(","):
        key, sep, value = entry.partition("=")
        if sep:
            labels[key] = value
    return labels


def _parse_managed_container_line(line: str) -> ManagedContainer | None:
    parsed = _parse_json(line)
    if not isinstance(parsed, dict):
        return None
    container_id = parsed.get("ID")
    name = parsed.get("Names")
    raw_labels = parsed.get("Labels", "")
    if not isinstance(container_id, str) or not isinstance(name, str) or not isinstance(raw_labels, str):
        return None
    return ManagedContainer(container_id=container_id, name=name, labels=_parse_label_map(raw_labels))


def _parse_managed_volume_line(line: str) -> ManagedVolume | None:
    parsed = _parse_json(line)
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("Name")
    raw_labels = parsed.get("Labels", "")
    if not isinstance(name, str) or not isinstance(raw_labels, str):
        return None
    return ManagedVolume(name=name, job_id=_parse_label_map(raw_labels).get(JOB_LABEL))


async def list_managed_containers(transport: SshTransport, backend_instance_id: str) -> list[ManagedContainer]:
    """List containers this installation provably owns, running or stopped.

    Filters on both `MANAGED_LABEL` and `INSTANCE_LABEL`: a remote server can be
    shared by another Studio installation, whose containers must never be
    listed (and therefore never swept) by this one.
    """
    result = await transport.run_command(
        [
            "docker",
            "ps",
            "--all",
            "--filter",
            f"label={MANAGED_LABEL}=true",
            "--filter",
            f"label={INSTANCE_LABEL}={backend_instance_id}",
            "--format",
            "{{json .}}",
        ]
    )
    if not result.ok:
        return []
    containers = (_parse_managed_container_line(line) for line in result.stdout.splitlines() if line.strip())
    return [container for container in containers if container is not None]


async def list_managed_volumes(transport: SshTransport, backend_instance_id: str) -> list[ManagedVolume]:
    """List data volumes this installation provably owns.

    Filters on both `MANAGED_LABEL` and `INSTANCE_LABEL`, exactly like
    `list_managed_containers`, so a volume left by a different Studio
    installation on a shared host is never swept by this one.
    """
    result = await transport.run_command(
        [
            "docker",
            "volume",
            "ls",
            "--filter",
            f"label={MANAGED_LABEL}=true",
            "--filter",
            f"label={INSTANCE_LABEL}={backend_instance_id}",
            "--format",
            "{{json .}}",
        ]
    )
    if not result.ok:
        return []
    volumes = (_parse_managed_volume_line(line) for line in result.stdout.splitlines() if line.strip())
    return [volume for volume in volumes if volume is not None]
