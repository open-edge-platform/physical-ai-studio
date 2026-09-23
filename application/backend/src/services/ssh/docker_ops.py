# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Docker image resolution and container lifecycle for SSH-tunneled trainers.

Everything here runs a command over an already-connected
:class:`~services.ssh.transport.SshTransport`. No function opens or closes a
connection.

Image resolution never pulls a layer: ``docker buildx imagetools inspect``
reads the registry manifest and image-config labels (including the
`physicalai-train` version) over the registry API, so a version-policy
rejection never costs a multi-gigabyte transfer.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from loguru import logger

from exceptions import (
    TrainerContainerLaunchError,
    TrainerImagePullError,
    TrainerImageResolutionError,
    TrainerImageVerificationError,
)
from schemas.hardware import DeviceType
from services.ssh import sigstore_verify
from services.ssh.trainer_image import (
    PROTOCOL_LABEL,
    image_present_locally,
    protocol_tag,
    pull_in_progress,
    trainer_image_ref,
)
from services.ssh.transport import SshTransport
from settings import Settings

if TYPE_CHECKING:
    from services.ssh.transport import CommandResult

# Label carrying the `physicalai-train` version baked into the image, read
# from the registry manifest before any pull.
LIBRARY_VERSION_LABEL: Final = "org.open-edge-platform.physicalai.trainer.library-version"

# Management labels identify containers owned by this installation and trainer.
# Cleanup checks the managed, instance, and trainer identifiers together.
MANAGED_LABEL: Final = "org.open-edge-platform.physicalai.managed"
JOB_LABEL: Final = "org.open-edge-platform.physicalai.job-id"
SERVER_LABEL: Final = "org.open-edge-platform.physicalai.server-id"
INSTANCE_LABEL: Final = "org.open-edge-platform.physicalai.backend-instance-id"
# Records the image digest used to start the container.
IMAGE_DIGEST_LABEL: Final = "org.open-edge-platform.physicalai.image-digest"

# Named volume backing the trainer's writable storage directory. Disk-backed,
# unlike the `/tmp` tmpfs, so datasets/artifacts consume disk rather than RAM.
_DATA_VOLUME_NAME_PREFIX: Final = "physicalai-trainer-data-"

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
            subsequent pull/run call uses. Never the mutable tag.
        digest: The resolved manifest digest, e.g. `sha256:...`.
        library_version: The `physicalai-train` version reported by the
            registry manifest's label, or ``None`` if the image carries none.
    """

    tag_reference: str
    digest_reference: str
    digest: str
    library_version: str | None


def data_volume_name(trainer_id: str) -> str:
    """Return a trainer's deterministic data-volume name."""
    return f"{_DATA_VOLUME_NAME_PREFIX}{trainer_id}"


def management_labels(*, job_id: str, server_id: str, backend_instance_id: str, image_digest: str) -> dict[str, str]:
    """Return the labels every Studio-launched trainer container carries.

    Port cleanup checks `MANAGED_LABEL`, `INSTANCE_LABEL`, and `SERVER_LABEL`
    before removing a container. `IMAGE_DIGEST_LABEL` records its image digest.
    """
    return {
        MANAGED_LABEL: "true",
        JOB_LABEL: job_id,
        SERVER_LABEL: server_id,
        INSTANCE_LABEL: backend_instance_id,
        IMAGE_DIGEST_LABEL: image_digest,
    }


@dataclass(frozen=True, slots=True)
class ContainerInspection:
    """A container's running state and labels, as reported by `docker inspect`."""

    running: bool
    labels: dict[str, str]


class ContainerInspectionError(Exception):
    """`docker inspect` failed for a reason other than the container being absent.

    Raised instead of folding into `inspect_container`'s ``None`` return, so a
    caller never mistakes an operational failure for the container
    legitimately not existing.
    """

    def __init__(self, name_or_id: str, detail: str | None = None) -> None:
        self.name_or_id = name_or_id
        self.detail = detail
        message = f"docker inspect failed for container '{name_or_id}'"
        super().__init__(f"{message}: {detail}" if detail else message)


# Substrings docker's own CLI/daemon use to report that the named object truly
# does not exist, across the docker versions this module targets. Any other
# failure (permission denied, daemon unavailable, a malformed name) is
# inconclusive and must not be treated the same as "gone".
_NOT_FOUND_MARKERS: Final = ("no such object", "no such container")


def _container_not_found(result: CommandResult) -> bool:
    """True when a failed `docker inspect` means the container does not exist."""
    stderr = (result.stderr or "").lower()
    return any(marker in stderr for marker in _NOT_FOUND_MARKERS)


async def inspect_container(transport: SshTransport, name_or_id: str) -> ContainerInspection | None:
    """Inspect a container's running state and labels.

    Returns:
        ``None`` when `docker inspect` confirms the container does not exist
        at all (its stderr reports "No such object"/"No such container"). A
        container that exists but is stopped is still returned, with
        ``running=False``.

    Raises:
        ContainerInspectionError: Either inspect call failed for some other
            reason - the container's existence, or its labels, could not be
            determined. Never raised for a container confirmed absent by the
            first call.
    """
    running_result = await transport.run_command(["docker", "inspect", "--format", "{{.State.Running}}", name_or_id])
    if not running_result.ok:
        if _container_not_found(running_result):
            return None
        raise ContainerInspectionError(name_or_id, detail=running_result.stderr or running_result.stdout or None)

    labels_result = await transport.run_command(
        ["docker", "inspect", "--format", "{{json .Config.Labels}}", name_or_id]
    )
    if not labels_result.ok:
        # The first inspect call succeeded, so a failure here is operational,
        # not evidence the container is gone. Folding this into `labels={}`
        # would let ownership/digest checks read a real error as "mismatch".
        if _container_not_found(labels_result):
            return None
        raise ContainerInspectionError(name_or_id, detail=labels_result.stderr or labels_result.stdout or None)
    labels = _parse_json(labels_result.stdout)
    labels = labels if isinstance(labels, dict) else {}
    return ContainerInspection(running=running_result.first_line().lower() == "true", labels=labels)


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

    There is deliberately no fallback tag: a launch that cannot verify the
    exact protocol it will run against must fail rather than guess.

    Reads the manifest digest and the image config's labels (including
    `LIBRARY_VERSION_LABEL` and `PROTOCOL_LABEL`) via `docker buildx imagetools
    inspect --format`, which reads the registry manifest over the registry API
    and pulls no layer.

    Args:
        transport: An open transport to the SSH host.
        device_type: The detected accelerator.
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


async def verify_image_signature(image: ResolvedImage, settings: Settings) -> None:
    """Authenticate the exact digest before any pull or launch; fail closed."""
    try:
        await sigstore_verify.verify_signature(
            image.digest_reference,
            identity_regexp=settings.cosign_certificate_identity_regexp,
            oidc_issuer=settings.cosign_oidc_issuer,
        )
    except (sigstore_verify.SignatureUnavailableError, sigstore_verify.SignatureVerificationError) as error:
        raise TrainerImageVerificationError(image.digest_reference, str(error)) from error


def _device_run_args(device_type: DeviceType, render_gid: str | None = None) -> list[str]:
    """Return the `docker run` flags that expose the accelerator.

    `render_gid`, when known, is added via `--group-add` so the container's
    fixed non-root user can read/write the render node - `--device /dev/dri`
    alone passes the node through but its group ownership still gates access.
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
    * `-p 127.0.0.1::<port>` publishes an OS-assigned ephemeral port bound
      only to loopback.
    * `--restart=no` - a crashed container must surface as a failed job, not
      restart silently.
    * Non-root, every capability dropped, no `--privileged`, and only the
      device nodes the configured accelerator needs are passed through.
    * `--read-only` root filesystem, a bounded `--tmpfs` for `/tmp` scratch,
      and a trainer-scoped data volume for datasets and model artifacts.

    `render_gid`, from `services.ssh.trainer_image.resolve_render_group_gid`,
    is required for a working XPU container: without it the fixed non-root
    user cannot open the render node and `torch.xpu.is_available()` reports
    zero devices. Ignored for CUDA.

    The `/tmp` tmpfs carries explicit `uid=`/`gid=` mount options matching
    `--user`, since an unqualified `--tmpfs` mounts root-owned. The data
    volume needs no such options: Docker initializes it with the image's
    directory ownership.
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
    """Pull the resolved image by digest, skipping a no-op re-pull.

    Checks the local image store first (Docker records the pulled digest as a
    `RepoDigest` regardless of which reference was used) before asking the
    daemon to re-fetch anything.

    If a background pull of the same tag is still in flight (started by a
    concurrent launch of another trainer on the same host), this waits for it
    with `pull_in_progress` rather than starting a second, concurrent pull of
    identical content, then re-checks the image store.

    Uses `settings.ssh_image_pull_timeout_s` rather than the default command
    timeout, which is sized for cheap probes and far too short for a
    multi-gigabyte transfer.

    Once the digest is confirmed present, prunes any other locally cached
    image of the same repository via `prune_stale_images` (best effort,
    never raised), so a long-lived server only accumulates the image each
    trainer actually needed.

    Raises:
        TrainerImagePullError: The pull failed, or did not finish within
            `settings.ssh_image_pull_timeout_s`.
    """
    if not await image_present_locally(transport, image.digest_reference):
        if await pull_in_progress(transport, image.tag_reference):
            await _await_background_pull(transport, image.tag_reference, settings)

        if not await image_present_locally(transport, image.digest_reference):
            result = await transport.run_command(
                ["docker", "pull", image.digest_reference], timeout=settings.ssh_image_pull_timeout_s
            )
            if not result.ok:
                raise TrainerImagePullError(image.digest_reference, detail=result.stderr or result.stdout or None)

    await prune_stale_images(transport, image)


# Substrings of a `docker rmi` failure that mean "already handled, not an
# error": the image is still referenced by a running/stopped container, or it
# is already gone (a concurrent prune on the same SSH host won the race).
_TOLERATED_RMI_FAILURES: Final = ("image is being used", "no such image")


async def prune_stale_images(transport: SshTransport, image: ResolvedImage) -> list[str]:
    """Remove other locally cached images of `image`'s repository, freeing disk.

    An SSH host otherwise keeps every distinct digest it has ever pulled
    for a `physicalai-trainer-<device>` repository, filling its disk from
    accumulated image history. Called after `pull_image` confirms
    `image.digest_reference` is present, so reclaiming here never costs a
    re-pull of the image a launch just verified it needs.

    Never touches `image.digest_reference` itself. Tolerates - without
    raising - a `docker rmi` refused because a container still references
    that image, and a failing listing command: reclaiming disk is
    best-effort bookkeeping, never something a launch should fail over.

    Args:
        transport: An open transport to the SSH host.
        image: The resolved image just confirmed present; its repository
            (everything before the ``@``) is what gets swept, and its digest
            is the one entry that is always kept.

    Returns:
        IDs of the images actually removed.
    """
    repository = image.digest_reference.rsplit("@", 1)[0]
    try:
        result = await transport.run_command(
            ["docker", "images", repository, "--digests", "--format", "{{.ID}} {{.Digest}}"]
        )
    except Exception as error:
        logger.warning("Could not list images for '{}' to prune stale ones: {}", repository, error)
        return []
    if not result.ok:
        return []

    stale_ids: set[str] = set()
    for line in result.stdout.splitlines():
        image_id, _, digest = line.partition(" ")
        image_id, digest = image_id.strip(), digest.strip()
        if image_id and digest != image.digest:
            stale_ids.add(image_id)

    removed: list[str] = []
    for image_id in stale_ids:
        try:
            rm_result = await transport.run_command(["docker", "rmi", image_id])
        except Exception as error:
            logger.warning("Could not remove stale image '{}': {}", image_id, error)
            continue
        if rm_result.ok:
            removed.append(image_id)
        elif not any(reason in (rm_result.stderr or "").lower() for reason in _TOLERATED_RMI_FAILURES):
            logger.warning("docker rmi reported a failure for stale image '{}': {}", image_id, rm_result.stderr)
    return removed


# Interval between polls of an in-flight background pull's PID.
_PULL_POLL_INTERVAL_S: Final = 5.0


async def _await_background_pull(transport: SshTransport, image_ref: str, settings: Settings) -> None:
    """Wait for a background pull of `image_ref` to finish, best effort.

    Bounded by `settings.ssh_image_pull_timeout_s`. Gives up silently rather
    than raising when the budget elapses - the caller re-checks the image
    store either way and falls back to its own pull.
    """
    deadline = time.monotonic() + settings.ssh_image_pull_timeout_s
    while time.monotonic() < deadline:
        if not await pull_in_progress(transport, image_ref):
            return
        await asyncio.sleep(_PULL_POLL_INTERVAL_S)


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


async def remove_stale_containers_on_port(
    transport: SshTransport, port: int, backend_instance_id: str, trainer_id: str
) -> list[str]:
    """Remove only this trainer's old containers already bound to `port`.

    A persistent SSH trainer always launches on a fixed, user-chosen loopback
    port, so a leftover container from an earlier attempt for the same
    trainer (a crash, a retried save, a deleted-then-recreated trainer)
    refuses `docker run` a second bind of that port. Only ever removes a
    container carrying this installation's own `MANAGED_LABEL` +
    `INSTANCE_LABEL` + `SERVER_LABEL`; another trainer or an unmanaged
    container holding the port is left alone, and `docker run` reports the
    collision instead of interrupting its job.

    Returns:
        The ids of any containers actually removed.
    """
    result = await transport.run_command(["docker", "ps", "-a", "--filter", f"publish={port}", "--format", "{{.ID}}"])
    if not result.ok:
        return []
    removed: list[str] = []
    for container_id in (line.strip() for line in result.stdout.splitlines() if line.strip()):
        inspection = await inspect_container(transport, container_id)
        if inspection is None:
            continue
        labels = inspection.labels
        owned = (
            labels.get(MANAGED_LABEL) == "true"
            and labels.get(INSTANCE_LABEL) == backend_instance_id
            and labels.get(SERVER_LABEL) == trainer_id
        )
        if not owned:
            continue
        await transport.run_command(["docker", "rm", "-f", container_id])
        removed.append(container_id)
    return removed


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
    """Create the trainer's data volume with management labels, tolerating an existing one.

    Created explicitly rather than letting `docker run --mount` auto-create it,
    so the volume carries the management labels this module relies on to
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
