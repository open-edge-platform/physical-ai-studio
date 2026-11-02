"""Shared SSH trainer-image helpers."""

from __future__ import annotations

import hashlib
from typing import Final

from schemas.hardware import DeviceType
from services.ssh.transport import SshTransport

DEFAULT_PROTOCOL_VERSION: Final = 1
PROTOCOL_LABEL: Final = "org.open-edge-platform.physicalai.trainer.api-protocol"


def trainer_image_ref(registry: str, device_type: DeviceType, tag: str) -> str:
    """Build the trainer image reference for an accelerator and tag."""
    return f"{registry.rstrip('/')}/physicalai-trainer-{device_type.value}:{tag}"


def protocol_tag(protocol_version: int) -> str:
    """Return the protocol-pinned trainer image tag."""
    return f"protocol-{protocol_version}"


async def resolve_render_group_gid(transport: SshTransport) -> str | None:
    """Return the host GID for the first Intel render node, if present."""
    result = await transport.run_command(
        ["sh", "-c", "stat -c %g $(ls /dev/dri/renderD* 2>/dev/null | head -n1) 2>/dev/null"]
    )
    gid = result.first_line().strip()
    return gid or None


async def image_present_locally(transport: SshTransport, image_ref: str) -> bool:
    """Return whether the trainer image is already in the remote image store."""
    return (await transport.run_command(["docker", "image", "inspect", image_ref])).ok


def _pull_pidfile(image_ref: str) -> str:
    digest = hashlib.sha256(image_ref.encode()).hexdigest()[:16]
    return f"/tmp/physicalai-pull-{digest}.pid"  # noqa: S108 # nosec B108


async def pull_in_progress(transport: SshTransport, image_ref: str) -> bool:
    """Return whether a prior background image pull is still running."""
    return (
        await transport.run_command(
            [
                "sh",
                "-c",
                'test -f "$1" && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null',
                "sh",
                _pull_pidfile(image_ref),
            ]
        )
    ).ok
