from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

KEY_PREFIX = "studio/rt"
# physicalai's pinned robot transport uses 20000-59999. Keep Studio's
# deterministic range disjoint so sessions cannot collide with any robot owner.
_PORT_BASE = 10000
_PORT_RANGE = 10000
_SESSION_NAME_RE = re.compile(r"^rt-[A-Za-z0-9_-]+$")


def runtime_session_name(follower_id: UUID | str) -> str:
    """Return the runtime-session identity for a follower robot."""
    return validate_session_name(f"rt-{follower_id}")


def validate_session_name(name: str) -> str:
    """Validate one load-bearing ``rt-`` session key segment."""
    if not _SESSION_NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid runtime session name {name!r}: expected 'rt-' followed by letters, digits, '_' or '-'"
        )
    return name


def session_prefix(name: str) -> str:
    """Return the key prefix for one runtime session."""
    return f"{KEY_PREFIX}/{validate_session_name(name)}"


def metadata_key(name: str) -> str:
    """Return the session metadata key."""
    return f"{session_prefix(name)}/metadata"


def command_key(name: str) -> str:
    """Return the idempotent-command key."""
    return f"{session_prefix(name)}/command"


def request_key(name: str) -> str:
    """Return the acknowledged-command key."""
    return f"{session_prefix(name)}/request"


def tick_key(name: str) -> str:
    """Return the observation telemetry key."""
    return f"{session_prefix(name)}/tick"


def state_key(name: str) -> str:
    """Return the state telemetry key."""
    return f"{session_prefix(name)}/state"


def error_key(name: str) -> str:
    """Return the error telemetry key."""
    return f"{session_prefix(name)}/error"


def lifecycle_key(name: str) -> str:
    """Return the lifecycle telemetry key."""
    return f"{session_prefix(name)}/lifecycle"


def _port_for_prefix(prefix: str) -> int:
    digest = hashlib.sha256(prefix.encode()).digest()
    return _PORT_BASE + int.from_bytes(digest[:4], "big") % _PORT_RANGE


def derive_endpoint_port(name: str) -> int:
    """Derive a Studio port outside physicalai's robot range."""
    return _port_for_prefix(f"{KEY_PREFIX}/{validate_session_name(name)}")
