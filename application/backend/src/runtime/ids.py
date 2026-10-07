from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

_SESSION_NAME_RE = re.compile(r"^rt-[A-Za-z0-9_-]+$")


def runtime_session_name(follower_id: UUID | str) -> str:
    """Return the runtime-session identity for a follower robot."""
    return validate_session_name(f"rt-{follower_id}")


def validate_session_name(name: str) -> str:
    """Validate an ``rt-`` runtime session name."""
    if not _SESSION_NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid runtime session name {name!r}: expected 'rt-' followed by letters, digits, '_' or '-'"
        )
    return name
