# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stable identity for this Studio installation, used to own SSH-provisioned containers.

Remote servers are global: two Studio installations can legitimately target the
same host, and a container's management labels alone cannot distinguish "mine,
now orphaned" from "a different, still-active installation's job". A container
is also labeled with the *owning* installation's id, generated once and
persisted to disk so it survives process restarts (the orphan sweep runs after
every restart) but is never shared between installations, e.g. by copying a
database between machines - copying the data directory copies this file too,
which is exactly the "same installation" case the orphan sweep needs to widen.
"""

import os
import uuid
from pathlib import Path
from threading import Lock

from settings import get_settings

_BACKEND_INSTANCE_ID_FILENAME = "backend_instance_id"

_cached_backend_instance_id: str | None = None
_lock = Lock()


def _backend_instance_id_path() -> Path:
    return get_settings().data_dir / _BACKEND_INSTANCE_ID_FILENAME


def _read_existing(path: Path) -> str | None:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return str(uuid.UUID(content))
    except ValueError:
        # Corrupt or foreign content: treat as absent rather than adopting it,
        # so a bad file can never be silently used as an ownership marker.
        return None


def _write_atomic(path: Path, content: str) -> None:
    """Write `content` to `path` via a temp file and atomic replace.

    A crash mid-write must never leave a partial file at `path`: `_read_existing`
    treats unparseable content as absent and would otherwise mint a fresh id on
    the next start, orphaning this installation's own containers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def get_backend_instance_id() -> str:
    """Return this installation's stable backend instance id, creating it once.

    Cached in-process after the first call within a run; re-read from disk on
    a fresh process so a restart keeps the same id and can reclaim its own
    still-running containers.
    """
    global _cached_backend_instance_id  # noqa: PLW0603 - process-wide identity, intentionally a singleton.
    if _cached_backend_instance_id is not None:
        return _cached_backend_instance_id

    with _lock:
        if _cached_backend_instance_id is not None:
            return _cached_backend_instance_id

        path = _backend_instance_id_path()
        existing = _read_existing(path)
        if existing is not None:
            _cached_backend_instance_id = existing
            return _cached_backend_instance_id

        generated = str(uuid.uuid4())
        _write_atomic(path, generated)
        _cached_backend_instance_id = generated
        return _cached_backend_instance_id


def reset_backend_instance_id_cache() -> None:
    """Drop the in-process cache. Test-support only."""
    global _cached_backend_instance_id  # noqa: PLW0603 - test-support reset of the process-wide singleton.
    _cached_backend_instance_id = None
