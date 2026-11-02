"""Host-local exclusivity lock for Studio runtime sessions.

Same mechanism as physicalai's robot name lock — a non-blocking ``flock`` on a
user-private file, released by the kernel even on ``SIGKILL`` — in a separate
directory so the two namespaces cannot collide on disk. The identity is the
``rt-<follower uuid>`` session name, never the robot's own SharedRobot name.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Self

from runtime.transport.ids import validate_session_name

SESSION_LOCK_KIND = "rt-name"


def _lock_dir() -> Path:
    """Return the user-scoped Studio runtime-lock directory, creating it if needed.

    Prefers ``$XDG_RUNTIME_DIR/physicalai-studio/runtime-locks``. When that
    variable is unset (some containers), falls back to
    ``<tempdir>/physicalai-studio-<uid>/runtime-locks``.

    Raises:
        RuntimeError: If the resulting directory is not private or is not
            owned by the current user.
    """
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        lock_dir = Path(runtime_dir) / "physicalai-studio" / "runtime-locks"
    else:
        lock_dir = Path(tempfile.gettempdir()) / f"physicalai-studio-{os.getuid()}" / "runtime-locks"

    lock_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    lock_dir_stat = lock_dir.stat()
    if not stat.S_ISDIR(lock_dir_stat.st_mode) or lock_dir_stat.st_uid != os.getuid() or lock_dir_stat.st_mode & 0o077:
        raise RuntimeError(f"runtime lock directory must be private and owned by this user: {lock_dir}")
    return lock_dir


def session_lock_path(identity: str) -> Path:
    """Return the lock-file path for one runtime session identity.

    Hashing keeps filesystem-unsafe characters out of the filename and keeps
    this namespace distinct from physicalai's robot locks even when the raw
    identity strings happen to match.
    """
    digest = hashlib.sha256(f"{SESSION_LOCK_KIND}:{identity}".encode()).hexdigest()
    return _lock_dir() / f"{digest}.lock"


def _read_live_diagnostics(path: Path) -> dict[str, object] | None:
    """Parse and liveness-check one lock file.

    Lock files are never deleted, so a well-formed file is not enough: the
    recorded pid must still exist and the flock must still be held.
    """
    try:
        diagnostics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(diagnostics, dict) or diagnostics.get("kind") != SESSION_LOCK_KIND:
        return None

    pid = diagnostics.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        # Signal 0 sends nothing; the kernel only checks that the pid exists
        # and is signalable. Valid here because the trust model is a single host.
        os.kill(pid, 0)
    except OSError:
        return None

    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    held = False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        held = True
    finally:
        os.close(fd)
    return diagnostics if held else None


def registered_session_names() -> list[str]:
    """Return session names whose lock file records a live, currently held flock.

    Files are retained after release, so both the pid check and the flock
    check are required. The result is only a set of candidates: callers still
    confirm liveness through the session's metadata queryable.
    """
    names: set[str] = set()
    for path in _lock_dir().glob("*.lock"):
        diagnostics = _read_live_diagnostics(path)
        if diagnostics is None:
            continue
        identity = diagnostics.get("identity")
        if isinstance(identity, str):
            names.add(identity)
    return sorted(names)


def live_session_pid(name: str) -> int | None:
    """Return the holder's pid for a live session, or ``None``.

    This is the cheap first half of the deletion guard: a filesystem read that
    misses without opening a Zenoh session.
    """
    diagnostics = _read_live_diagnostics(session_lock_path(name))
    if diagnostics is None or diagnostics.get("identity") != name:
        return None
    pid = diagnostics.get("pid")
    return pid if isinstance(pid, int) else None


class SessionNameLock:
    """Exclusive, non-blocking advisory lock on one runtime session name.

    Held for the child's lifetime and released on process exit even after a
    crash (``flock`` locks die with the file descriptor). Lock files are
    never deleted on release — the kernel lock, not file existence,
    determines ownership.
    """

    def __init__(self, identity: str) -> None:
        self._identity = validate_session_name(identity)
        self._path = session_lock_path(self._identity)
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        """The lock-file path."""
        return self._path

    @property
    def identity(self) -> str:
        """The runtime session name this lock guards."""
        return self._identity

    def acquire(self) -> bool:
        """Try to acquire the lock without blocking.

        Returns:
            True if this process now holds the lock, False if another
            process already holds it.
        """
        if self._fd is not None:
            return True
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        diagnostics = {
            "kind": SESSION_LOCK_KIND,
            "identity": self._identity,
            "pid": os.getpid(),
            "created_at": time.time(),
        }
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(diagnostics).encode())
        self._fd = fd
        return True

    def release(self) -> None:
        """Release the lock. No-op when not held."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def __enter__(self) -> Self:
        """Acquire on entry.

        Raises:
            RuntimeError: If the lock is already held by another process.
        """
        if not self.acquire():
            raise RuntimeError(f"runtime session lock already held: {self._identity}")
        return self

    def __exit__(self, *args: object) -> None:
        """Release on exit."""
        self.release()
