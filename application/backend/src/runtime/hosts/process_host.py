from __future__ import annotations

import contextlib
import json
import os
import select
import subprocess  # nosec: B404
import sys
import threading
from typing import Any

from loguru import logger

from exceptions import BaseException as AppBaseException

_DEFAULT_START_TIMEOUT = 30.0


class RuntimeProcessHost:
    """Spawn and supervise one detached RuntimeSession worker."""

    def __init__(
        self,
        name: str,
        document: dict[str, Any],
        *,
        follower_name: str | None = None,
        leader_name: str | None = None,
        start_timeout: float = _DEFAULT_START_TIMEOUT,
    ) -> None:
        self._session_name = name
        self._document = document
        self._follower_name = follower_name
        self._leader_name = leader_name
        self._start_timeout = start_timeout
        self._proc: subprocess.Popen[str] | None = None
        self._error: AppBaseException | None = None
        self._stop_requested = False
        self._lock = threading.Lock()

    @property
    def error(self) -> AppBaseException | None:
        """Return a worker startup failure that Zenoh could not report."""
        return self._error

    @property
    def exited_cleanly(self) -> bool:
        """Return whether the worker exited with a successful status."""
        return self._proc is not None and self._proc.poll() == 0

    @property
    def pid(self) -> int | None:
        """Return the worker process id after it has been spawned."""
        return None if self._proc is None else self._proc.pid

    def start(self) -> None:
        """Spawn the worker and wait for its one-line startup handshake."""
        payload = json.dumps(
            {
                "session_name": self._session_name,
                "document": self._document,
                "follower_name": self._follower_name,
                "leader_name": self._leader_name,
                "parent_pid": os.getpid(),
            }
        )
        with self._lock:
            if self._proc is not None:
                if self._proc.poll() is None:
                    return
                raise RuntimeError("Runtime session host cannot be started more than once")
            self._error = None
            # B603 suppressed: argv is the active interpreter plus a hardcoded
            # internal module. shell=True is not used, so it has no shell-injection
            # surface. Configuration is sent as JSON on stdin instead of argv.
            # Popen does not inherit sys.path, so preserve pytest-only imports such
            # as tests.runtime.fakes for the detached worker.
            self._proc = subprocess.Popen(  # noqa: S603 # nosec: B603
                [sys.executable, "-m", "runtime.hosts.session_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                start_new_session=True,
                env={**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)},
                text=True,
            )
            proc = self._proc
            stop_requested = self._stop_requested

        if stop_requested:
            self._close_stdin(proc)
            self._close_stdout(proc)
            self.stop()
            return

        try:
            assert proc.stdin is not None  # noqa: S101
            proc.stdin.write(payload)
            proc.stdin.close()
        except (OSError, ValueError) as exc:
            with contextlib.suppress(OSError, ValueError):
                if proc.stdin is not None:
                    proc.stdin.close()
            if self._is_stop_requested():
                return
            self.stop()
            raise AppBaseException(
                message="Failed to start the runtime session process.",
                error_code="robot_connection_failed",
                http_status=500,
            ) from exc

        if self._is_stop_requested():
            self.stop()
            return

        line = self._read_stdout_line(proc, self._start_timeout)
        self._close_stdout(proc)
        if self._is_stop_requested():
            return
        if line is None:
            self.stop()
            raise AppBaseException(
                message=f"Runtime session {self._session_name} did not become ready within {self._start_timeout:.1f}s.",
                error_code="robot_connection_failed",
                http_status=500,
            )
        if line.startswith("ERROR:"):
            self._error = self._error_from_line(line)
            self.stop()
            raise self._error
        if line != "READY":
            self.stop()
            raise AppBaseException(
                message=f"Runtime session worker returned an unexpected startup response: {line!r}",
                error_code="robot_connection_failed",
                http_status=500,
            )

    def is_alive(self) -> bool:
        """Return whether the detached worker is still running."""
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker gracefully, then kill it if it does not exit."""
        with self._lock:
            self._stop_requested = True
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return

        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Runtime session process {} did not stop within {}s, killing", proc.pid, timeout)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=1.0)

    def kill(self) -> None:
        """Immediately kill the worker process."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker, matching multiprocessing.Process.join semantics."""
        proc = self._proc
        if proc is None:
            return
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=timeout)

    def _is_stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    @staticmethod
    def _close_stdout(proc: subprocess.Popen[str]) -> None:
        if proc.stdout is not None:
            with contextlib.suppress(OSError, ValueError):
                proc.stdout.close()

    @staticmethod
    def _close_stdin(proc: subprocess.Popen[str]) -> None:
        if proc.stdin is not None:
            with contextlib.suppress(OSError, ValueError):
                proc.stdin.close()

    @staticmethod
    def _read_stdout_line(proc: subprocess.Popen[str], timeout: float) -> str | None:
        if proc.stdout is None:
            return None
        # select.select does not work with Windows pipes. A thread-based reader
        # is needed before this detached host can support Windows.
        readable, _, _ = select.select([proc.stdout], [], [], timeout)
        if not readable:
            return None
        line = proc.stdout.readline()
        return line.strip() or None

    @staticmethod
    def _error_from_line(line: str) -> AppBaseException:
        try:
            payload = json.loads(line.removeprefix("ERROR:"))
            message = payload["msg"]
            error_code = payload["error_code"]
            if not isinstance(message, str) or not isinstance(error_code, str):
                raise TypeError("msg and error_code must be strings")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AppBaseException(
                message=f"Runtime session worker returned a malformed startup error: {line!r}",
                error_code="robot_connection_failed",
                http_status=500,
            ) from exc
        return AppBaseException(message=message, error_code=error_code, http_status=500)
