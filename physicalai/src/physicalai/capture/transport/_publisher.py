# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory camera publisher transport based on iceoryx2."""

from __future__ import annotations

import json
import select
import subprocess  # noqa: S404
import sys
from typing import TYPE_CHECKING, Self

from physicalai.capture.errors import CaptureError

if TYPE_CHECKING:
    from physicalai.capture.transport._spec import CameraSpec


class CameraPublisher:
    """Publishes camera frames to shared memory via iceoryx2.

    Spawns a detached subprocess that owns the camera device and
    publishes frames.  The subprocess outlives the parent process and
    self-terminates via idle timeout when zero subscribers remain.

    Args:
        spec: Camera construction specification.
        service_name: iceoryx2 service name for the pub-sub channel.
        idle_timeout: Seconds with zero subscribers before self-exit.
        max_subscribers: Maximum concurrent subscribers.
    """

    def __init__(
        self,
        spec: CameraSpec,
        service_name: str,
        *,
        idle_timeout: float = 5,
        max_subscribers: int = 64,
        _factory_override: str | None = None,
    ) -> None:
        self._spec = spec
        self._service_name = service_name
        self._idle_timeout = idle_timeout
        self._max_subscribers = max_subscribers
        self._factory_override = _factory_override
        self._process: subprocess.Popen[bytes] | None = None

    def start(self, timeout: float = 10.0) -> None:
        """Start the publisher subprocess.

        Args:
            timeout: Maximum seconds to wait for the subprocess to
                report readiness.

        Raises:
            CaptureError: If the subprocess fails to start or does not
                become ready within *timeout* seconds.
        """
        if self.is_alive:
            return

        config: dict = {
            "camera_type": self._spec.camera_type,
            "camera_kwargs": self._spec.camera_kwargs,
            "service_name": self._service_name,
            "idle_timeout": self._idle_timeout,
            "max_subscribers": self._max_subscribers,
        }
        if self._factory_override is not None:
            config["_factory_override"] = self._factory_override

        self._process = subprocess.Popen(
            [sys.executable, "-m", "physicalai.capture.transport._publisher_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Detach into a new session so the worker survives when the
            # parent exits / Ctrl+C's. Subsequent subscribers re-attach
            # by service name.
            start_new_session=True,
        )
        assert self._process.stdin is not None  # noqa: S101
        self._process.stdin.write(json.dumps(config).encode())
        self._process.stdin.close()

        line = self._read_stdout_line(timeout)
        if line is None:
            self.stop()
            msg = f"camera publisher did not become ready within {timeout:.1f}s"
            raise CaptureError(msg)
        if line.startswith("ERROR:"):
            self.stop()
            error_msg = json.loads(line[len("ERROR:") :])
            msg = f"failed to start camera publisher: {error_msg}"
            raise CaptureError(msg)
        if line != "READY":
            self.stop()
            msg = f"unexpected publisher response: {line!r}"
            raise CaptureError(msg)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the publisher subprocess.

        Sends SIGTERM, waits up to *timeout* seconds, then SIGKILL.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown.
        """
        proc = self._process
        if proc is None or proc.poll() is not None:
            self._process = None
            return

        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1)

        self._process = None

    @property
    def is_alive(self) -> bool:
        """Whether the publisher subprocess is running."""
        return self._process is not None and self._process.poll() is None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def _read_stdout_line(self, timeout: float) -> str | None:
        """Read one line from the subprocess stdout with a timeout.

        Args:
            timeout: Maximum seconds to wait for a line.

        Returns:
            The stripped line, or ``None`` on timeout / EOF.
        """
        proc = self._process
        if proc is None or proc.stdout is None:
            return None

        # select.select doesn't work on Windows pipes;
        # a thread-based fallback would be needed for Windows support.
        readable, _, _ = select.select([proc.stdout], [], [], timeout)
        if not readable:
            return None

        raw = proc.stdout.readline()
        if not raw:
            return None
        return raw.decode().strip()
