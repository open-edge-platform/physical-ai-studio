"""Coordinate one USB host helper without exposing a listener on the user's computer."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from core.security import is_loopback_host
from exceptions import BaseException as APIError
from schemas.runtime_session import ModelControlConfig
from schemas.streamdeck import StreamDeckHeartbeat, StreamDeckStatus, StreamDeckTarget
from settings import get_settings

if TYPE_CHECKING:
    from runtime_session.registry import RuntimeSessionRegistry


class StreamDeckService:
    """An expiring helper lease; a reconnect never silently reattaches to a workspace."""

    LEASE_SECONDS = 15.0

    def __init__(self) -> None:
        self.heartbeat: StreamDeckHeartbeat | None = None
        self.last_seen = 0.0
        self.target: StreamDeckTarget | None = None
        self.process: subprocess.Popen[bytes] | None = None

    @staticmethod
    def native_launch_available() -> bool:
        """Never spawn a helper inside Docker or on a network-exposed backend."""
        return (
            is_loopback_host(get_settings().host)
            and not Path("/.dockerenv").exists()
            and not Path("/run/.containerenv").exists()
            and StreamDeckService.bridge_path().is_file()
            and shutil.which("uv") is not None
        )

    @staticmethod
    def bridge_path() -> Path:
        """Locate the repository helper without accepting a path from the browser."""
        return Path(__file__).resolve().parents[3] / "streamdeck" / "streamdeck_bridge.py"

    def start(self) -> None:
        """Start at most one owned helper, using only local backend configuration."""
        if not self.native_launch_available():
            raise APIError(
                "Use the installed host launcher for this Studio deployment.", "streamdeck_host_launcher", 409
            )
        if self.process is not None and self.process.poll() is None:
            return
        settings = get_settings()
        host = f"[{settings.host}]" if ":" in settings.host else settings.host
        port = int(os.environ.get("HTTP_SERVER_PORT", settings.port))
        log_path = settings.log_dir / "streamdeck-helper.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        with log_path.open("ab") as log:
            self.process = subprocess.Popen(  # noqa: S603 - fixed executable, script and locally configured URL
                [
                    shutil.which("uv") or "uv",
                    "run",
                    "--no-project",
                    str(self.bridge_path()),
                    "--url",
                    f"http://{host}:{port}",
                ],
                cwd=self.bridge_path().parent,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=sys.platform != "win32",
                creationflags=creationflags,
            )

    def close(self) -> None:
        """Stop the owned helper tree at shutdown; never stop an independently launched helper."""
        if self.process is None or self.process.poll() is not None:
            return
        if sys.platform == "win32":
            subprocess.run(  # noqa: S603 - PID is from our own child process
                ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],  # noqa: S607 - Windows system utility
                check=False,
                timeout=5,
                capture_output=True,
            )
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)

    def status(self, registry: RuntimeSessionRegistry) -> StreamDeckStatus:
        """Expire disconnected helpers and targets whose owning browser has closed."""
        if self.heartbeat is not None and time.monotonic() - self.last_seen >= self.LEASE_SECONDS:
            self.heartbeat = None
            self.target = None
        if self.target is not None:
            session = registry.get(self.target.session_id)
            if session is None or session.closed.is_set():
                self.target = None
        report = self.heartbeat
        return StreamDeckStatus(
            helper_online=report is not None,
            helper_starting=report is None and self.process is not None and self.process.poll() is None,
            connected=bool(
                report and self.target and report.device and report.connection_id == self.target.connection_id
            ),
            device=report.device if report else None,
            error=report.error
            if report
            else (
                f"Helper exited. See {get_settings().log_dir / 'streamdeck-helper.log'}."
                if self.process is not None and self.process.poll() is not None
                else None
            ),
            target=self.target,
            dictation_status=report.dictation_status
            if report and self.target and report.connection_id == self.target.connection_id
            else "idle",
        )

    def report(self, heartbeat: StreamDeckHeartbeat, registry: RuntimeSessionRegistry) -> StreamDeckStatus:
        """Accept only one live helper, preventing another host from taking over its deck."""
        self.status(registry)
        if self.heartbeat is not None and self.heartbeat.helper_id != heartbeat.helper_id:
            raise APIError("Another Stream Deck helper is already running.", "streamdeck_busy", 409)
        self.heartbeat = heartbeat
        self.last_seen = time.monotonic()
        return self.status(registry)

    def require_target(
        self, session_id: str, connection_id: str, registry: RuntimeSessionRegistry, *, page: str
    ) -> StreamDeckTarget:
        """Fence detached helpers and keys queued on a different page before dispatch."""
        target = self.status(registry).target
        if (
            target is None
            or target.session_id != session_id
            or target.connection_id != connection_id
            or target.page != page
        ):
            raise APIError(
                "Stream Deck was disconnected or changed page; command not sent.", "streamdeck_detached", 409
            )
        return target

    def connect(
        self,
        session_id: str,
        task: str,
        registry: RuntimeSessionRegistry,
        *,
        page: str = "record",
        inference: ModelControlConfig | None = None,
    ) -> StreamDeckStatus:
        """Attach explicitly, or update the task for the workspace already attached."""
        self.status(registry)
        if self.heartbeat is None:
            raise APIError("Start the Stream Deck helper on the USB host first.", "streamdeck_offline", 409)
        if self.target is not None and self.target.session_id != session_id:
            raise APIError("Disconnect Stream Deck in the other workspace first.", "streamdeck_busy", 409)
        settings = {"task": task, "page": page, "inference": inference or ModelControlConfig()}
        if self.target is None or self.target.page != page:
            self.target = StreamDeckTarget(session_id=session_id, connection_id=uuid4().hex, **settings)
        else:
            self.target = self.target.model_copy(update=settings)
        return self.status(registry)

    def disconnect(self, session_id: str, registry: RuntimeSessionRegistry) -> StreamDeckStatus:
        """Only the attached workspace can release its target; in-flight commands are not undone."""
        self.status(registry)
        if self.target is not None and self.target.session_id == session_id:
            self.target = None
        return self.status(registry)
