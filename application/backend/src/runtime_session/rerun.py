"""Best-effort Rerun visualization, independent of robot and inference loops.

Each model run owns a recording and a thread that dispatches events to Runtime's
RerunCallback. Actions are targets, not the robot worker's interpolated commands;
images are the latest store frames, not necessarily those used by an inference.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from physicalai.capture.frame import Frame
from physicalai.runtime import RerunCallback
from physicalai.runtime.events import InferenceEvent, LifecycleEvent, MetricsEvent, TickEvent

from settings import RerunSettings
from workers.base import ThreadWorker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from rerun import RecordingStream

    from runtime_session.store import Store
    from schemas.runtime_session import ErrorMessage, StateMessage

_SPAWN_LOCK = threading.Lock()
_BACKGROUND_SERVERS: dict[int, _BackgroundServer] = {}


@dataclass
class _BackgroundServer:
    process: subprocess.Popen[bytes]
    users: int = 0


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.1):
            return True
    except OSError:
        return False


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@contextmanager
def _background_server(port: int) -> Iterator[None]:
    """Share a loopback-only headless server; stop ours when its last session closes."""
    with _SPAWN_LOCK:
        server = _BACKGROUND_SERVERS.get(port)
        if server is None and not _listening(port):
            process = subprocess.Popen(  # noqa: S603 - fixed executable and flags; port is validated
                [
                    str(files("rerun_cli").joinpath("rerun.exe" if sys.platform == "win32" else "rerun")),
                    "--serve-grpc",
                    "--bind",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--server-memory-limit",
                    "256MiB",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                deadline = time.monotonic() + 10
                while not _listening(port):
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(f"Rerun background server could not start on port {port}")
                    time.sleep(0.05)
            except BaseException:
                _stop_server(process)
                raise
            server = _BackgroundServer(process)
            _BACKGROUND_SERVERS[port] = server
        if server is not None:
            server.users += 1
    try:
        yield
    finally:
        with _SPAWN_LOCK:
            if server is not None:
                server.users -= 1
                if server.users == 0:
                    _stop_server(server.process)
                    del _BACKGROUND_SERVERS[port]


@dataclass(frozen=True)
class _ModelEvents:
    observer: RerunObserver
    hz: float
    ready_at: float

    def emit_inference(self, event: InferenceEvent) -> None:
        # Executors have already consumed the offset; plot only the remaining actions.
        chunk = event.chunk[event.offset :]
        self.observer._events.append(
            replace(event, chunk=chunk.copy(), offset=0, timestamp=max(event.timestamp + 1 / self.hz, self.ready_at))
        )

    def emit_metrics(self, event: MetricsEvent) -> None:
        self.observer._events.append(event)

    def emit_tick(self, event: TickEvent) -> None:
        self.observer._events.append(event)


class _SessionRerunCallback(RerunCallback):
    """Use Runtime's plots without its process-global recording initialization.

    The observer owns the connection and binds its RecordingStream to its thread.
    Initialization and prediction timing adapt to Studio; all rendering stays in RerunCallback.
    """

    def _init_rerun(self, session_id: str, metadata: dict[str, Any]) -> None:  # noqa: ARG002
        self._fps = metadata["fps"]
        self._multiplier = 1
        self._joint_names = metadata["joint_names"]
        self._camera_names = metadata["camera_names"]
        self._tick_timestamp = time.time()
        self._initialized = True
        self._send_series_styles()
        self._send_default_blueprint()

    def on_tick(self, event: TickEvent) -> None:
        """Keep the wall/step correspondence for Studio's initial action grace period."""
        self._tick_timestamp = event.timestamp
        super().on_tick(event)

    def on_inference(self, event: InferenceEvent) -> None:
        """Align predictions with the delayed targets without duplicating their rendering."""
        last_step = self._last_step
        self._last_step += max(0, round((event.timestamp - self._tick_timestamp) * self._fps) - 1)
        try:
            super().on_inference(event)
        finally:
            self._last_step = last_step


class RerunObserver(ThreadWorker[None]):
    """Dispatch control events to RerunCallback off the control and API loops."""

    def __init__(self, session_id: str, *, settings: RerunSettings | None = None, hz: float = 30) -> None:
        super().__init__(name=f"RerunObserver-{session_id}")
        self.session_id = session_id
        self.settings = settings or RerunSettings()
        self._hz = hz
        self._lock = threading.Lock()
        self._store: Store | None = None
        self._next_images = 0.0
        self._events: deque[InferenceEvent | MetricsEvent | TickEvent | StateMessage | ErrorMessage] = deque(maxlen=64)
        self._recording: RecordingStream
        self._callback: _SessionRerunCallback

    @property
    def viewer_url(self) -> str:
        """Return the matching hosted viewer URL using this session's startup settings."""
        import rerun as rr

        stream_url = (
            self.settings.connect_url
            if self.settings.mode == "connect"
            else f"rerun+http://127.0.0.1:{self.settings.spawn_port}/proxy"
        )
        return f"https://app.rerun.io/version/{rr.__version__}/?{urlencode({'url': stream_url})}"

    def set_store(self, store: Store | None) -> None:
        """Detach before the session unlinks a store; only copying samples holds the lock."""
        with self._lock:
            self._store = store

    def control_events(self, hz: float, grace_s: float) -> _ModelEvents:
        """Return a nonblocking sink for PolicySource's existing execution events."""
        return _ModelEvents(self, hz, time.time() + grace_s)

    def emit_session(self, event: StateMessage | ErrorMessage) -> None:
        """Log session changes without doing visualization work on the API loop."""
        self._events.append(event)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[None]:
        import rerun as rr

        self._recording = rr.RecordingStream("physicalai-studio", recording_id=self.session_id)
        with ExitStack() as stack:
            try:
                if self.settings.mode == "connect":
                    self._recording.connect_grpc(self.settings.connect_url)
                elif self.settings.open_viewer:
                    # The SDK reuses a viewer already listening on this port.
                    with _SPAWN_LOCK:
                        rr.spawn(port=self.settings.spawn_port, recording=self._recording)
                else:
                    stack.enter_context(_background_server(self.settings.spawn_port))
                    self._recording.connect_grpc(f"rerun+http://127.0.0.1:{self.settings.spawn_port}/proxy")
                self._callback = _SessionRerunCallback(image_decimation=1, image_jpeg_quality=80)
                try:
                    # Module-level rr.log calls in RerunCallback resolve to this thread's
                    # recording, never a global shared with another runtime session.
                    with self._recording:
                        self._callback.on_lifecycle(
                            LifecycleEvent(self.session_id, time.time(), "start", self._metadata())
                        )
                        try:
                            yield
                        finally:
                            try:
                                self._drain_events()
                            finally:
                                self._callback.on_lifecycle(LifecycleEvent(self.session_id, time.time(), "stop", {}))
                finally:
                    self._callback.close()
            finally:
                self._recording.disconnect()

    async def run_loop(self, context: None) -> None:  # noqa: ARG002
        while not self.should_stop():
            self._drain_events()
            await self.sleep(1 / 30)

    def _metadata(self) -> dict[str, Any]:
        with self._lock:
            store = self._store
            return {
                "fps": self._hz,
                "joint_names": list(store.spec.joint_names) if store is not None else [],
                "camera_names": list(store.spec.observations) if store is not None else [],
            }

    def _camera_frames(self) -> dict[str, Frame]:
        now = time.monotonic()
        if now < self._next_images:
            return {}
        self._next_images = now + 0.1
        # Store.read_observation copies the pixels before releasing this lock.
        # Compression and plotting never hold a shared-memory view.
        with self._lock:
            store = self._store
            if store is None:
                return {}
            return {
                key: Frame(sample.value, sample.timestamp, 0)
                for key in store.spec.observations
                if (sample := store.read_observation(key)) is not None
            }

    def _drain_events(self) -> None:
        import rerun as rr

        # Bounded even when producers are faster than the sampler.
        for _ in range(64):
            try:
                event = self._events.popleft()
            except IndexError:
                break
            if isinstance(event, TickEvent):
                self._callback.on_tick(replace(event, camera_frames=self._camera_frames()))
            elif isinstance(event, MetricsEvent):
                self._callback.on_metrics(event)
            elif isinstance(event, InferenceEvent):
                self._callback.on_inference(event)
            else:
                rr.log("runtime/session", rr.TextLog(event.model_dump_json()))
