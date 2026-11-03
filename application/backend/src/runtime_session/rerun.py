"""Best-effort Rerun visualization, independent of robot and inference loops.

Each session owns a recording and a sampling thread. Store actions are targets,
not the robot worker's interpolated commands; images are the latest store frames,
not necessarily the frames used by an inference.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import numpy as np
import psutil

from runtime_session.store import ACTION, STATE
from settings import RerunSettings
from workers.base import ThreadWorker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from physicalai.runtime.events import InferenceEvent, MetricsEvent
    from rerun import RecordingStream

    from runtime_session.store import FeatureSpec, Store
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
        self.observer._events.append((event, self.hz, self.ready_at, time.monotonic()))

    def emit_metrics(self, event: MetricsEvent) -> None:
        self.observer._events.append((event, self.hz, self.ready_at, time.monotonic()))


class RerunObserver(ThreadWorker[None]):
    """Sample a session at 30 Hz, with images capped at 10 Hz, off the API loop."""

    def __init__(self, session_id: str, *, settings: RerunSettings | None = None) -> None:
        super().__init__(name=f"RerunObserver-{session_id}")
        self.session_id = session_id
        self.settings = settings or RerunSettings()
        self._origin = time.monotonic()
        self._lock = threading.Lock()
        self._store: Store | None = None
        self._last_store: Store | None = None
        self._sent: dict[str, float] = {}
        self._next_images = 0.0
        self._events: deque[tuple[InferenceEvent | MetricsEvent | StateMessage | ErrorMessage, float, float, float]] = (
            deque(maxlen=64)
        )
        self._latencies: deque[float] = deque(maxlen=200)
        self._recording: RecordingStream
        self._process: psutil.Process | None = None
        self._next_system = 0.0

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
        return _ModelEvents(self, hz, time.monotonic() + grace_s)

    def emit_session(self, event: StateMessage | ErrorMessage) -> None:
        """Log session changes without doing visualization work on the API loop."""
        self._events.append((event, 0.0, 0.0, time.monotonic()))

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
                self._recording.log("runtime/lifecycle", rr.TextLog("Session started"))
                yield
            finally:
                try:
                    self._drain_events()
                    self._recording.set_time("session", duration=time.monotonic() - self._origin)
                    self._recording.log("runtime/lifecycle", rr.TextLog("Session stopped"))
                finally:
                    self._recording.disconnect()

    async def run_loop(self, context: None) -> None:  # noqa: ARG002
        while not self.should_stop():
            self._sample()
            self._drain_events()
            self._sample_system()
            await self.sleep(1 / 30)

    def _sample_system(self) -> None:
        import rerun as rr

        now = time.monotonic()
        if now < self._next_system:
            return
        self._next_system = now + 1
        if self._process is None:
            self._process = psutil.Process()
            self._process.cpu_percent()
            psutil.cpu_percent()
            return  # the first CPU sample only primes the counters
        self._recording.set_time("session", duration=now - self._origin)
        values = {
            "host/cpu_percent": psutil.cpu_percent(),
            "host/memory_percent": psutil.virtual_memory().percent,
            "studio/cpu_percent": self._process.cpu_percent() / (psutil.cpu_count() or 1),
            "studio/rss_mib": self._process.memory_info().rss / 2**20,
        }
        for key, value in values.items():
            self._recording.log(f"runtime/{key}", rr.Scalars(value))

    def _sample(self) -> None:
        import rerun as rr

        now = time.monotonic()
        images_due = now >= self._next_images
        # Never retain a shared-memory view outside this lock. Rerun/compression
        # runs after releasing it, so unlinking does not wait on viewer I/O.
        with self._lock:
            store = self._store
            if store is None:
                return
            changed = store is not self._last_store
            if changed:
                self._sent.clear()
                self._last_store = store
            spec = store.spec
            samples = {key: store.read(key) for key in (STATE, ACTION)}
            if images_due:
                samples.update({f"camera/{key}": store.read_observation(key) for key in spec.observations})
        if changed:
            self._configure(spec)
        if images_due:
            self._next_images = now + 0.1
        for key, sample in samples.items():
            if sample is None or sample.timestamp == self._sent.get(key):
                continue
            self._recording.set_time("session", duration=sample.timestamp - self._origin)
            if key.startswith("camera/"):
                self._recording.log(key, rr.Image(sample.value).compress(jpeg_quality=80))
            else:
                path = "robot/joints" if key == STATE else "robot/targets"
                self._recording.log(path, rr.Scalars(sample.value))
            self._sent[key] = sample.timestamp

    def _configure(self, spec: FeatureSpec) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        recording = self._recording
        recording.log("/", rr.Clear(recursive=True))
        for path, suffix in (("joints", "measured"), ("targets", "target"), ("predicted", "prediction")):
            recording.log(
                f"robot/{path}",
                rr.SeriesLines(names=[f"{name} ({suffix})" for name in spec.joint_names]),
                static=True,
            )
        views = [rrb.Spatial2DView(origin=f"/camera/{name}", name=f"{name} (latest)") for name in spec.observations]
        recording.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(
                    *([rrb.Grid(*views)] if views else []),
                    rrb.TimeSeriesView(origin="/robot", name="Measured joints, targets and predictions"),
                    rrb.TimeSeriesView(origin="/queue", name="Action queue"),
                    rrb.TimeSeriesView(origin="/inference", name="Inference round-trip latency (ms)"),
                    rrb.TextDocumentView(origin="/runtime/monitoring", name="Inference statistics"),
                    rrb.TimeSeriesView(origin="/runtime/studio", name="Studio backend CPU (%) and RSS (MiB)"),
                    rrb.TimeSeriesView(origin="/runtime/host", name="Host CPU and memory (%)"),
                ),
                collapse_panels=True,
            )
        )

    def _drain_events(self) -> None:
        import rerun as rr
        from physicalai.runtime.events import InferenceEvent, MetricsEvent

        recording = self._recording
        # Bounded even when producers are faster than the sampler.
        for _ in range(64):
            try:
                event, hz, ready_at, received_at = self._events.popleft()
            except IndexError:
                break
            elapsed = received_at - self._origin
            recording.set_time("session", duration=elapsed)
            if isinstance(event, MetricsEvent):
                if (remaining := event.values.get("queue_remaining")) is not None:
                    recording.log("queue/remaining", rr.Scalars(remaining))
            elif isinstance(event, InferenceEvent):
                chunk = event.chunk[event.offset :]
                if chunk.size:
                    recording.log("robot/predicted", rr.Clear(recursive=False))
                    start = max(received_at + 1 / hz, ready_at) - self._origin
                    recording.send_columns(
                        "robot/predicted",
                        indexes=[rr.TimeColumn("session", duration=start + np.arange(len(chunk)) / hz)],
                        columns=rr.Scalars.columns(scalars=chunk.ravel()).partition([chunk.shape[1]] * len(chunk)),
                    )
                recording.log("inference/latency_ms", rr.Scalars(event.latency_s * 1000))
                self._latencies.append(event.latency_s * 1000)
                p50, p95, p99 = np.percentile(self._latencies, [50, 95, 99])
                recording.log(
                    "runtime/monitoring",
                    rr.TextDocument(
                        f"Round-trip latency: p50 {p50:.1f}, p95 {p95:.1f}, p99 {p99:.1f} ms "
                        f"({len(self._latencies)} samples)"
                    ),
                )
            else:
                if event.type == "state" and (event.data.control is None or event.data.control.type != "model"):
                    recording.log("robot/predicted", rr.Clear(recursive=False))
                    recording.log("queue", rr.Clear(recursive=True))
                recording.log("runtime/lifecycle", rr.TextLog(event.model_dump_json()))
