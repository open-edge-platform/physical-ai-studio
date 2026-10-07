"""Parent-side handle for one runtime session worker process."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from exceptions import BaseException as AppBaseException
from runtime.config_builder import runtime_camera_keys
from runtime.contract import QueueEventSink, StateEvent
from runtime.session import RECORDING_TEARDOWN_TIMEOUT_S
from runtime.worker import RuntimeSessionWorker, WorkerFatal

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass
    from uuid import UUID

    from runtime.contract import Command, RuntimeEvent, StateData

# Teardown may copy a recording cache back to the dataset; give it that long
# before escalating to SIGTERM.
STOP_TIMEOUT_S = RECORDING_TEARDOWN_TIMEOUT_S + 5.0
_EVENT_QUEUE_SIZE = 256
_READY_POLL_S = 0.02
_JOIN_POLL_S = 0.1

SessionStatus = Literal["starting", "running", "stopped", "error"]


class RuntimeProcessError(AppBaseException):
    def __init__(self, message: str, error_code: str = "robot_connection_failed") -> None:
        super().__init__(message=message, error_code=error_code, http_status=500)


class RuntimeSessionHandle:
    """Start, talk to and stop one RuntimeSessionWorker.

    The session lives exactly as long as the websocket that created it: there
    is no reattach, so whoever creates a handle is responsible for ``stop()``.
    """

    def __init__(
        self,
        session_name: str,
        *,
        follower_id: UUID,
        document: dict[str, Any],
        follower_name: str | None,
        leader_name: str | None,
        stop_event: EventClass,
    ) -> None:
        self.session_name = session_name
        self.follower_id = follower_id
        self.follower_name = follower_name
        self.leader_name = leader_name
        self.camera_keys = runtime_camera_keys(document)
        self.started_at = datetime.now(UTC)
        self.state: StateData | None = None
        self.error: RuntimeProcessError | None = None
        self._command_queue: mp.Queue = mp.Queue()
        self._event_queue: mp.Queue = mp.Queue(maxsize=_EVENT_QUEUE_SIZE)
        self._worker = RuntimeSessionWorker(
            document=document,
            follower_name=follower_name,
            leader_name=leader_name,
            stop_event=stop_event,
            command_queue=self._command_queue,
            event_queue=self._event_queue,
        )
        self._events = QueueEventSink()
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._stopped = threading.Event()
        # Serializes start against stop so a stop during spawn cannot orphan the child.
        self._lifecycle_lock = threading.Lock()
        self._drain_lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        return self._worker.pid

    @property
    def stopping(self) -> bool:
        """Whether a stop has been requested; the slot is on its way to being freed."""
        return self._stopping.is_set()

    @property
    def status(self) -> SessionStatus:
        if self.error is not None:
            return "error"
        if self._stopping.is_set():
            return "stopped"
        if self._ready.is_set():
            return "running"
        return "starting"

    def is_alive(self) -> bool:
        return self._worker.is_alive()

    def start(self) -> None:
        """Spawn the worker. Blocking: a spawn re-imports the backend."""
        with self._lifecycle_lock:
            if self._stopping.is_set():
                raise RuntimeProcessError("Runtime session was stopped before it started")
            self._worker.start()

    def wait_until_ready(self) -> None:
        """Block until the robot reports connected, the worker fails, or a stop is requested."""
        while True:
            self._drain()
            if self._ready.is_set():
                return
            if self.error is not None:
                raise self.error
            if self._stopping.is_set():
                raise RuntimeProcessError("Runtime session was stopped before becoming ready")
            if not self.is_alive():
                # The child flushes its queue before exiting; read what it left.
                self._drain()
                if self.error is not None:
                    raise self.error
                raise RuntimeProcessError("Runtime session stopped before becoming ready")
            time.sleep(_READY_POLL_S)

    def apply(self, command: Command) -> None:
        """Send a command. Acked requests answer with an ``AckEvent`` on the event stream."""
        if self._stopping.is_set():
            return
        self._command_queue.put(command)

    def get_nowait(self) -> RuntimeEvent:
        self._drain()
        return self._events.get_nowait()

    def stop(self) -> None:
        """Stop the worker through its stop event and wait for teardown. Blocking, idempotent."""
        self._stopping.set()
        with self._lifecycle_lock:
            if self._stopped.is_set():
                return
            try:
                self._stop_worker()
            finally:
                # Nothing reads commands any more; do not let a buffered put
                # block this process at exit.
                self._command_queue.cancel_join_thread()
                self._command_queue.close()
                self._stopped.set()

    def _stop_worker(self) -> None:
        if self._worker.pid is None:
            return
        self._worker.request_stop()
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while self._worker.is_alive() and time.monotonic() < deadline:
            # Keep reading: a child cannot exit while its queue feeder is
            # blocked on a full pipe.
            self._drain()
            self._worker.join(_JOIN_POLL_S)
        if self._worker.is_alive():
            logger.warning(
                "Runtime session {} (pid {}) did not stop within {}s, terminating",
                self.session_name,
                self._worker.pid,
                STOP_TIMEOUT_S,
            )
            self._worker.terminate()
            self._worker.join(2.0)
            if self._worker.is_alive():
                logger.error("Killing runtime session {} (pid {})", self.session_name, self._worker.pid)
                self._worker.kill()
                self._worker.join(1.0)
        self._drain()

    def _drain(self) -> None:
        with self._drain_lock:
            while True:
                try:
                    item = self._event_queue.get_nowait()
                except (queue.Empty, OSError, ValueError):
                    return
                self._accept(item)

    def _accept(self, item: RuntimeEvent | WorkerFatal) -> None:
        if isinstance(item, WorkerFatal):
            self.error = RuntimeProcessError(item.message, item.error_code)
            return
        if isinstance(item, StateEvent):
            self.state = item.data
            if item.data.connected:
                self._ready.set()
        self._events.emit(item)
