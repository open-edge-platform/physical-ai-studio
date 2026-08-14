from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from exceptions import BaseException as AppBaseException
from runtime.contract import AckEvent, LifecycleEvent, QueueEventSink, StateEvent
from runtime.transport.codec import decode_event, decode_metadata, encode_command
from runtime.transport.ids import command_key, error_key, lifecycle_key, metadata_key, request_key, state_key, tick_key
from runtime.transport.session import open_session

if TYPE_CHECKING:
    from runtime.contract import Command, RuntimeEvent


class RuntimeProcessError(AppBaseException):
    def __init__(self, message: str, error_code: str = "robot_connection_failed") -> None:
        super().__init__(message=message, error_code=error_code, http_status=500)


class RuntimeSessionClient:
    """Parent-side command and event client for one runtime process."""

    def __init__(self, name: str, *, instance_id: str | None = None) -> None:
        self._name = name
        self._instance_id = instance_id
        self._session: Any = None
        self._command_pub: Any = None
        self._subscribers: list[Any] = []
        self._events = QueueEventSink()
        self._metadata_ready = threading.Event()
        self._hardware_ready = threading.Event()
        self._shutdown_received = threading.Event()
        self._pending_command: Command | None = None
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._last_state: StateEvent | None = None
        self.error: RuntimeProcessError | None = None

    @property
    def shutdown_received(self) -> bool:
        return self._shutdown_received.is_set()

    def open(self) -> None:
        """Declare telemetry subscribers before the child can publish events."""
        import zenoh

        self._session = open_session(self._name, listen=False)
        for key in (tick_key(self._name), state_key(self._name), error_key(self._name), lifecycle_key(self._name)):
            self._subscribers.append(self._session.declare_subscriber(key, self._receive_event))
        self._command_pub = self._session.declare_publisher(
            command_key(self._name),
            reliability=zenoh.Reliability.BEST_EFFORT,
            congestion_control=zenoh.CongestionControl.DROP,
            encoding=zenoh.Encoding("application/msgpack"),
        )

    def probe(self, timeout: float = 1.0) -> dict[str, Any] | None:
        """Query metadata once. No retry and no client state change."""
        if self._session is None:
            raise RuntimeError("Runtime session client is not open")
        try:
            replies = self._session.get(metadata_key(self._name), timeout=timeout)
            for reply in replies:
                sample = reply.ok
                if sample is None:
                    continue
                metadata = decode_metadata(sample.payload.to_bytes())
                if self._instance_id is not None and metadata.get("instance_id") != self._instance_id:
                    continue
                return metadata
        except Exception:
            logger.debug("Runtime metadata query failed for {}", self._name, exc_info=True)
        return None

    def probe_with_retry(self, timeout: float) -> dict[str, Any] | None:
        """Poll ``probe`` until metadata answers or the deadline elapses."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            metadata = self.probe(timeout=min(1.0, remaining))
            if metadata is not None:
                return metadata
            time.sleep(min(0.05, remaining))

    def attach(self, metadata: dict[str, Any]) -> None:
        """Adopt the session generation id and flush any command buffered before it."""
        instance_id = metadata.get("instance_id")
        if instance_id is not None and (not isinstance(instance_id, str) or not instance_id):
            raise RuntimeError("Runtime session metadata instance_id must be a string")
        self._instance_id = instance_id
        self._metadata_ready.set()
        self._flush_pending_command()

    def connect(self, timeout: float = 10.0, *, process: Any = None) -> dict[str, Any]:
        """Wait for metadata before allowing any command publication."""
        if self._session is None:
            raise RuntimeError("Runtime session client is not open")
        deadline = time.monotonic() + timeout
        while True:
            if self.error is not None:
                raise self.error
            if process is not None and not process.is_alive():
                process_error = getattr(process, "error", None)
                if process_error is not None:
                    raise RuntimeProcessError(process_error.message, process_error.error_code)
                raise RuntimeProcessError("Runtime session stopped before answering metadata")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Runtime session {self._name} did not answer metadata")
            metadata = self.probe(timeout=min(1.0, remaining))
            if metadata is not None:
                self.attach(metadata)
                return metadata
            time.sleep(min(0.05, remaining))

    def apply(self, command: Command) -> None:
        """Publish now when metadata is ready, otherwise retain the newest command."""
        with self._lock:
            if not self._metadata_ready.is_set():
                self._pending_command = command
                return
            self._command_pub.put(encode_command(command, instance_id=self._instance_id))

    def request(self, command: Command, timeout: float = 5.0) -> AckEvent:
        if not self._metadata_ready.is_set():
            raise RuntimeError("Runtime session metadata is not ready")
        replies = self._session.get(
            request_key(self._name),
            payload=encode_command(command, instance_id=self._instance_id),
            encoding="application/msgpack",
            timeout=timeout,
        )
        for reply in replies:
            sample = reply.ok
            if sample is None:
                continue
            event, _, instance_id = decode_event(sample.payload.to_bytes())
            if instance_id != self._instance_id:
                continue
            if not isinstance(event, AckEvent):
                raise TypeError(f"Expected an ack reply, got {event.event}")
            ack = event
            if ack.data.request_id != command.request_id:
                raise RuntimeError("Runtime acknowledgement request_id does not match the request")
            return ack
        raise TimeoutError(f"Runtime request {command.command} received no reply")

    def get_nowait(self) -> RuntimeEvent:
        return self._events.get_nowait()

    def wait_until_ready(self, process: Any, timeout: float | None = None) -> None:
        """Wait for hardware readiness, optionally bounded by a timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._hardware_ready.is_set():
            if self.error is not None:
                raise self.error
            process_error = getattr(process, "error", None)
            if process_error is not None:
                raise RuntimeProcessError(process_error.message, process_error.error_code)
            if not process.is_alive():
                raise RuntimeProcessError("Runtime session stopped before becoming ready")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Runtime session did not become ready")
            self._recover_ready_state_from_metadata()
            if not self._hardware_ready.is_set():
                time.sleep(0.02)

    def close(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.undeclare()
            except Exception:
                logger.debug("Failed to undeclare runtime subscriber", exc_info=True)
        if self._command_pub is not None:
            try:
                self._command_pub.undeclare()
            except Exception:
                logger.debug("Failed to undeclare runtime command publisher", exc_info=True)
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                logger.debug("Failed to close runtime Zenoh session", exc_info=True)

    def _flush_pending_command(self) -> None:
        with self._lock:
            if self._pending_command is not None:
                self._command_pub.put(encode_command(self._pending_command, instance_id=self._instance_id))
                self._pending_command = None

    def _receive_event(self, sample: Any) -> None:
        try:
            event, fatal, instance_id = decode_event(sample.payload.to_bytes())
            if self._instance_id is not None and instance_id != self._instance_id:
                logger.warning("Rejected runtime event for a different instance")
                return
            if fatal:
                error_code = getattr(event, "error_code", "robot_connection_failed")
                message = getattr(event, "message", "Runtime session failed")
                self.error = RuntimeProcessError(message, error_code)
                return
            if isinstance(event, StateEvent) and event.data.connected:
                self._accept_ready_state(event)
                return
            if isinstance(event, LifecycleEvent) and event.data.event == "shutdown":
                self._shutdown_received.set()
            self._events.emit(event)
        except Exception:
            logger.exception("Rejected malformed runtime event")

    def _recover_ready_state_from_metadata(self) -> None:
        try:
            replies = self._session.get(metadata_key(self._name), timeout=0.2)
            for reply in replies:
                sample = reply.ok
                if sample is None:
                    continue
                metadata = decode_metadata(sample.payload.to_bytes())
                if self._instance_id is not None and metadata.get("instance_id") != self._instance_id:
                    continue
                state = metadata.get("state")
                if metadata.get("status") == "running" and state is not None:
                    self._accept_ready_state(StateEvent.model_validate(state), initial_only=True)
                    return
        except Exception:
            logger.debug("Runtime ready-state query failed for {}", self._name, exc_info=True)

    def _accept_ready_state(self, event: StateEvent, *, initial_only: bool = False) -> None:
        with self._state_lock:
            if event == self._last_state:
                return
            if self._hardware_ready.is_set():
                if not initial_only:
                    self._events.emit(event)
                    self._last_state = event
                return
            self._events.emit(event)
            self._last_state = event
            self._hardware_ready.set()
