from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from exceptions import RobotDeviceAlreadyOwnedError
from runtime.contract import AckData, AckEvent, ErrorEvent, LifecycleEvent, ObservationEvent, StateEvent
from runtime.transport.codec import decode_command, encode_event, encode_metadata
from runtime.transport.ids import command_key, error_key, lifecycle_key, metadata_key, request_key, state_key, tick_key
from runtime.transport.session import open_session

if TYPE_CHECKING:
    from collections.abc import Callable

    from runtime.contract import Command, RuntimeEvent


class RuntimeZenohServer:
    """Bind a process-local RuntimeSession to Studio's Zenoh contract."""

    def __init__(self, name: str, *, instance_id: str | None = None) -> None:
        self._name = name
        self._instance_id = instance_id
        self._session: Any = None
        self._command_sub: Any = None
        self._request_queryable: Any = None
        self._metadata_queryable: Any = None
        self._publishers: dict[type, Any] = {}
        self._command_handler: Callable[[Command], None] | None = None
        self._stop = threading.Event()
        self._command_thread: threading.Thread | None = None
        self._metadata_lock = threading.Lock()
        # This public generation id prevents stale process traffic. It is not
        # authentication: any local process that reaches the loopback endpoint
        # is trusted in B1. Multi-user hosts require a separate auth design.
        self._metadata: dict[str, Any] = {
            "protocol_version": 1,
            "name": name,
            "instance_id": instance_id,
            "status": "starting",
        }

    @property
    def is_open(self) -> bool:
        """Return whether all transport publishers were declared."""
        return bool(self._publishers)

    def update_metadata(self, **values: Any) -> None:
        """Update queryable session status under the metadata lock."""
        with self._metadata_lock:
            self._metadata.update(values)

    def open(self, command_handler: Callable[[Command], None]) -> None:
        """Declare every endpoint, then expose metadata as the readiness gate."""
        import zenoh

        self._command_handler = command_handler
        try:
            self._session = open_session(self._name, listen=True)
        except Exception as exc:
            if "Address already in use" in str(exc):
                logger.warning("Runtime session {} port is taken: {}", self._name, exc)
                raise RobotDeviceAlreadyOwnedError from exc
            raise
        self._command_sub = self._session.declare_subscriber(
            command_key(self._name),
            zenoh.handlers.RingChannel(1),
        )
        self._request_queryable = self._session.declare_queryable(
            request_key(self._name),
            self._answer_request,
        )
        publisher_qos = {
            "reliability": zenoh.Reliability.BEST_EFFORT,
            "congestion_control": zenoh.CongestionControl.DROP,
            "express": True,
            "encoding": zenoh.Encoding("application/msgpack"),
        }
        self._publishers = {
            ObservationEvent: self._session.declare_publisher(tick_key(self._name), **publisher_qos),
            StateEvent: self._session.declare_publisher(state_key(self._name), **publisher_qos),
            ErrorEvent: self._session.declare_publisher(error_key(self._name), **publisher_qos),
            LifecycleEvent: self._session.declare_publisher(
                lifecycle_key(self._name),
                reliability=zenoh.Reliability.BEST_EFFORT,
                congestion_control=zenoh.CongestionControl.DROP,
                encoding=zenoh.Encoding("application/msgpack"),
            ),
        }
        self._command_thread = threading.Thread(
            target=self._pump_commands,
            name=f"{self._name}-commands",
            daemon=True,
        )
        self._command_thread.start()

        def answer_metadata(query: Any) -> None:
            with self._metadata_lock:
                metadata_payload = encode_metadata(self._metadata)
            query.reply(
                metadata_key(self._name),
                metadata_payload,
                encoding=zenoh.Encoding("application/msgpack"),
            )

        self._metadata_queryable = self._session.declare_queryable(metadata_key(self._name), answer_metadata)

    def has_matching_subscribers(self) -> bool:
        """Return whether at least one client is subscribed to session state."""
        publisher = self._publishers.get(StateEvent)
        return publisher is not None and bool(publisher.matching_status.matching)

    def emit(self, event: RuntimeEvent, *, fatal: bool = False) -> None:
        """Publish an event without allowing transport failure to stop the robot loop."""
        with self._metadata_lock:
            if isinstance(event, StateEvent) and event.data.connected:
                self._metadata["status"] = "running"
                self._metadata["state"] = event.model_dump(mode="json")
            elif isinstance(event, LifecycleEvent) and event.data.event == "shutdown":
                self._metadata["status"] = "stopped"
            elif fatal:
                self._metadata["status"] = "error"
                self._metadata["error"] = event.model_dump(mode="json")
        publisher = self._publishers.get(type(event))
        if publisher is None:
            logger.warning("No Zenoh publisher for runtime event {}", event.event)
            return
        try:
            publisher.put(encode_event(event, fatal=fatal, instance_id=self._instance_id))
        except Exception:
            logger.exception("Failed to publish runtime event {}", event.event)

    def wait_for_client(self, timeout: float = 5.0) -> None:
        """Wait until the parent subscriber can receive the one-time ready state."""
        state_publisher = self._publishers[StateEvent]
        deadline = time.monotonic() + timeout
        while not state_publisher.matching_status.matching:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Runtime session {self._name} has no state subscriber")
            time.sleep(0.01)

    def close(self) -> None:
        self._stop.set()
        if self._command_thread is not None:
            self._command_thread.join(timeout=1.0)
        endpoints = (
            self._metadata_queryable,
            self._request_queryable,
            self._command_sub,
            *self._publishers.values(),
        )
        for endpoint in endpoints:
            if endpoint is not None:
                try:
                    endpoint.undeclare()
                except Exception:
                    logger.debug("Failed to undeclare runtime Zenoh endpoint", exc_info=True)
        if self._session is not None:
            self._session.close()

    def _pump_commands(self) -> None:
        while not self._stop.is_set():
            sample = self._command_sub.try_recv()
            if sample is None:
                self._stop.wait(0.005)
                continue
            try:
                command, instance_id = decode_command(sample.payload.to_bytes())
                if instance_id != self._instance_id:
                    logger.warning("Rejected runtime command for a different instance")
                    continue
                if self._command_handler is not None:
                    self._command_handler(command)
            except Exception:
                logger.exception("Rejected malformed runtime command")

    def _answer_request(self, query: Any) -> None:
        import zenoh

        try:
            if query.payload is None:
                raise ValueError("Runtime request has no payload")
            command, instance_id = decode_command(query.payload.to_bytes())
            if instance_id != self._instance_id:
                raise ValueError("Runtime request targets a different instance")
            request_id = command.request_id
            if request_id is None:
                raise ValueError("Runtime request has no request_id")
            ack = AckEvent(
                data=AckData(
                    request_id=request_id,
                    ok=False,
                    error=f"{command.command} is not supported by this runtime session",
                )
            )
            query.reply(
                request_key(self._name),
                encode_event(ack, instance_id=self._instance_id),
                encoding=zenoh.Encoding("application/msgpack"),
            )
        except Exception as exc:
            logger.warning("Rejected malformed runtime request: {}", exc)
            query.reply_err(str(exc), encoding=zenoh.Encoding.TEXT_PLAIN)
