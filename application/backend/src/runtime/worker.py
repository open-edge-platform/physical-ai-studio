"""Process worker that owns one RuntimeSession for the lifetime of a websocket."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from exceptions import BaseException as AppBaseException
from runtime.contract import AckData, AckEvent, DiscardEpisodeCommand, ObservationEvent, SaveEpisodeCommand
from runtime.session import RuntimeSession
from workers.base import BaseProcessWorker

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass

    from runtime.contract import Command, RuntimeEvent

_COMMAND_POLL_S = 0.1
_EVENT_PUT_TIMEOUT_S = 1.0


@dataclass(frozen=True, slots=True)
class WorkerFatal:
    """A failure that ended the session, reported to the parent over the event queue."""

    message: str
    error_code: str


def fatal_from_exception(exc: BaseException) -> WorkerFatal:
    """Map a session failure onto the error the websocket reports."""
    if isinstance(exc, AppBaseException):
        return WorkerFatal(message=exc.message, error_code=exc.error_code)
    return WorkerFatal(message=str(exc) or "Failed to connect to the robot.", error_code="robot_connection_failed")


class _ProcessEventSink:
    """Forward session events to the parent without letting a slow reader stall the control loop."""

    def __init__(self, events: mp.Queue) -> None:
        self._events = events

    def emit(self, event: RuntimeEvent | WorkerFatal) -> None:
        try:
            if isinstance(event, ObservationEvent):
                # Observations are superseded by the next tick; drop rather than block.
                self._events.put_nowait(event)
            else:
                self._events.put(event, timeout=_EVENT_PUT_TIMEOUT_S)
        except queue.Full:
            if not isinstance(event, ObservationEvent):
                logger.warning("Runtime event queue is full; dropped {}", type(event).__name__)


class _WorkerStopSignal:
    """``StopSignal`` for ``RobotRuntime``: set on stop, app shutdown, or parent death."""

    def __init__(self, worker: RuntimeSessionWorker) -> None:
        self._worker = worker

    def is_set(self) -> bool:
        return self._worker.should_stop()


class RuntimeSessionWorker(BaseProcessWorker):
    """Run one RuntimeSession in a child process, driven by command and event queues."""

    ROLE = "RuntimeSession"

    def __init__(
        self,
        *,
        document: dict[str, Any],
        follower_name: str | None,
        leader_name: str | None,
        stop_event: EventClass,
        command_queue: mp.Queue,
        event_queue: mp.Queue,
    ) -> None:
        super().__init__(stop_event=stop_event)
        self._document = document
        self._follower_name = follower_name
        self._leader_name = leader_name
        self._command_queue = command_queue
        self._event_queue = event_queue
        # Created in the child: none of these survive pickling for spawn.
        self._sink: _ProcessEventSink | None = None
        self._session: RuntimeSession | None = None
        self._pump: threading.Thread | None = None
        self._pump_stop: threading.Event | None = None
        self._requests: ThreadPoolExecutor | None = None

    async def setup(self) -> None:
        await super().setup()
        # tqdm renders to stderr, the session's log sink, on every saved episode.
        from datasets.utils import disable_progress_bars

        disable_progress_bars()

        self._sink = _ProcessEventSink(self._event_queue)
        self._session = RuntimeSession(
            self._document,
            event_sink=self._sink,
            follower_name=self._follower_name,
            leader_name=self._leader_name,
        )
        try:
            await self._session.setup()
        except Exception as exc:
            self._sink.emit(fatal_from_exception(exc))
            raise
        self._pump_stop = threading.Event()
        self._requests = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runtime-requests")
        self._pump = threading.Thread(target=self._pump_commands, name="runtime-commands", daemon=True)
        self._pump.start()

    async def run_loop(self) -> None:
        if self._session is None or self._sink is None:
            return
        try:
            # Returns when the stop signal is set; teardown then releases the devices.
            self._session.run(_WorkerStopSignal(self))
        except Exception as exc:
            logger.exception("Runtime session failed")
            self._sink.emit(fatal_from_exception(exc))

    async def teardown(self) -> None:
        if self._pump_stop is not None:
            self._pump_stop.set()
        if self._pump is not None:
            self._pump.join(timeout=1.0)
        if self._session is not None:
            try:
                # Drains in-flight save/discard jobs and finalizes the recording.
                await self._session.teardown()
            except Exception as exc:
                logger.exception("Runtime session teardown failed")
                if self._sink is not None:
                    self._sink.emit(fatal_from_exception(exc))
        if self._requests is not None:
            self._requests.shutdown(wait=True)

    def _pump_commands(self) -> None:
        assert self._pump_stop is not None  # noqa: S101
        while not self._pump_stop.is_set():
            try:
                command = self._command_queue.get(timeout=_COMMAND_POLL_S)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                return
            try:
                self._dispatch(command)
            except Exception:
                logger.exception("Runtime command {} failed", getattr(command, "command", command))

    def _dispatch(self, command: Command) -> None:
        if self._session is None or self._requests is None:
            return
        if isinstance(command, SaveEpisodeCommand | DiscardEpisodeCommand):
            # Acked requests can block for the whole recording timeout; keep them
            # off the pump so mode changes still land meanwhile.
            self._requests.submit(self._answer_request, command)
            return
        self._session.apply(command)

    def _answer_request(self, command: SaveEpisodeCommand | DiscardEpisodeCommand) -> None:
        if self._session is None or self._sink is None:
            return
        try:
            self._session.handle_request(command)
        except Exception as exc:
            ack = AckEvent(data=AckData(request_id=command.request_id, ok=False, error=str(exc)))
        else:
            ack = AckEvent(data=AckData(request_id=command.request_id, ok=True))
        self._sink.emit(ack)
