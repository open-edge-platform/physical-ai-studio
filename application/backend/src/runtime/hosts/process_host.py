from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any

from exceptions import BaseException as AppBaseException
from runtime.contract import ErrorEvent
from runtime.hosts.thread_host import WorkerStopSignal
from runtime.session import RuntimeSession
from runtime.transport.server import RuntimeZenohServer
from workers.base import BaseProcessWorker

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass


class RuntimeProcessHost(BaseProcessWorker):
    """Run one RuntimeSession in a child process controlled over Zenoh."""

    ROLE = "RuntimeProcessHost"

    def __init__(
        self,
        name: str,
        document: dict[str, Any],
        *,
        stop_event: EventClass,
        instance_id: str | None = None,
        follower_name: str | None = None,
        leader_name: str | None = None,
    ) -> None:
        super().__init__(stop_event=stop_event)
        self._session_name = name
        self._instance_id = instance_id
        self._document = document
        self._follower_name = follower_name
        self._leader_name = leader_name
        self._server: RuntimeZenohServer | None = None
        self._runtime_session: RuntimeSession | None = None
        self._run_completed = False
        self.completed_cleanly = mp.Event()
        self._error_reader, self._error_writer = mp.Pipe(duplex=False)
        self._error: AppBaseException | None = None

    @property
    def error(self) -> AppBaseException | None:
        """Return a child failure even when Zenoh could not open to report it."""
        if self._error is None and self._error_reader.poll():
            message, error_code = self._error_reader.recv()
            self._error = AppBaseException(message=message, error_code=error_code, http_status=500)
        return self._error

    async def setup(self) -> None:
        await super().setup()
        try:
            self._server = RuntimeZenohServer(self._session_name, instance_id=self._instance_id)
            self._runtime_session = RuntimeSession(
                self._document,
                event_sink=self._server,
                follower_name=self._follower_name,
                leader_name=self._leader_name,
            )
            self._server.open(self._runtime_session.apply)
            self._server.wait_for_client()
            await self._runtime_session.setup()
        except Exception as exc:
            self._publish_fatal_error(exc)
            raise

    async def run_loop(self) -> None:
        if self._runtime_session is None:
            raise RuntimeError("Runtime process host has not been set up")
        try:
            self._runtime_session.run(WorkerStopSignal(self))
            self._run_completed = True
        except Exception as exc:
            self._publish_fatal_error(exc)
            raise

    async def teardown(self) -> None:
        try:
            try:
                if self._runtime_session is not None:
                    await self._runtime_session.teardown()
            finally:
                if self._server is not None:
                    self._server.close()
        except Exception as exc:
            self._publish_fatal_error(exc)
            raise
        if self._run_completed:
            self.completed_cleanly.set()

    def stop(self) -> None:
        """Stop cooperatively, then terminate and kill a wedged child."""
        super().stop()
        if self.is_alive():
            self.kill()
            self.join(timeout=2.0)

    def _publish_fatal_error(self, exc: Exception) -> None:
        if isinstance(exc, AppBaseException):
            event = ErrorEvent(message=exc.message, error_code=exc.error_code)
        else:
            event = ErrorEvent(
                message=str(exc) or "Failed to connect to the robot.",
                error_code="robot_connection_failed",
            )
        self._error_writer.send((event.message, event.error_code))
        if self._server is not None and self._server.is_open:
            self._server.emit(event, fatal=True)
