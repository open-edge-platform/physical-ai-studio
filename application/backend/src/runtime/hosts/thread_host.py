from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from workers.base import BaseThreadWorker, StoppableMixin

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass

    from runtime.session import RuntimeSession


class WorkerStopSignal:
    """Bridge Studio worker stop semantics to physicalai's StopSignal protocol."""

    def __init__(self, worker: StoppableMixin) -> None:
        self._worker = worker

    def is_set(self) -> bool:
        return self._worker.should_stop()


class RuntimeThreadHost(BaseThreadWorker):
    ROLE = "RuntimeThreadHost"

    def __init__(self, session: RuntimeSession, *, stop_event: EventClass) -> None:
        super().__init__(stop_event=stop_event)
        self.session = session
        self.completed = threading.Event()
        self.error: Exception | None = None

    async def run_loop(self) -> None:
        try:
            await self.session.setup()
            self.session.run(WorkerStopSignal(self))
        except Exception as exc:
            self.error = exc
        finally:
            self.completed.set()

    async def teardown(self) -> None:
        await self.session.teardown()

    async def wait_until_ready(self, poll_interval: float = 0.02) -> None:
        while not self.session.ready.is_set():
            if self.completed.is_set():
                if self.error is not None:
                    raise self.error
                raise RuntimeError("Runtime session stopped before becoming ready")
            await asyncio.sleep(poll_interval)
