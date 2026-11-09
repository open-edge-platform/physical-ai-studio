"""Workers that own a resource for as long as they run.

A worker is a thread or a process whose body is::

    async with self.lifecycle() as context:
        await self.run_loop(context)

``lifecycle`` acquires what the worker holds (a robot, a camera, a dataset) and
releases it on the way out, on the worker's own thread or in its own process.
That keeps acquiring and releasing next to each other, and a device is always
released by whoever opened it.

The parent enters the worker as an async context manager. ``async with worker:``
starts it, waits until ``lifecycle`` has yielded, and raises if it failed to get
there. Leaving the block asks the worker to stop, waits for it, and kills a
process that will not stop. A session holds its workers on one
``AsyncExitStack`` so they are torn down in reverse order.

Workers are single-use, like the thread or process they are.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import signal
import threading
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Self, cast

import anyio
from loguru import logger

from utils.multiprocessing import ensure_spawn_start_method

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as ProcessEvent
    from types import FrameType, TracebackType

_POLL_INTERVAL_S = 0.05
"""How often the parent checks on a worker it is waiting for."""


class WorkerStartupError(RuntimeError):
    """The worker failed, died or timed out before ``lifecycle`` yielded."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"Worker {name} failed to start: {reason}")
        self.worker_name = name
        self.reason = reason


def _describe(error: BaseException) -> str:
    return "".join(traceback.format_exception_only(error)).strip()


class _Worker[C](ABC):
    """What thread and process workers share: the body and the parent side.

    Subclasses mix this with ``threading.Thread`` or ``multiprocessing.Process``
    and supply the primitives that differ between the two.
    """

    ready_timeout_s: float = 30.0
    """How long the parent waits for ``lifecycle`` to yield."""

    stop_timeout_s: float = 10.0
    """How long the parent waits for the worker to finish after asking it to stop.

    A worker whose teardown is slow on purpose -- copying a recording back to
    the dataset -- raises this rather than being killed in the middle of it.
    """

    if TYPE_CHECKING:
        # threading.Thread and multiprocessing.Process provide these.
        name: str

        def start(self) -> None: ...
        def join(self, timeout: float | None = None) -> None: ...
        def is_alive(self) -> bool: ...

    # Hooks for subclasses

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[C]:
        """Acquire what this worker holds, yield it, and release it after.

        Runs on the worker's own thread or in its own process. Override it as
        an ``@asynccontextmanager``; the default holds nothing.
        """
        yield cast("C", None)

    @abstractmethod
    async def run_loop(self, context: C) -> None:
        """Do the work until :meth:`should_stop`, then return."""

    def should_stop(self) -> bool:
        """Whether the parent has asked this worker to stop."""
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        """Ask the worker to stop without waiting for it; leaving ``async with`` waits."""
        self._stop_event.set()

    async def sleep(self, seconds: float) -> None:
        """Sleep up to ``seconds``, waking early once a stop is requested."""
        deadline = time.monotonic() + seconds
        # Polled: the stop is a multiprocessing event, which has nothing to await.
        while not self.should_stop() and (remaining := deadline - time.monotonic()) > 0:  # noqa: ASYNC110
            await asyncio.sleep(min(remaining, _POLL_INTERVAL_S))

    def _init_events(self) -> None:
        """Create the events the parent and the worker share.

        ``multiprocessing`` events for both kinds, so a thread and a process
        coordinate the same way. Created before ``start`` so spawning carries
        them into a child.
        """
        context = mp.get_context("spawn")
        self._stop_event: ProcessEvent = context.Event()
        # ``lifecycle`` has yielded: the worker holds its resource.
        self._started_event: ProcessEvent = context.Event()
        # Startup is over, either way: the parent can stop waiting.
        self._settled_event: ProcessEvent = context.Event()

    # What differs between a thread and a process

    @abstractmethod
    def _report_failure(self, error: BaseException) -> None: ...

    @abstractmethod
    def _reported_failure(self) -> str | None: ...

    @abstractmethod
    def _kill(self) -> None: ...

    # The worker's body

    async def _main(self) -> None:
        try:
            async with self.lifecycle() as context:
                self._started_event.set()
                self._settled_event.set()
                await self.run_loop(context)
        except BaseException as error:
            logger.exception("Worker {} failed", self.name)
            self._report_failure(error)
        finally:
            # Also set when acquiring failed, so the parent stops waiting.
            self._settled_event.set()

    def _run_body(self) -> None:
        # Keyed by class: LogConfig routes each worker type to its own log file.
        with logger.contextualize(worker=type(self).__name__):
            logger.info("Starting worker {}", self.name)
            asyncio.run(self._main())
            logger.info("Stopped worker {}", self.name)

    # The parent side

    @property
    def failure(self) -> str | None:
        """Why the worker stopped, if it stopped because of an error."""
        return self._reported_failure()

    async def wait_stopped(self) -> None:
        """Return once the worker has exited, for whatever reason."""
        # Polled: a process exit has no awaitable, and a thread blocked in
        # ``join`` would outlive a cancelled wait.
        while self.is_alive():  # noqa: ASYNC110
            await asyncio.sleep(_POLL_INTERVAL_S)

    async def __aenter__(self) -> Self:
        self.start()
        try:
            await self._wait_until_ready()
        except BaseException:
            # Failed or cancelled while starting: whatever it acquired so far
            # is released by its own lifecycle once it sees the stop.
            await self._stop_and_join()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._stop_and_join()
        failure = self.failure
        if failure is not None:
            # It died while the parent thought it was running. Raising here
            # would mask whatever the parent's block is exiting with.
            logger.error("Worker {} stopped early: {}", self.name, failure)

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout_s
        while not self._settled_event.is_set():
            if not self.is_alive():
                # Died without reaching its ``finally``: a hard exit or a signal.
                break
            if time.monotonic() >= deadline:
                raise WorkerStartupError(self.name, f"not ready within {self.ready_timeout_s:.0f}s")
            await asyncio.sleep(_POLL_INTERVAL_S)
        if self._started_event.is_set():
            # Started, even if it has failed since; ``__aexit__`` reports that.
            return
        failure = self.failure
        raise WorkerStartupError(self.name, failure if failure is not None else self._exit_description())

    def _exit_description(self) -> str:
        return "exited before it was ready"

    async def _stop_and_join(self) -> None:
        self._stop_event.set()
        # Shielded: a cancelled parent (a closed websocket) still waits for the device to be released.
        with anyio.CancelScope(shield=True):
            await asyncio.to_thread(self.join, self.stop_timeout_s)
            if self.is_alive():
                logger.warning("Worker {} did not stop within {:.0f}s", self.name, self.stop_timeout_s)
                self._kill()


class ThreadWorker[C](threading.Thread, _Worker[C], ABC):
    """A worker on its own thread, for work that releases the GIL or is light.

    A thread cannot be killed: one that ignores the stop is left running as a
    daemon and logged. Anything holding hardware that must be released
    belongs in a :class:`ProcessWorker`.
    """

    def __init__(self, *, name: str, daemon: bool = True) -> None:
        super().__init__(name=name, daemon=daemon)
        self._init_events()
        self._failure: str | None = None

    def run(self) -> None:
        self._run_body()

    def _report_failure(self, error: BaseException) -> None:
        self._failure = _describe(error)

    def _reported_failure(self) -> str | None:
        return self._failure

    def _kill(self) -> None:
        logger.error("Worker {} is a thread and cannot be killed; leaving it running", self.name)


class ProcessWorker[C](mp.Process, _Worker[C], ABC):
    """A worker in its own process, for hardware and anything heavy.

    The worker object is pickled into the child, so it may only hold what
    pickles: configuration, not drivers. Build the device in ``lifecycle``,
    which runs in the child.

    The child ignores SIGINT, so a Ctrl+C on the backend stops it through the
    parent rather than tearing it down twice. SIGTERM asks it to stop the same
    way the parent does, so a terminated child still releases its device. A
    child whose parent has died stops by itself.
    """

    def __init__(self, *, name: str, daemon: bool = True) -> None:
        """Pass ``daemon=False`` for a worker that starts processes of its own."""
        ensure_spawn_start_method()
        super().__init__(name=name, daemon=daemon)
        self._init_events()
        # A failure crosses back as text: the child writes to its own copy of
        # this object, so the parent cannot read an attribute the child set.
        self._failed_event: ProcessEvent = mp.get_context("spawn").Event()
        self._failures: Queue[str] = mp.get_context("spawn").Queue()
        self._failure: str | None = None

    def run(self) -> None:
        from core.logging import setup_logging

        # Loguru sinks do not cross the process boundary.
        setup_logging()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, self._on_sigterm)
        self._run_body()

    def _on_sigterm(self, _signum: int, _frame: FrameType | None) -> None:
        logger.info("Worker {} received SIGTERM, stopping", self.name)
        self._stop_event.set()

    def should_stop(self) -> bool:
        parent = mp.parent_process()
        return super().should_stop() or (parent is not None and not parent.is_alive())

    def _report_failure(self, error: BaseException) -> None:
        # Message before flag: a parent that sees the flag then reads the queue.
        self._failures.put(_describe(error))
        self._failed_event.set()

    def _reported_failure(self) -> str | None:
        if self._failure is None and self._failed_event.is_set():
            try:
                self._failure = self._failures.get(timeout=1.0)
            except Exception:  # an empty or closed queue tells us nothing more
                self._failure = "no reason reported"
        return self._failure

    def _exit_description(self) -> str:
        return f"exited with code {self.exitcode} before it was ready"

    def _kill(self) -> None:
        logger.warning("Killing worker {}", self.name)
        self.kill()
        self.join(timeout=1.0)
