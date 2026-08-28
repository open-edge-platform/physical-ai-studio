from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from runtime.contract import AckData

if TYPE_CHECKING:
    from collections.abc import Callable

_SHUTDOWN = object()


@dataclass(frozen=True, slots=True)
class _Job:
    name: str
    fn: Callable[[], None]
    request_id: str | None


class CommandWorker:
    """Run blocking session commands off the control thread, one at a time.

    Ordering is the property that matters: ``save_episode`` then
    ``discard_episode`` must not interleave. Model loads stay on their own
    loader thread so a dataset ``copytree`` cannot sit in front of them.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue[_Job | object] = queue.Queue()
        self._results: dict[str, AckData] = {}
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._thread: threading.Thread | None = None

    def submit(self, name: str, fn: Callable[[], None], *, request_id: str | None = None) -> None:
        """Enqueue ``fn`` for FIFO execution. Optional ``request_id`` can be waited on."""
        with self._lock:
            if self._closed:
                raise RuntimeError(f"Cannot submit {name} after the command worker has shut down")
            if request_id is not None:
                self._events[request_id] = threading.Event()
            self._ensure_thread_locked()
        self._jobs.put(_Job(name=name, fn=fn, request_id=request_id))

    def wait(self, request_id: str, timeout: float) -> AckData:
        """Block until the job with ``request_id`` finishes, or the timeout elapses."""
        with self._lock:
            event = self._events.get(request_id)
        if event is None:
            raise KeyError(f"No command worker job with request_id {request_id!r}")
        if not event.wait(timeout):
            return AckData(request_id=request_id, ok=False, error=f"Command timed out after {timeout:.0f}s")
        with self._lock:
            self._events.pop(request_id, None)
            return self._results.pop(request_id)

    def shutdown(self, timeout: float) -> None:
        """Drain queued work, then stop. A stop cannot strand an in-flight save."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is None:
            return
        self._jobs.put(_SHUTDOWN)
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.error(
                "Command worker did not finish within {}s; in-flight recording work may be lost",
                timeout,
            )

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is _SHUTDOWN:
                return
            if not isinstance(job, _Job):
                continue
            try:
                job.fn()
            except Exception as exc:
                logger.exception("Runtime command {} failed", job.name)
                ack = AckData(
                    request_id=job.request_id or "",
                    ok=False,
                    error=str(exc) or f"{job.name} failed",
                )
            else:
                ack = AckData(request_id=job.request_id or "", ok=True)
            if job.request_id is None:
                continue
            with self._lock:
                self._results[job.request_id] = ack
                event = self._events.get(job.request_id)
            if event is not None:
                event.set()

    def _ensure_thread_locked(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="runtime-command-worker",
            daemon=True,
        )
        self._thread.start()
