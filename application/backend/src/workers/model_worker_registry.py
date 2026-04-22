"""Registry for managing pre-spawned model worker processes."""

import asyncio
import time
from multiprocessing.synchronize import Event as EventClass
from uuid import UUID, uuid4

from loguru import logger

from schemas import Model

from .model_worker import ModelWorker


class ModelWorkerRegistry:
    """
    Manages a pool of pre-spawned idle ModelWorker processes.

    Workers are spawned at startup (not on demand) to avoid Python's spawn
    overhead mid-lifetime. When a model is requested, an idle worker receives
    a load command and transitions to LOADED state. On release it returns to
    idle, ready to accept the next model.
    """

    def __init__(
        self,
        max_workers: int,
        stop_event: EventClass,
        shutdown_timeout_s: float = 10.0,
    ) -> None:
        self._stop_event = stop_event
        self._max_workers = max_workers
        self._shutdown_timeout_s = shutdown_timeout_s
        self._lock = asyncio.Lock()

        self._workers: dict[UUID, ModelWorker] = {}
        self._idle: set[UUID] = set()
        self._busy: set[UUID] = set()

        self._spawn_workers()

    def _spawn_workers(self) -> None:
        """Pre-spawn all worker processes at registry creation time."""
        for _ in range(self._max_workers):
            worker_id = uuid4()
            worker = ModelWorker(stop_event=self._stop_event)
            worker.start()
            self._workers[worker_id] = worker
            self._idle.add(worker_id)
            logger.info(f"Model worker pre-spawned: {worker_id} (pid={worker.pid})")

    async def acquire(self, model: Model, backend: str) -> tuple[UUID, ModelWorker]:
        """
        Assign an idle worker to load the given model.

        Returns (worker_id, worker). Raises ValueError if no idle workers.
        """
        async with self._lock:
            if not self._idle:
                raise ValueError(f"No idle model workers available (all {self._max_workers} workers are busy)")
            worker_id = next(iter(self._idle))
            self._idle.discard(worker_id)
            self._busy.add(worker_id)

        worker = self._workers[worker_id]
        worker.load_model(model, backend)
        logger.info(f"Model worker {worker_id} acquired for model '{model.name}' ({backend})")
        return worker_id, worker

    async def release(self, worker_id: UUID) -> None:
        """
        Unload the model from a worker and return it to the idle pool.
        """
        async with self._lock:
            if worker_id not in self._busy:
                return
            worker = self._workers.get(worker_id)

        if worker is None:
            return

        worker.unload_model()
        # Poll until the worker clears model_loaded_event (signals unload complete)
        deadline = time.monotonic() + self._shutdown_timeout_s

        def _wait_for_unload() -> None:
            while worker.model_loaded_event.is_set():
                if time.monotonic() > deadline:
                    logger.warning(f"Timed out waiting for worker {worker_id} to unload")
                    break
                time.sleep(0.05)

        await asyncio.to_thread(_wait_for_unload)

        async with self._lock:
            self._busy.discard(worker_id)
            if worker_id in self._workers:
                self._idle.add(worker_id)

        logger.info(f"Model worker {worker_id} released and returned to idle pool")

    def get(self, worker_id: UUID) -> ModelWorker | None:
        return self._workers.get(worker_id)
