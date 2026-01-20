"""Registry for managing camera workers."""

import asyncio
from types import TracebackType
from uuid import UUID

from loguru import logger

from .camera_worker import CameraWorker


class CameraWorkerRegistry:
    """Manages lifecycle of camera workers."""

    def __init__(self, max_workers: int = 10, shutdown_timeout_s: float = 10.0):
        self._workers: dict[UUID, CameraWorker] = {}
        self._lock = asyncio.Lock()
        self._max_workers = max_workers
        self._shutdown_timeout_s = shutdown_timeout_s

    async def create_and_register(
        self,
        worker_id: UUID,
        worker: CameraWorker,
    ) -> None:
        """
        Create and register a new camera worker.

        Raises:
            ValueError: If worker_id already exists or max_workers exceeded.
        """
        async with self._lock:
            if worker_id in self._workers:
                raise ValueError(f"Worker {worker_id} already exists")

            if len(self._workers) >= self._max_workers:
                raise ValueError(f"Maximum number of workers ({self._max_workers}) reached")

            self._workers[worker_id] = worker
            logger.info(
                f"Camera worker registered: {worker_id} ({worker.config.name}). "
                f"Total: {len(self._workers)}/{self._max_workers}"
            )

    async def unregister(self, worker_id: UUID) -> None:
        """Unregister and shutdown a worker."""
        async with self._lock:
            worker = self._workers.pop(worker_id, None)

        if worker:
            try:
                await worker.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down worker {worker_id}: {e}")
            logger.info(f"Camera worker unregistered: {worker_id}")

    async def get(self, worker_id: UUID) -> CameraWorker | None:
        """Get a worker by id."""
        return self._workers.get(worker_id)

    def list_all(self) -> list[CameraWorker]:
        """List all active workers."""
        return list(self._workers.values())

    def get_status_summary(self) -> dict:
        """Get summary of all worker statuses."""
        return {
            "total_workers": len(self._workers),
            "max_workers": self._max_workers,
            "workers": {
                str(worker_id): {
                    "name": worker.config.name,
                    "state": worker.state.value,
                    "error": worker.error_message,
                }
                for worker_id, worker in self._workers.items()
            },
        }

    async def shutdown_all(self) -> None:
        """Gracefully shutdown all workers."""
        logger.info(f"Shutting down {len(self._workers)} camera workers...")

        async with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()

        # Shutdown all concurrently
        tasks = [worker.shutdown() for worker in workers]

        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self._shutdown_timeout_s,
                )
            except TimeoutError:
                logger.error(f"Some workers did not shutdown within {self._shutdown_timeout_s}s")

        logger.info("All camera workers shut down")

    async def __aenter__(self):
        """Async context manager support."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup on context exit."""
        await self.shutdown_all()
