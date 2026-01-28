"""Registry for managing robot workers."""

import asyncio
from types import TracebackType
from typing import Self
from uuid import UUID

from loguru import logger

from workers.robots.robot_worker import RobotWorker


class RobotWorkerRegistry:
    """
    Manages the lifecycle and registration of active robot workers.

    This registry ensures that the number of active robot workers is limited
    and provides methods for starting, stopping, and monitoring them.
    """

    def __init__(self, max_workers: int = 10, shutdown_timeout_s: float = 10.0) -> None:
        """
        Initialize the registry.

        Args:
            max_workers: Maximum number of concurrent workers allowed.
            shutdown_timeout_s: Seconds to wait for workers to shutdown gracefully.
        """
        self._workers: dict[UUID, RobotWorker] = {}
        self._lock = asyncio.Lock()
        self._max_workers = max_workers
        self._shutdown_timeout_s = shutdown_timeout_s

    async def create_and_register(
        self,
        worker_id: UUID,
        worker: RobotWorker,
    ) -> None:
        """
        Register a new robot worker in the registry.

        Args:
            worker_id: Unique identifier for the worker.
            worker: The RobotWorker instance to register.

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
                f"Robot worker registered: {worker_id} ({worker.robot.id}). "
                f"Total: {len(self._workers)}/{self._max_workers}"
            )

    async def unregister(self, worker_id: UUID) -> None:
        """
        Unregister and initiate shutdown for a specific worker.

        Args:
            worker_id: The ID of the worker to unregister.
        """
        async with self._lock:
            worker = self._workers.pop(worker_id, None)

        if worker:
            try:
                await worker.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down worker {worker_id}: {e}")
            logger.info(f"Robot worker unregistered: {worker_id}")

    async def get(self, worker_id: UUID) -> RobotWorker | None:
        """
        Retrieve a worker by its ID.

        Args:
            worker_id: The ID of the worker.

        Returns:
            The RobotWorker instance or None if not found.
        """
        return self._workers.get(worker_id)

    def list_all(self) -> list[RobotWorker]:
        """
        Get a list of all currently registered workers.

        Returns:
            List of RobotWorker instances.
        """
        return list(self._workers.values())

    def get_status_summary(self) -> dict:
        """
        Generate a summary of the status of all registered workers.

        Returns:
            Dictionary containing total count and individual worker statuses.
        """
        return {
            "total_workers": len(self._workers),
            "max_workers": self._max_workers,
            "workers": {
                str(worker_id): {
                    "name": worker.robot.id,
                    "state": worker.state.value,
                    "error": getattr(worker, "error_message", None),
                }
                for worker_id, worker in self._workers.items()
            },
        }

    async def shutdown_all(self) -> None:
        """
        Concurrently shutdown all registered workers and clear the registry.
        """
        logger.info(f"Shutting down {len(self._workers)} robot workers...")

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

        logger.info("All robot workers shut down")

    async def __aenter__(self) -> Self:
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
