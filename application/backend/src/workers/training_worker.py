from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass

from services.training_service import TrainingService
from workers.base import BaseProcessWorker

MAX_CONCURRENT_TRAINING = 1
SCHEDULE_INTERVAL_SEC = 5

import loguru
from loguru import logger

class TrainingWorker(BaseProcessWorker):
    ROLE = "Training"

    def __init__(self, stop_event: EventClass):
        super().__init__(stop_event=stop_event)

    async def run_loop(self) -> None:
        """Main training loop that polls for jobs and manages concurrent training tasks."""
        print("run loop")
        training_service = TrainingService()
        running_tasks: set[asyncio.Task] = set()

        logger.info("Training Worker is running")
        while not self.should_stop():
            try:
                # Clean up completed tasks
                running_tasks = {task for task in running_tasks if not task.done()}

                # Start new training if under capacity limit
                # Using async tasks allows:
                # - Multiple training jobs to run concurrently
                # - Event loop to remain responsive for shutdown signals
                if len(running_tasks) < MAX_CONCURRENT_TRAINING:
                    running_tasks.add(asyncio.create_task(training_service.train_pending_job(self._stop_event)))
            except Exception as e:
                logger.error(f"Error occurred in training loop: {e}", exc_info=True)

            # Check for shutdown signals frequently
            for _ in range(SCHEDULE_INTERVAL_SEC * 2):
                if self.should_stop():
                    break
                await asyncio.sleep(0.5)

        # Cancel any remaining tasks on shutdown
        for task in running_tasks:
            task.cancel()
        if running_tasks:
            try:
                await asyncio.gather(*running_tasks, return_exceptions=True)
            except Exception as e:
                # Log exceptions during cancellation to ensure clean shutdown and aid debugging
                logger.error(f"Exception during task cancellation: {e}", exc_info=True)

    def setup(self) -> None:
        super().setup()
        # TODO Nuke orphans
        #with logger.contextualize(worker=self.__class__.__name__):
        #    asyncio.run(TrainingService.abort_orphan_jobs())

    def teardown(self) -> None:
        super().teardown()
        # TODO Nuke orphans
        #with logger.contextualize(worker=self.__class__.__name__):
        #    asyncio.run(TrainingService.abort_orphan_jobs())