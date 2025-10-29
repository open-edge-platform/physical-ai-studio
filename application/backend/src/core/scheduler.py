import asyncio
from loguru import logger
import multiprocessing as mp
import os
from typing import TYPE_CHECKING
from workers import TrainingWorker

import psutil

if TYPE_CHECKING:
    import threading


class Scheduler:
    """Manages application processes and threads."""

    def __init__(self) -> None:
        logger.info("Initializing Scheduler...")
        # Event to sync all processes on application shutdown
        self.mp_stop_event = mp.Event()
        self.training_interrupt_event = mp.Event()

        self.processes: list[mp.Process] = []
        self.threads: list[threading.Thread] = []
        self.tasks: list[asyncio.Task] = []
        logger.info("Scheduler initialized")

    def start_workers(self) -> None:
        training_proc = TrainingWorker(
            stop_event=self.mp_stop_event,
            interrupt_event=self.training_interrupt_event,
        )
        training_proc.daemon = False
        training_proc.start()
        self.processes.extend([training_proc])
        self.tasks.extend([asyncio.create_task(training_proc.event_processor())])

    def shutdown(self) -> None:
        """Shutdown all processes gracefully"""
        logger.info("Initiating graceful shutdown...")

        # Signal all processes to stop
        self.mp_stop_event.set()

        # Get current process info for debugging
        pid = os.getpid()
        cur_process = psutil.Process(pid)
        alive_children = [child.pid for child in cur_process.children(recursive=True) if child.is_running()]
        logger.debug(f"Alive children of process '{pid}': {alive_children}")

        # Join threads first
        for thread in self.threads:
            if thread.is_alive():
                logger.debug(f"Joining thread: {thread.name}")
                thread.join(timeout=10)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not terminate within timeout")

        # Join processes in reverse order so that consumers are terminated before producers.
        for process in self.processes[::-1]:
            if process.is_alive():
                logger.debug(f"Joining process: {process.name}")
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning("Force terminating process: %s", process.name)
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        logger.error("Force killing process %s", process.name)
                        process.kill()

        for task in self.tasks:
            task.cancel()

        logger.info("All workers shut down gracefully")

        # Clear references
        self.processes.clear()
        self.threads.clear()
