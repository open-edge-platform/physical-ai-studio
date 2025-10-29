from collections import defaultdict
import asyncio
from enum import StrEnum
import multiprocessing as mp
from typing import Callable, Sequence
from loguru import logger


class EventType(StrEnum):
    MODEL_UPDATE = "MODEL_UPDATE"
    JOB_UPDATE = "JOB_UPDATE"


class EventProcessor:
    def __init__(self, event_queue: mp.Queue) -> None:
        self._event_handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self._pending_events = []
        self.task = asyncio.create_task(self.processor())
        self.queue = event_queue

    def subscribe(self, event_types: Sequence[EventType], handler: Callable) -> None:
        """Subscribe to events of event bus."""
        logger.info("subscribing")
        for event_type in event_types:
            self._event_handlers[event_type].append(handler)

    def unsubscribe(self, event_types: Sequence[EventType], handler: Callable) -> None:
        """Subscribe to events of event bus."""
        logger.info("unsubscribing")
        for event_type in event_types:
            self._event_handlers[event_type] = [h for h in self._event_handlers[event_type] if h != handler]

    async def processor(self):
        """Run inside FastAPI (async task). Empties queus and dispatches events with payloads."""
        logger.info("Started event processor.")
        try:
            while True:
                try:
                    event, payload = self.queue.get_nowait()
                    logger.info(event)
                    for handler in self._event_handlers[event]:
                        if asyncio.iscoroutinefunction(handler):
                            asyncio.create_task(handler(event, payload))
                        else:
                            handler(event, payload)
                except mp.queues.Empty:
                    await asyncio.sleep(0.05)

        except Exception as e:
            logger.error(f"Outgoing task stopped: {e}")

    def shutdown(self):
        """Stop processor task."""
        logger.info("Shutdown event processor.")
        self.task.cancel()
