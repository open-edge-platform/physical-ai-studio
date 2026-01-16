import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar

from workers.transport.worker_transport import WorkerTransport


class WorkerState(str, Enum):
    """Worker lifecycle states."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class Serializable(Protocol):
    """Protocol for objects that can be serialized to dict."""

    def model_dump(self) -> dict: ...


ConfigT = TypeVar("ConfigT", bound=Serializable | dict)


@dataclass
class WorkerStatus(Generic[ConfigT]):
    state: WorkerState
    message: str = ""
    config: ConfigT | None = None

    def to_json(self) -> dict:
        return {
            "event": "status",
            "state": self.state.value,
            "message": self.message,
            "config": self._serialize_config(),
        }

    def _serialize_config(self) -> dict | None:
        """Convert config to serializable format."""
        if self.config is None:
            return None

        if hasattr(self.config, "model_dump"):
            return self.config.model_dump()
        if isinstance(self.config, dict):
            return self.config

        return None


class TransportWorker(Generic[ConfigT]):
    """Base class for workers that communicate via a transport."""

    def __init__(self, transport: WorkerTransport) -> None:
        self.transport = transport
        self.state = WorkerState.INITIALIZING
        self.error_message: str | None = None
        self._stop_requested = False

    async def run_concurrent(
        self,
        *tasks: asyncio.Task,
    ) -> None:
        """Run multiple tasks concurrently, stopping on first completion."""
        try:
            _, pending = await asyncio.wait(
                set(tasks),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, TimeoutError):
                    pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self.state = WorkerState.SHUTTING_DOWN
        self._stop_requested = True
        await self.transport.close()
        self.state = WorkerState.STOPPED
