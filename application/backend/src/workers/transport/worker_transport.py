from abc import ABC, abstractmethod
from typing import Any


class WorkerTransport(ABC):
    """Abstract base for worker communication transports."""

    @abstractmethod
    async def connect(self) -> None:
        """Initialize the transport."""

    @abstractmethod
    async def send_json(self, data: Any) -> None:
        """Send JSON message."""

    @abstractmethod
    async def send_bytes(self, data: bytes) -> None:
        """Send binary data."""

    @abstractmethod
    async def receive_command(self) -> dict | None:
        """Receive a command from the client. Returns None on timeout."""

    @abstractmethod
    async def close(self) -> None:
        """Close the transport connection."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if transport is active."""
