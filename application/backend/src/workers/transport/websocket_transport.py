import asyncio
import json
from typing import Any

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect
from loguru import logger

from .worker_transport import WorkerTransport


class WebSocketTransport(WorkerTransport):
    """WebSocket transport for worker communication."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._connected = False

    async def connect(self) -> None:
        """Already accepted by FastAPI, mark as ready."""
        self._connected = True
        logger.info("WebSocket transport connected")

    async def send_json(self, data: Any) -> None:
        """Send JSON message over WebSocket."""
        try:
            await self.websocket.send_json(data)
        except Exception as e:
            logger.warning(f"Failed to send JSON: {e}")

    async def send_bytes(self, data: bytes) -> None:
        """Send binary data over WebSocket."""
        try:
            await self.websocket.send_bytes(data)
        except Exception as e:
            logger.warning(f"Failed to send bytes: {e}")

    async def receive_command(self) -> dict | None:
        """Receive command from WebSocket client."""
        try:
            async with asyncio.timeout(30.0):
                message_text = await self.websocket.receive_text()
            return json.loads(message_text)
        except TimeoutError:
            # Timeout is normal, just return None
            return None
        except WebSocketDisconnect:
            # Client disconnected, re-raise so worker can shut down
            raise
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from client: {e}")
            return None
        except RuntimeError as e:
            # "Cannot call receive once a disconnect message has been received"
            logger.debug(f"WebSocket already disconnected: {e}")
            raise WebSocketDisconnect(1000) from e
        except Exception as e:
            logger.warning(f"Error receiving command: {e}")
            raise WebSocketDisconnect(1000) from e

    async def close(self) -> None:
        """Close WebSocket connection."""
        self._connected = False
        try:
            await self.websocket.close(code=1000, reason="Normal shutdown")
        except Exception as e:
            logger.debug(f"Error closing websocket: {e}")

    async def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected
