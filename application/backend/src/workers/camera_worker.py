import asyncio

import cv2
import numpy as np
from fastapi.websockets import WebSocketDisconnect
from frame_source import FrameSourceFactory
from frame_source.video_capture_base import VideoCaptureBase
from loguru import logger

from schemas.project_camera import Camera
from utils.async_camera_capture import AsyncCameraCapture
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState, WorkerStatus


def create_frames_source_from_camera(camera: Camera) -> VideoCaptureBase:
    """Very FrameSource factory call from camera schema object."""
    return FrameSourceFactory.create(
        "webcam" if camera.driver == "usb_camera" else camera.driver,
        camera.fingerprint,
        **camera.payload.model_dump(),
    )


class EmptyFrameError(Exception):
    pass


class CameraWorker(TransportWorker[Camera]):
    """Orchestrates camera streaming over configurable transport."""

    def __init__(
        self,
        config: Camera,
        transport: WorkerTransport,
    ):
        super().__init__(transport)
        self.config = config

        cam = create_frames_source_from_camera(config)
        self.connection = AsyncCameraCapture(camera=cam, fps=config.payload.fps)

    async def run(self) -> None:
        """Main worker loop."""
        try:
            await self.transport.connect()

            self.state = WorkerState.RUNNING
            await self.transport.send_json(
                WorkerStatus(
                    state=self.state,
                    config=self.config,
                    message="Camera connected",
                ).to_json()
            )

            await self.run_concurrent(
                asyncio.create_task(self._capture_loop()),
                asyncio.create_task(self._command_loop()),
            )

        except Exception as e:
            self.state = WorkerState.ERROR
            self.error_message = str(e)
            logger.error(f"Worker error: {e}")
            await self.transport.send_json(WorkerStatus(state=self.state, message=str(e)).to_json())
        finally:
            await self.shutdown()

    async def _capture_loop(self) -> None:
        """Continuously capture and send frames."""
        try:
            await self.connection.start(self._send_frame)
            await self.connection.wait()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            self._stop_requested = True
            raise

    async def _send_frame(self, frame: np.ndarray) -> None:
        """Send frame via transport."""
        success, jpeg = cv2.imencode(".jpg", frame)
        if not success or jpeg is None:
            raise RuntimeError("Failed to encode frame")
        await self.transport.send_bytes(jpeg.tobytes())

    async def _command_loop(self) -> None:
        """Handle incoming commands from client."""
        try:
            while not self._stop_requested:
                command = await self.transport.receive_command()
                if command:
                    await self._handle_command(command)
        except (WebSocketDisconnect, RuntimeError):
            self._stop_requested = True
        except asyncio.CancelledError:
            pass

    async def _handle_command(self, command: dict) -> None:
        """Handle a single command."""
        event = command.get("event")

        match event:
            case "ping":
                await self.transport.send_json(WorkerStatus(state=WorkerState.RUNNING, message="pong").to_json())
            case "disconnect":
                logger.info("Client requested disconnect")
                self._stop_requested = True

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info(f"Shutting down camera: {self.config.name}")
        await super().shutdown()
        await self.connection.stop()
