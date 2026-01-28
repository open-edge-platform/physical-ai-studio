from frame_source.video_capture_base import VideoCaptureBase
import asyncio
import time
from collections.abc import Awaitable, Callable

import cv2
import numpy as np
from fastapi.websockets import WebSocketDisconnect
from frame_source import FrameSourceFactory
from loguru import logger
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from schemas.project_camera import Camera
from webrtc.framesource_track import EmptyFrameError
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState, WorkerStatus


def create_frames_source_from_camera(camera: Camera) -> VideoCaptureBase:
    """Very FrameSource factory call from camera schema object."""
    return FrameSourceFactory.create(
        "webcam" if camera.driver == "usb_camera" else camera.driver,
        camera.fingerprint,
        **camera.payload.model_dump(),
    )


class CameraConnectionManager:
    """Handles camera connection."""

    MAX_ATTEMPTS = 3
    INITIAL_BACKOFF = 1.0
    MAX_BACKOFF = 10.0

    camera_connection: VideoCaptureBase | None

    def __init__(self, camera: Camera):
        self.camera = camera
        self.camera_connection = None

    @retry(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=INITIAL_BACKOFF, max=MAX_BACKOFF),
        reraise=True,
    )
    def _connect_impl(self):
        """Connect to camera (with automatic retry via tenacity)."""

        logger.info("Connecting to camera: {}", self.camera.name)
        self.camera_connection = create_frames_source_from_camera(self.camera)
        self.camera_connection.connect()
        return self.camera_connection

    async def connect_async(self) -> None:
        """Async wrapper for connection."""
        try:
            await asyncio.to_thread(self._connect_impl)
            logger.info(f"Camera connected: {self.camera.name}")
        except RetryError as e:
            raise RuntimeError(
                f"Failed to connect to {self.camera.name} after "
                f"{self.MAX_ATTEMPTS} attempts: {e.last_attempt.exception()}"
            ) from e

    async def disconnect(self) -> None:
        """Safely disconnect from camera."""
        if self.camera_connection is None:
            return

        try:
            await asyncio.to_thread(self.camera_connection.disconnect)
            logger.debug(f"Camera disconnected: {self.camera.name}")
        except Exception as e:
            logger.warning(f"Error disconnecting camera: {e}")
        finally:
            self.camera_connection = None


class FrameCapture:
    """Handles frame capture with error resilience."""

    MAX_CONSECUTIVE_ERRORS = 10

    def __init__(self, connection_manager: CameraConnectionManager, fps: int):
        self.connection = connection_manager
        self.fps = fps
        self.last_frame: np.ndarray | None = None

    async def capture_loop(self, send_frame_fn: Callable[[np.ndarray], Awaitable[None]]) -> None:
        """Continuously capture and send frames at target FPS."""
        target_frame_time = 1.0 / self.fps

        while True:
            start_time = time.perf_counter()

            try:
                frame = await asyncio.to_thread(self._read_frame)
                self.last_frame = frame
                await send_frame_fn(frame)
            except RetryError as e:
                logger.error(f"Failed to capture frame after {self.MAX_CONSECUTIVE_ERRORS} attempts")
                if self.last_frame is not None:
                    logger.info("Returning cached frame")
                    await send_frame_fn(self.last_frame)
                else:
                    raise RuntimeError("No frame available") from e

            except Exception as e:
                logger.error(f"Frame capture error: {e}")
                raise

            elapsed = time.perf_counter() - start_time
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    @retry(
        stop=stop_after_attempt(MAX_CONSECUTIVE_ERRORS),
        wait=wait_exponential(multiplier=0.1, min=0, max=1.0),
        retry=retry_if_exception_type(EmptyFrameError),
        reraise=True,
    )
    def _read_frame(self) -> np.ndarray:
        """Synchronous frame read with automatic retry on transient errors."""
        if self.connection.camera_connection is None:
            raise RuntimeError("No camera connection")

        ret, frame = self.connection.camera_connection.read()
        if not ret or frame is None:
            raise EmptyFrameError("Empty frame from camera")

        return frame


class CameraWorker(TransportWorker[Camera]):
    """Orchestrates camera streaming over configurable transport."""

    def __init__(
        self,
        config: Camera,
        transport: WorkerTransport,
    ):
        super().__init__(transport)
        self.config = config
        self.connection = CameraConnectionManager(config)
        self.frame_capture = FrameCapture(self.connection, config.payload.fps)

    async def run(self) -> None:
        """Main worker loop."""
        try:
            await self.transport.connect()
            await self.connection.connect_async()

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
            await self.frame_capture.capture_loop(self._send_frame)
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
        await self.connection.disconnect()
