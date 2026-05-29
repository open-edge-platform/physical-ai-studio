from __future__ import annotations

import asyncio

from fastapi.websockets import WebSocketDisconnect
from loguru import logger
from physicalai.capture.errors import CaptureError, CaptureTimeoutError

from schemas.project_camera import Camera
from utils.camera_factory import build_shared_camera
from utils.jpeg import encode_jpeg_rgb
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState, WorkerStatus


class CameraWorker(TransportWorker[Camera]):
    """Orchestrates camera streaming over configurable transport."""

    def __init__(
        self,
        config: Camera,
        transport: WorkerTransport,
        is_locked: bool = False,
    ) -> None:
        super().__init__(transport)
        self.config = config
        # When locked by active recording: attach passively (don't reconfigure
        # the camera). Otherwise let preview drive settings so the camera
        # edit page sees live changes.
        self.cam = build_shared_camera(
            config=config,
            validate_on_connect=False,
            overwrite_settings=not is_locked,
        )
        self._stop_requested: bool = False

    async def run(self) -> None:
        """Main worker loop."""
        try:
            await self.transport.connect()

            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, self.cam.connect)
            except CaptureError as exc:
                self.state = WorkerState.ERROR
                self.error_message = str(exc)
                logger.error(f"Failed to connect camera {self.config.name}: {exc}")
                await self.transport.send_json(
                    WorkerStatus(state=self.state, config=self.config, message=str(exc)).to_json()
                )
                return

            self.state = WorkerState.RUNNING
            status_msg = self._build_connect_message()
            await self.transport.send_json(
                WorkerStatus(
                    state=self.state,
                    config=self.config,
                    message=status_msg,
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
        while not self._stop_requested:
            try:
                frame = await self.cam.async_read(timeout=1.0)
            except CaptureTimeoutError:
                continue
            except CaptureError as exc:
                logger.error(f"capture error on {self.config.fingerprint}: {exc}")
                break

            await self.transport.send_bytes(encode_jpeg_rgb(frame.data))

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

    def _build_connect_message(self) -> str:
        actual_w = getattr(self.cam, "actual_width", None)
        actual_h = getattr(self.cam, "actual_height", None)
        actual_fps = getattr(self.cam, "actual_fps", None)
        if actual_w is None or actual_h is None:
            return "Camera connected"

        payload = self.config.payload
        req_w = getattr(payload, "width", None)
        req_h = getattr(payload, "height", None)
        req_fps = getattr(payload, "fps", None)

        w_mismatch = req_w is not None and actual_w != req_w
        h_mismatch = req_h is not None and actual_h != req_h
        fps_mismatch = req_fps is not None and actual_fps not in (None, 0, req_fps)

        if not (w_mismatch or h_mismatch or fps_mismatch):
            return "Camera connected"

        return (
            f"showing {actual_w}x{actual_h}@{actual_fps}fps"
            f" (requested {req_w}x{req_h}@{req_fps}fps, another stream owns this camera)"
        )

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if not self.cam.is_connected:
            return
        logger.info(f"Shutting down camera: {self.config.name}")
        self._stop_requested = True
        await super().shutdown()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.cam.disconnect)
