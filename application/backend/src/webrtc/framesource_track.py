import asyncio
import logging

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from frame_source import FrameSourceFactory

from schemas import Camera, CameraProfile

logger = logging.getLogger(__name__)


class EmptyFrameError(Exception):
    pass


class FrameSourceVideoStreamTrack(VideoStreamTrack):
    """Video stream track that captures frames directly from FrameSource."""

    def __init__(self, camera: Camera, stream_profile: CameraProfile):
        super().__init__()
        _id: str | int
        self.driver = camera.driver
        self.device = camera.port_or_device_id
        self.cam = FrameSourceFactory.create(
            camera.driver,
            camera.port_or_device_id,
            width=stream_profile.width,
            height=stream_profile.height,
            fps=stream_profile.fps,
        )
        self.cam.connect()
        self._running = True
        self._last_frame: VideoFrame | None = None
        self._error_counter = 0

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = None

        try:
            frame_data = await asyncio.to_thread(self._read_frame)
            frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
            self._last_frame = frame
            self._error_counter = 0
            frame.pts = pts
            frame.time_base = time_base
        except EmptyFrameError:
            self._error_counter += 1
            if self._error_counter > 100:
                raise RuntimeError("Failed to receive frames for too long")
            if self._last_frame is not None:
                return self._last_frame
            raise RuntimeError("No received frame available")
        except Exception as e:
            logger.error(f"Error capturing from {self.device} at driver {self.driver}: {e}")
            # fallback gray frame
            fallback = np.full((64, 64, 3), 16, dtype=np.uint8)
            frame = VideoFrame.from_ndarray(fallback, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
        return frame

    def _read_frame(self) -> np.ndarray:
        ret, frame = self.cam.read()
        if not ret:
            raise EmptyFrameError("Got empty frame from camera")
        return frame

    def stop(self) -> None:
        self._running = False
        self.cam.disconnect()
        super().stop()
