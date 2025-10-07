import asyncio
import logging

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from frame_source import FrameSourceFactory

logger = logging.getLogger(__name__)


class FrameSourceVideoStreamTrack(VideoStreamTrack):
    """Video stream track that captures frames directly from FrameSource."""

    def __init__(self, driver: str, device: str):
        super().__init__()
        _id: str | int
        if device.isdigit():
            _id = int(device)
        else:
            _id = str(device)

        self.driver = driver
        self.device = _id
        self.cam = FrameSourceFactory.create(driver, _id)
        self.cam.connect()
        self._running = True

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = None

        try:
            frame_data = await asyncio.to_thread(self._read_frame)
            frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
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
            raise RuntimeError("Failed to read from device")
        return frame

    def stop(self) -> None:
        self._running = False
        self.cam.disconnect()
        super().stop()
