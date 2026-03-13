import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from frame_source.video_capture_base import VideoCaptureBase
from loguru import logger
from tenacity import RetryError


class EmptyFrameError(RuntimeError):
    pass


@dataclass
class LatestFrame:
    frame: np.ndarray | None = None
    t_perf: float = 0.0
    ok: bool = False
    error: str | None = None
    seq: int = 0  # increments on every successful new frame


class AsyncCameraCapture:
    """
    Continuously captures frames from a camera with connect() and synchronous read().
    - Runs read() in a thread via asyncio.to_thread so the event loop stays unblocked
    - Retries transient empty frames (tenacity)
    - Caches last good frame and can serve it if camera glitches
    - Stores only the latest frame (no queue growth)
    """

    MAX_CONSECUTIVE_ERRORS = 10

    def __init__(
        self,
        camera: VideoCaptureBase,
        fps: float,
        process_fn: Callable[[np.ndarray], np.ndarray] | None = None,  # e.g. BGR->RGB
        use_cached_on_failure: bool = True,
    ):
        self.camera = camera
        self.fps = float(fps)
        self.process_fn = process_fn
        self.use_cached_on_failure = use_cached_on_failure

        self._latest = LatestFrame()
        self._lock = asyncio.Lock()
        self._stop_evt = asyncio.Event()
        self._task: asyncio.Task | None = None

        self._last_good_frame: np.ndarray | None = None
        self._seq = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cam")

    async def start(self) -> None:
        """Connect camera and start background capture task."""
        # connect() is sync in your world; keep it off the event loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.camera.connect)  # same thread as reads
        self._stop_evt.clear()
        if self._task is None:
            self._task = asyncio.create_task(self._capture_loop(), name="AsyncCameraCaptureLoop")

    async def stop(self) -> None:
        """Stop background task and disconnect camera."""
        self._stop_evt.set()
        if self._task is not None:
            try:
                await self._task
            finally:
                self._task = None
        # disconnect() may or may not exist; be defensive.
        disconnect = getattr(self.camera, "disconnect", None)
        if callable(disconnect):
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, disconnect)
            except Exception:
                logger.info("Failed disconnecting camera. Ignoring")

    async def get_latest(self) -> tuple[np.ndarray | None, float, bool, str | None, int]:
        """Non-blocking snapshot: (frame, t_perf, ok, error, seq)."""
        async with self._lock:
            lf = self._latest
            return lf.frame, lf.t_perf, lf.ok, lf.error, lf.seq

    async def _set_latest(self, frame: np.ndarray | None, ok: bool, error: str | None, seq: int) -> None:
        async with self._lock:
            self._latest = LatestFrame(
                frame=frame,
                t_perf=time.perf_counter(),
                ok=ok,
                error=error,
                seq=seq,
            )

    async def _capture_loop(self) -> None:
        target_dt = 1.0 / self.fps if self.fps > 0 else 0.0
        logger.info("Start capture loop")
        loop = asyncio.get_running_loop()
        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            error = None

            try:
                frame = await loop.run_in_executor(self._executor, self._read_frame_with_retry)
                if self.process_fn is not None:
                    frame = self.process_fn(frame)

                self._last_good_frame = frame
                self._seq += 1
                await self._set_latest(frame=frame, ok=True, error=None, seq=self._seq)

            except RetryError as e:
                error = f"retry_exhausted: {e}"
                logger.error(error)
                if self.use_cached_on_failure and self._last_good_frame is not None:
                    # Serve cached frame (don't bump seq; not a new frame)
                    await self._set_latest(
                        frame=self._last_good_frame,
                        ok=True,
                        error="using_cached_frame",
                        seq=self._seq,
                    )
                else:
                    await self._set_latest(frame=None, ok=False, error="no_frame_available", seq=self._seq)
            except asyncio.CancelledError:
                logger.warning("Capture loop was cancelled while awaiting run_in_executor()")
                raise
            except Exception as e:
                # Keep loop alive unless you *want* to crash on unexpected errors.
                error = f"capture_error: {e}"
                logger.exception(error)
                if self.use_cached_on_failure and self._last_good_frame is not None:
                    await self._set_latest(frame=self._last_good_frame, ok=True, error=error, seq=self._seq)
                else:
                    await self._set_latest(frame=None, ok=False, error=error, seq=self._seq)
            # FPS pacing (also exits quickly if stop is set)
            if target_dt > 0:
                elapsed = time.perf_counter() - t0
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    try:
                        await asyncio.wait_for(self._stop_evt.wait(), timeout=sleep_time)
                    except TimeoutError:
                        pass

    def _read_frame_with_retry(self) -> np.ndarray:
        ret, frame = self.camera.read()
        if not ret or frame is None:
            raise EmptyFrameError("Empty frame from camera")
        return frame
