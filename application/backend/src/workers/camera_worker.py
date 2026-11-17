import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext

import cv2
from loguru import logger
from physicalai.capture import SharedCamera

from runtime_session.bus import Bus, publish_status
from runtime_session.store import ImageSpec, Store
from schemas.project_camera import Camera
from schemas.runtime_session import ComponentState
from utils.camera_factory import build_shared_camera
from workers.base import ThreadWorker
from workers.rate import Rate


def camera_image_spec(config: Camera) -> ImageSpec:
    """The frame shape a camera's store slot has."""
    # TODO explicitly add width, height to ip camera
    return ImageSpec(height=config.payload.height or 480, width=config.payload.width or 640)


class CameraWorker(ThreadWorker[SharedCamera]):
    """Read one camera into its store slot at the camera's fps.

    A thread: ``SharedCamera`` already runs the capture in a publisher process.

    A camera that drops off USB is reopened: its publisher exits once the device
    stops answering, after which ``read_latest`` keeps returning the last frame.
    So a camera that sends nothing new for :attr:`stale_after_s` is released,
    reported as ``starting`` and reopened every :attr:`reconnect_interval_s`
    until it is back, without writing the frozen frame in the meantime.
    """

    stale_after_s: float = 0.5
    """How long a camera may go without a new frame before it counts as gone."""

    reconnect_interval_s: float = 1.0
    """How long to wait between attempts to reopen a camera that is gone."""

    def __init__(
        self,
        config: Camera,
        store: Store,
        key: str,
        is_locked: bool = False,
        *,
        session_id: str | None = None,
        component: str | None = None,
    ) -> None:
        """Pass ``session_id`` and ``component`` to report a failure on that session's bus."""
        super().__init__(name=f"CameraWorker-{config.name}")
        self.config = config
        self.session_id = session_id
        self.component = component
        self._bus: Bus | None = None
        self._camera: SharedCamera | None = None
        self.store = store
        self.key = key
        self.is_locked = is_locked
        self._image = camera_image_spec(config)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[SharedCamera]:
        camera = self._open()
        self._camera = camera
        try:
            with Bus.join(self.session_id) if self.session_id else nullcontext() as self._bus:
                yield camera
        finally:
            logger.info("Camera run loop stopped. Disconnecting")
            # Whichever camera is open now: a reconnect replaces the one yielded.
            if self._camera is not None and self._camera.is_connected:
                self._camera.disconnect()

    def _open(self) -> SharedCamera:
        camera = build_shared_camera(
            config=self.config,
            validate_on_connect=False,
            overwrite_settings=not self.is_locked,
        )
        camera.connect()
        return camera

    async def run_loop(self, context: SharedCamera) -> None:
        try:
            await self._capture(context)
        except Exception as error:
            self.store.invalidate_observation(self.key)
            self._publish("failed", str(error))
            raise

    async def _capture(self, context: SharedCamera) -> None:
        rate = Rate(self.config.payload.fps, name=self.name)
        size = (self._image.width, self._image.height)
        camera: SharedCamera | None = context
        previous = None
        reconnecting = False
        last_new_frame = time.monotonic()
        while camera is not None and not self.should_stop():
            frame = camera.read_latest()
            if frame is not previous:
                previous, last_new_frame = frame, time.monotonic()
                data = frame.data if frame.data.shape[:2] == self._image.shape[:2] else cv2.resize(frame.data, size)
                self.store.write_observation(self.key, data, timestamp=frame.timestamp)
                if reconnecting:
                    self._publish("running")
                    reconnecting = False
            elif time.monotonic() - last_new_frame > self.stale_after_s:
                reconnecting = True
                camera = await self._reconnect(camera)
                previous, last_new_frame = None, time.monotonic()
                continue
            await rate.sleep()

    async def _reconnect(self, lost: SharedCamera) -> SharedCamera | None:
        """Release a camera that stopped sending frames and reopen it until it is back.

        Returns:
            The reopened camera, or ``None`` if the worker was asked to stop first.
        """
        logger.warning("Camera {!r} sent no frame for {:.1f}s; reconnecting", self.config.name, self.stale_after_s)
        self.store.invalidate_observation(self.key)
        self._publish("starting", f"no frames for {self.stale_after_s:.1f}s; reconnecting")
        self._camera = None
        try:
            lost.disconnect()
        except Exception:
            logger.exception("Disconnecting the lost camera {!r} failed", self.config.name)
        attempts = 0
        while not self.should_stop():
            await self.sleep(self.reconnect_interval_s)
            if self.should_stop():
                break
            attempts += 1
            try:
                camera = await asyncio.to_thread(self._open)
            except Exception as error:
                logger.debug("Camera {!r} not back yet (attempt {}): {}", self.config.name, attempts, error)
                continue
            self._camera = camera
            logger.info("Camera {!r} reconnected after {} attempt(s)", self.config.name, attempts)
            return camera
        return None

    def _publish(self, state: ComponentState, error: str | None = None) -> None:
        if self._bus is not None and self.component is not None:
            publish_status(self._bus, self.component, state, error)
