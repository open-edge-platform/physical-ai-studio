from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext

import cv2
from loguru import logger
from physicalai.capture import SharedCamera

from runtime_session.bus import Bus, publish_status
from runtime_session.store import ImageSpec, Store
from schemas.project_camera import Camera
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
    """

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
        self.store = store
        self.key = key
        self.is_locked = is_locked
        self._image = camera_image_spec(config)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[SharedCamera]:
        camera = build_shared_camera(
            config=self.config,
            validate_on_connect=False,
            overwrite_settings=not self.is_locked,
        )
        camera.connect()
        try:
            with Bus.join(self.session_id) if self.session_id else nullcontext() as self._bus:
                yield camera
        finally:
            logger.info("Camera run loop stopped. Disconnecting")
            if camera.is_connected:
                camera.disconnect()

    async def run_loop(self, context: SharedCamera) -> None:
        try:
            await self._capture(context)
        except Exception as error:
            if self._bus is not None and self.component is not None:
                publish_status(self._bus, self.component, "failed", str(error))
            raise

    async def _capture(self, context: SharedCamera) -> None:
        rate = Rate(self.config.payload.fps, name=self.name)
        size = (self._image.width, self._image.height)
        while not self.should_stop():
            frame = context.read_latest()
            data = frame.data if frame.data.shape[:2] == self._image.shape[:2] else cv2.resize(frame.data, size)
            self.store.write_observation(self.key, data, timestamp=frame.timestamp)
            await rate.sleep()
