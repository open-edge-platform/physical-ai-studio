import asyncio
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from runtime_session.store import FeatureSpec, Store
from workers.camera_worker import CameraWorker, camera_image_spec


def _make_config(width=640, height=480, fps=30):
    from schemas.project_camera import CameraAdapter

    return CameraAdapter.validate_python(
        {
            "driver": "usb_camera",
            "name": "test_cam",
            "fingerprint": {"path": "/dev/video0"},
            "hardware_name": None,
            "payload": {"width": width, "height": height, "fps": fps},
        }
    )


def _make_frame(height: int = 480, width: int = 640, value: int = 0, timestamp: float = 1.0):
    mock_frame = MagicMock()
    mock_frame.data = np.full((height, width, 3), value, dtype=np.uint8)
    mock_frame.timestamp = timestamp
    return mock_frame


@pytest.fixture
def store():
    store = Store.create(FeatureSpec((), {"cam": camera_image_spec(_make_config())}), uuid.uuid4().hex)
    yield store
    store.unlink()


def _enter_and_leave(worker: CameraWorker, check=lambda camera: None) -> None:
    async def run() -> None:
        async with worker.lifecycle() as camera:
            check(camera)

    asyncio.run(run())


class TestCameraWorker:
    def test_lifecycle_builds_and_connects_the_camera(self, store):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_cam = mock_build.return_value
            mock_cam.is_connected = True
            worker = CameraWorker(config, store, "cam")
            mock_build.assert_not_called()

            def check(camera) -> None:
                assert camera is mock_cam
                mock_cam.connect.assert_called_once()
                mock_cam.disconnect.assert_not_called()

            _enter_and_leave(worker, check)
            mock_build.assert_called_once_with(config=config, validate_on_connect=False, overwrite_settings=True)
            mock_cam.disconnect.assert_called_once()

    def test_a_locked_camera_keeps_its_settings(self, store):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            _enter_and_leave(CameraWorker(config, store, "cam", is_locked=True))
            mock_build.assert_called_once_with(config=config, validate_on_connect=False, overwrite_settings=False)

    def test_lifecycle_skips_disconnect_if_not_connected(self, store):
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_build.return_value.is_connected = False
            _enter_and_leave(CameraWorker(_make_config(), store, "cam"))
            mock_build.return_value.disconnect.assert_not_called()

    def test_run_loop_writes_frames_into_the_store_until_stopped(self, store):
        worker = CameraWorker(_make_config(), store, "cam")
        frames = [_make_frame(value=1, timestamp=1.0), _make_frame(240, 320, value=2, timestamp=2.0)]

        def read_latest():
            if len(frames) == 1:
                worker.request_stop()
            return frames.pop(0)

        camera = MagicMock()
        camera.read_latest.side_effect = read_latest
        asyncio.run(worker.run_loop(camera))

        sample = store.read_observation("cam")
        assert sample.timestamp == 2.0
        # A frame at another resolution is resized to the slot.
        assert sample.value.shape == (480, 640, 3)
        assert np.all(sample.value == 2)
