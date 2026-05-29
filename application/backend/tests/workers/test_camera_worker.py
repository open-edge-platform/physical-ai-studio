import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from physicalai.capture.errors import CaptureError, CaptureTimeoutError

from workers.camera_worker import CameraWorker
from workers.camera_worker_registry import CameraWorkerRegistry


def _make_config():
    from schemas.project_camera import CameraAdapter

    return CameraAdapter.validate_python(
        {
            "driver": "usb_camera",
            "name": "test_cam",
            "fingerprint": "/dev/video0",
            "hardware_name": None,
            "payload": {"width": 640, "height": 480, "fps": 30},
        }
    )


def _make_frame(height: int = 480, width: int = 640):
    mock_frame = MagicMock()
    mock_frame.data = np.zeros((height, width, 3), dtype=np.uint8)
    return mock_frame


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestCameraWorker:
    def test_build_shared_camera_called(self):
        config = _make_config()
        transport = MagicMock()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_build.return_value = MagicMock()
            worker = CameraWorker(config, transport)
            mock_build.assert_called_once_with(config=config, validate_on_connect=False, overwrite_settings=True)
            assert worker.cam is not None

    def test_shutdown_disconnects_camera(self, event_loop):
        config = _make_config()
        transport = MagicMock()
        transport.close = AsyncMock()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_cam = MagicMock()
            mock_cam.is_connected = True
            mock_cam.disconnect.side_effect = lambda: setattr(mock_cam, "is_connected", False)
            mock_build.return_value = mock_cam
            worker = CameraWorker(config, transport)
            event_loop.run_until_complete(worker.shutdown())
            mock_cam.disconnect.assert_called_once()
            assert not worker.cam.is_connected

    def test_shutdown_is_idempotent(self, event_loop):
        config = _make_config()
        transport = MagicMock()
        transport.close = AsyncMock()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_cam = MagicMock()
            mock_cam.is_connected = True
            mock_cam.disconnect.side_effect = lambda: setattr(mock_cam, "is_connected", False)
            mock_build.return_value = mock_cam
            worker = CameraWorker(config, transport)

            async def _double_shutdown():
                await worker.shutdown()
                await worker.shutdown()

            event_loop.run_until_complete(_double_shutdown())
            mock_cam.disconnect.assert_called_once()

    def test_capture_loop_handles_timeout(self, event_loop):
        config = _make_config()
        transport = MagicMock()
        transport.close = AsyncMock()
        transport.send_bytes = AsyncMock()

        with (
            patch("workers.camera_worker.build_shared_camera") as mock_build,
            patch("workers.camera_worker.encode_jpeg_rgb", return_value=b"\xff\xd8fake-jpeg"),
        ):
            mock_cam = MagicMock()
            call_count = 0

            async def async_read_side_effect(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise CaptureTimeoutError("timeout")
                if call_count == 3:
                    return _make_frame()
                # After first successful frame, block until cancelled
                await asyncio.sleep(999)

            mock_cam.async_read = async_read_side_effect
            mock_build.return_value = mock_cam

            worker = CameraWorker(config, transport)

            async def _run():
                task = asyncio.create_task(worker._capture_loop())
                # Wait until at least one frame was sent
                ready = asyncio.Event()

                original_send = transport.send_bytes.side_effect

                async def _notify(*args, **kw):
                    if original_send:
                        await original_send(*args, **kw)
                    ready.set()

                transport.send_bytes.side_effect = _notify
                await asyncio.wait_for(ready.wait(), timeout=5)
                worker._stop_requested = True
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            event_loop.run_until_complete(_run())
            assert transport.send_bytes.call_count >= 1

    def test_capture_loop_breaks_on_capture_error(self, event_loop):
        config = _make_config()
        transport = MagicMock()
        transport.close = AsyncMock()
        transport.send_bytes = AsyncMock()

        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_cam = MagicMock()
            mock_cam.async_read = AsyncMock(side_effect=CaptureError("hardware failure"))
            mock_build.return_value = mock_cam

            worker = CameraWorker(config, transport)
            event_loop.run_until_complete(asyncio.wait_for(worker._capture_loop(), timeout=2.0))
            transport.send_bytes.assert_not_called()

    def test_run_sends_error_on_connect_failure(self, event_loop):
        config = _make_config()
        transport = MagicMock()
        transport.connect = AsyncMock()
        transport.close = AsyncMock()
        transport.send_json = AsyncMock()

        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_cam = MagicMock()
            mock_cam.connect.side_effect = CaptureError("device is busy")
            mock_build.return_value = mock_cam

            worker = CameraWorker(config, transport)
            event_loop.run_until_complete(worker.run())

            transport.send_json.assert_called()
            sent_data = transport.send_json.call_args[0][0]
            assert "device is busy" in str(sent_data)


class TestCameraWorkerRegistry:
    def test_evicts_stale_worker_with_same_fingerprint(self, event_loop):
        registry = CameraWorkerRegistry()

        config1 = _make_config()
        config2 = _make_config()
        transport = MagicMock()

        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_build.return_value = MagicMock()
            worker1 = CameraWorker(config1, transport)
            worker2 = CameraWorker(config2, transport)

        async def _register_both():
            id1 = uuid4()
            await registry.create_and_register(id1, worker1)
            await registry.create_and_register(uuid4(), worker2)
            # Old worker was removed from registry (not shut down — that's
            # left to the old websocket handler's finally block).
            assert await registry.get(id1) is None

        event_loop.run_until_complete(_register_both())
