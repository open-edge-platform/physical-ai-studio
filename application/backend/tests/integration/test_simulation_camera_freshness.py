# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check actual MJPEG image age, not the timestamp assigned when an old frame is read."""

import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from physicalai.config import Config

from schemas.project_camera import IPCamera, IPCameraPayload
from services.runtime_export import _shared_camera_config

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_camera_latest_frame_does_not_backlog() -> None:
    """Inference pauses must not leave the camera reading historical MJPEG frames."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("iceoryx2")
    fps = 30
    started = time.monotonic()

    def frame_id() -> int:
        return int((time.monotonic() - started) * fps)

    def encode(counter: int) -> bytes:
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        for bit in range(16):
            image[:, bit * 8 : (bit + 1) * 8] = 255 if counter & (1 << bit) else 0
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        assert ok
        return encoded.tobytes()

    def decode(image: np.ndarray) -> int:
        return sum(1 << bit for bit in range(16) if image[32, bit * 8 + 4].mean() > 127)

    for counter in (0, 1, 0x55AA, 65535):
        image = cv2.imdecode(np.frombuffer(encode(counter), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decode(image) == counter

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    payload = encode(frame_id())
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(payload)).encode()
                        + b"\r\n\r\n"
                        + payload
                        + b"\r\n"
                    )
                    self.wfile.flush()
                    time.sleep(1 / fps)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/stream"
    camera = None
    try:
        profile = IPCamera(
            name="freshness-probe",
            driver="ipcam",
            fingerprint={"url": url},
            hardware_name="MJPEG freshness probe",
            payload=IPCameraPayload(url=url, fps=fps),
        )
        camera = Config.from_dict(_shared_camera_config(profile)).instantiate()
        camera.connect()
        camera.read_latest()
        time.sleep(1.5)
        frame = camera.read_latest()
        actual_age_seconds = (frame_id() - decode(frame.data)) / fps
        assert 0 <= actual_age_seconds <= 0.25, f"Camera returned a {actual_age_seconds:.2f}s-old image"
    finally:
        try:
            if camera is not None:
                # This test owns its unique publisher; do not wait for idle cleanup.
                publisher = getattr(camera, "_publisher", None)
                try:
                    camera.disconnect()
                finally:
                    if publisher is not None:
                        publisher.stop()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            assert not thread.is_alive()


@pytest.mark.parametrize("end_one_feed", [False, True])
def test_video_requires_both_mjpeg_feeds_throughout_rollout(tmp_path, end_one_feed) -> None:
    """An ended view must fail recording, not become a frozen half of a valid MP4."""
    from tests.integration.test_simulation_e2e import _record_video

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the real two-stream recording regression")
    cv2 = pytest.importorskip("cv2")
    ok, encoded = cv2.imencode(".jpg", np.full((48, 64, 3), 128, dtype=np.uint8))
    assert ok
    payload = encoded.tobytes()
    packet = (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(payload)).encode()
        + b"\r\n\r\n"
        + payload
        + b"\r\n"
    )
    end_overview = threading.Event()
    shutdown = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not shutdown.is_set():
                    if "overview" in self.path and end_overview.is_set():
                        return
                    self.wfile.write(packet)
                    self.wfile.flush()
                    shutdown.wait(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def record() -> None:
        with _record_video(tmp_path, url, ["overview", "wrist"], {}, 30, "act", "conveyor_sort"):
            time.sleep(0.5)
            if end_one_feed:
                end_overview.set()
            time.sleep(1.5)

    try:
        if end_one_feed:
            with pytest.raises(RuntimeError, match=r"ffmpeg failed|does not cover"):
                record()
            assert end_overview.is_set(), "Feed must end during the rollout, not during startup"
        else:
            record()
        assert (tmp_path / "inference-act-conveyor_sort.mp4").is_file()
    finally:
        shutdown.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
