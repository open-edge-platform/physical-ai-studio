import asyncio
import time
import uuid
from typing import List, Any

import cv2
from frame_source import RealsenseCapture, VideoCaptureBase, FrameSourceFactory

from utils.async_camera_capture import AsyncCameraCapture


async def test_multiple_cameras(cameras: List[AsyncCameraCapture], threaded: bool = True):
    """Test connecting to multiple different cameras types and viewing them live concurrently."""

    capture_instances = []
    fps_counters = {}  # name -> FPSCounter
    grid_cols = 3
    grid_rows = 2
    win_w, win_h = 640, 480
    for idx, cam_cfg in enumerate(cameras):
        name = "blabla" + str(uuid.uuid4())
        cv2.namedWindow(f"{name}", cv2.WINDOW_NORMAL)
        # Set window size and position for grid
        cv2.resizeWindow(f"{name}", win_w, win_h)
        col = idx % grid_cols
        row = idx // grid_cols
        x = col * win_w
        y = row * win_h
        cv2.moveWindow(f"{name}", x, y + (25 * row))  # Add some vertical spacing

        await cam_cfg.start()
        capture_instances.append((name, cam_cfg))
        fps_counters[name] = FPSCounter()

    try:
        while True:
            for name, camera in capture_instances:
                frame, t_perf, ok, error, seq = await camera.get_latest()
                if ok and frame is not None:
                    fps = fps_counters[name].update()
                    frame = draw_fps(frame, fps, label=f"{name} | ")
                    cv2.imshow(f"{name}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Let other asyncio tasks run (your capture loops!)
            await asyncio.sleep(0)

    finally:
        for name, camera in capture_instances:
            await camera.stop()
            print(f"Disconnected from {name}")


if __name__ == '__main__':
    rs1 = RealsenseCapture("218622276042", width=640, height=480, fps=90)
    r1 = AsyncCameraCapture(rs1, 30.0)
    rs2 = RealsenseCapture("230422272459", width=640, height=480, fps=90)
    r2 = AsyncCameraCapture(rs2, 30.0)
    asyncio.run(test_multiple_cameras([r1, r2]))
