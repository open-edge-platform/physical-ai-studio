from frame_source.video_capture_base import VideoCaptureBase
import time

import cv2
import numpy as np
from frame_source import FrameSourceFactory
from lerobot.cameras import ColorMode, Cv2Rotation

from schemas.project_camera import Camera


class FrameSourceCameraBridge:
    def __init__(self, config: Camera):
        self.color_mode: ColorMode = ColorMode.RGB
        self.rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
        self.warmup_s: int = 1
        self.camera = self.create_frames_source_from_camera(config)

    @staticmethod
    def create_frames_source_from_camera(camera: Camera) -> VideoCaptureBase:
        """Very FrameSource factory call from camera schema object."""
        return FrameSourceFactory.create(
            "webcam" if camera.driver == "usb_camera" else camera.driver,
            camera.fingerprint,
            **camera.payload.model_dump(),
        )

    @property
    def is_connected(self) -> bool:
        """Check if the camera is currently connected.

        Returns:
            bool: True if the camera is connected and ready to capture frames,
                  False otherwise.
        """
        return self.camera.is_connected

    def connect(self, warmup: bool = True) -> None:
        """Establish connection to the camera.

        Args:
            warmup: If True (default), captures a warmup frame before returning. Useful
                   for cameras that require time to adjust capture settings.
                   If False, skips the warmup frame.
        """
        self.camera.connect()
        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

    def start_async(self) -> None:
        """Start async task of reading camera frames."""
        self.camera.start_async()

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """Capture and return a single frame from the camera.

        Args:
            color_mode: Desired color mode for the output frame. If None,
                        uses the camera's default color mode.

        Returns:
            np.ndarray: Captured frame as a numpy array.
        """
        retrycount = 3
        while retrycount > 0:
            success, frame = self.camera.read()
            if success:
                break
            print("failed reading... retrying")
            retrycount -= 1

        if not success or frame is None:
            raise RuntimeError(f"{self} read failed (status={success}).")
        frame = np.swapaxes(frame, 0, 1)  # FrameSource outputs WHC and we expect HWC
        return self._postprocess_image(frame, color_mode)

    def _postprocess_image(self, image: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        Applies color conversion, dimension validation, and rotation to a raw frame.

        Args:
            image (np.ndarray): The raw image frame (expected BGR format from OpenCV).
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR). If None,
                                             uses the instance's default `self.color_mode`.

        Returns:
            np.ndarray: The processed image frame.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match the configured
                          `width` and `height`.
        """
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        processed_image = image
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def async_read(self, _timeout_ms: float = 0.0) -> np.ndarray:
        """Asynchronously capture and return a single frame from the camera.

        Args:
            timeout_ms: Maximum time to wait for a frame in milliseconds.
                        Defaults to implementation-specific timeout.

        Returns:
            np.ndarray: Captured frame as a numpy array.
        """
        return self.read()

    def disconnect(self) -> None:
        """Disconnect from the camera and release resources."""
        self.camera.disconnect()
