"""JPEG encoding utilities using OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


def encode_jpeg_rgb(image: np.ndarray, quality: int = 80) -> bytes:
    """Encode an RGB image to JPEG bytes.

    Args:
        image: HxWx3 numpy array in RGB channel order.
        quality: JPEG quality 1-100 (default 80).

    Returns:
        JPEG-encoded bytes.
    """
    # Convert RGB → BGR (OpenCV convention) and ensure contiguous memory layout.
    bgr = np.ascontiguousarray(image[:, :, ::-1])
    success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        msg = "JPEG encoding failed"
        raise RuntimeError(msg)
    return buf.tobytes()
