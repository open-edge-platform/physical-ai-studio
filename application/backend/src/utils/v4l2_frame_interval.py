"""Repair V4L2 capture devices that report an invalid ``1/0`` frame interval.

Some UVC webcams (for example icSpring cameras) come up after being plugged in
with a current ``timeperframe`` of ``0/1`` (shown by ``v4l2-ctl`` as ``1/0`` fps)
until a frame rate is explicitly set. The ``nokhwa`` backend used by
``physicalai.capture`` rejects that value when opening the device, so the camera
is silently dropped from discovery and cannot be opened.

Setting any frame rate with ``VIDIOC_S_PARM`` puts the device into a valid state,
which is what ``v4l2-ctl -p 30`` does by hand. This module does the same for every
capture node that reports a zero interval. It is a no-op on non-Linux platforms.
"""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path

from loguru import logger

DEFAULT_FPS = 30

_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
_V4L2_CAP_TIMEPERFRAME = 0x1000


class _V4L2Fract(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


class _V4L2CaptureParm(ctypes.Structure):
    _fields_ = [
        ("capability", ctypes.c_uint32),
        ("capturemode", ctypes.c_uint32),
        ("timeperframe", _V4L2Fract),
        ("extendedmode", ctypes.c_uint32),
        ("readbuffers", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class _V4L2StreamParmUnion(ctypes.Union):
    _fields_ = [("capture", _V4L2CaptureParm), ("raw_data", ctypes.c_uint8 * 200)]


class _V4L2StreamParm(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("parm", _V4L2StreamParmUnion)]


def _iowr(type_char: str, nr: int, size: int) -> int:
    # Linux _IOWR(): dir (read|write = 3) << 30 | size << 16 | type << 8 | nr
    return (3 << 30) | (size << 16) | (ord(type_char) << 8) | nr


VIDIOC_G_PARM = _iowr("V", 21, ctypes.sizeof(_V4L2StreamParm))
VIDIOC_S_PARM = _iowr("V", 22, ctypes.sizeof(_V4L2StreamParm))


def _has_zero_interval(fract: _V4L2Fract) -> bool:
    return fract.numerator == 0 or fract.denominator == 0


def fix_device_frame_interval(device: str | Path, fps: int = DEFAULT_FPS) -> bool:
    """Set a valid frame rate on *device* if it reports a zero frame interval.

    Args:
        device: Path to a V4L2 device node, e.g. ``/dev/video0``.
        fps: Frame rate to request. The driver picks the nearest supported rate.

    Returns:
        ``True`` if the device had a zero interval and was repaired, ``False``
        otherwise (interval already valid, not a capture node, not accessible, or
        not running on Linux).
    """
    if not sys.platform.startswith("linux"):
        return False

    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return False

    try:
        return _repair_open_device(fd, device, fps)
    finally:
        os.close(fd)


def _repair_open_device(fd: int, device: str | Path, fps: int) -> bool:
    # fcntl is POSIX-only; import it here so this module stays importable on Windows.
    import fcntl

    parm = _V4L2StreamParm(type=_V4L2_BUF_TYPE_VIDEO_CAPTURE)
    try:
        fcntl.ioctl(fd, VIDIOC_G_PARM, parm)
    except OSError:
        # Metadata/output nodes and some ISP nodes don't support G_PARM.
        return False

    capture = parm.parm.capture
    if not _has_zero_interval(capture.timeperframe):
        return False
    if not capture.capability & _V4L2_CAP_TIMEPERFRAME:
        logger.debug("{} reports a zero frame interval but cannot set one", device)
        return False

    capture.timeperframe.numerator = 1
    capture.timeperframe.denominator = fps
    try:
        fcntl.ioctl(fd, VIDIOC_S_PARM, parm)
    except OSError as exc:
        if exc.errno != errno.EBUSY:
            logger.warning("Failed to set frame rate on {}: {}", device, exc)
        return False

    fixed = not _has_zero_interval(parm.parm.capture.timeperframe)
    if fixed:
        logger.info(
            "Repaired zero frame interval on {} (now {}/{} s)",
            device,
            parm.parm.capture.timeperframe.numerator,
            parm.parm.capture.timeperframe.denominator,
        )
    return fixed


def fix_zero_frame_intervals(dev_dir: Path = Path("/dev"), fps: int = DEFAULT_FPS) -> list[str]:
    """Repair every ``/dev/video*`` capture node that reports a zero frame interval.

    Returns:
        Paths of the devices that were repaired.
    """
    if not sys.platform.startswith("linux"):
        return []
    return [str(dev) for dev in sorted(dev_dir.glob("video*")) if fix_device_frame_interval(dev, fps)]
