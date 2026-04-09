# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""V4L2 ioctl constants, ctypes ABI structs, and helpers.

This module defines Linux V4L2 kernel ABI bindings used by camera backends.
All ioctl request numbers are computed from `_IOC` macros rather than
hardcoded literals.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import struct
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ctypes import Structure

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


class v4l2_capability(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("driver", ctypes.c_char * 16),
            ("card", ctypes.c_char * 32),
            ("bus_info", ctypes.c_char * 32),
            ("version", ctypes.c_uint32),
            ("capabilities", ctypes.c_uint32),
            ("device_caps", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32 * 3),
        ],
    )


class v4l2_fmtdesc(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("index", ctypes.c_uint32),
            ("type", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("description", ctypes.c_char * 32),
            ("pixelformat", ctypes.c_uint32),
            ("mbus_code", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32 * 3),
        ],
    )


class v4l2_pix_format(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("width", ctypes.c_uint32),
            ("height", ctypes.c_uint32),
            ("pixelformat", ctypes.c_uint32),
            ("field", ctypes.c_uint32),
            ("bytesperline", ctypes.c_uint32),
            ("sizeimage", ctypes.c_uint32),
            ("colorspace", ctypes.c_uint32),
            ("priv", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("ycbcr_enc", ctypes.c_uint32),
            ("quantization", ctypes.c_uint32),
            ("xfer_func", ctypes.c_uint32),
        ],
    )


class v4l2_rect(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("left", ctypes.c_int32),
            ("top", ctypes.c_int32),
            ("width", ctypes.c_uint32),
            ("height", ctypes.c_uint32),
        ],
    )


class v4l2_clip(ctypes.Structure):  # noqa: N801
    pass


v4l2_clip._fields_ = cast(
    "Any",
    [
        ("c", v4l2_rect),
        ("next", ctypes.POINTER(v4l2_clip)),
    ],
)


class v4l2_window(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("w", v4l2_rect),
            ("field", ctypes.c_uint32),
            ("chromakey", ctypes.c_uint32),
            ("clips", ctypes.POINTER(v4l2_clip)),
            ("clipcount", ctypes.c_uint32),
            ("bitmap", ctypes.c_void_p),
            ("global_alpha", ctypes.c_uint8),
        ],
    )


class _v4l2_format_union(ctypes.Union):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("pix", v4l2_pix_format),
            ("win", v4l2_window),
            ("raw_data", ctypes.c_uint8 * 200),
        ],
    )


class v4l2_format(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("type", ctypes.c_uint32),
            ("fmt", _v4l2_format_union),
        ],
    )


class v4l2_requestbuffers(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("count", ctypes.c_uint32),
            ("type", ctypes.c_uint32),
            ("memory", ctypes.c_uint32),
            ("capabilities", ctypes.c_uint32),
            ("flags", ctypes.c_uint8),
            ("reserved", ctypes.c_uint8 * 3),
        ],
    )


class v4l2_timeval(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("tv_sec", ctypes.c_long),
            ("tv_usec", ctypes.c_long),
        ],
    )


class v4l2_timecode(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("type", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("frames", ctypes.c_uint8),
            ("seconds", ctypes.c_uint8),
            ("minutes", ctypes.c_uint8),
            ("hours", ctypes.c_uint8),
            ("userbits", ctypes.c_uint8 * 4),
        ],
    )


class v4l2_buffer(ctypes.Structure):  # noqa: N801
    class _m(ctypes.Union):  # noqa: N801
        _fields_ = cast(
            "Any",
            [
                ("offset", ctypes.c_uint32),
                ("userptr", ctypes.c_ulong),
                ("planes", ctypes.c_void_p),
                ("fd", ctypes.c_int32),
            ],
        )

    _fields_ = cast(
        "Any",
        [
            ("index", ctypes.c_uint32),
            ("type", ctypes.c_uint32),
            ("bytesused", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("field", ctypes.c_uint32),
            ("timestamp", v4l2_timeval),
            ("timecode", v4l2_timecode),
            ("sequence", ctypes.c_uint32),
            ("memory", ctypes.c_uint32),
            ("m", _m),
            ("length", ctypes.c_uint32),
            ("reserved2", ctypes.c_uint32),
            ("request_fd", ctypes.c_int32),
        ],
    )


class v4l2_fract(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("numerator", ctypes.c_uint32),
            ("denominator", ctypes.c_uint32),
        ],
    )


class v4l2_captureparm(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("capability", ctypes.c_uint32),
            ("capturemode", ctypes.c_uint32),
            ("timeperframe", v4l2_fract),
            ("extendedmode", ctypes.c_uint32),
            ("readbuffers", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32 * 4),
        ],
    )


class _v4l2_streamparm_union(ctypes.Union):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("capture", v4l2_captureparm),
            ("raw_data", ctypes.c_uint8 * 200),
        ],
    )


class v4l2_streamparm(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("type", ctypes.c_uint32),
            ("parm", _v4l2_streamparm_union),
        ],
    )


def _IOC(dir_: int, type_char: str, nr: int, size: int) -> int:  # noqa: N802
    return (
        (dir_ << _IOC_DIRSHIFT) | (ord(type_char) << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)
    )


def _IOR(type_char: str, nr: int, data_type: type[ctypes.Structure]) -> int:  # noqa: N802
    return _IOC(_IOC_READ, type_char, nr, ctypes.sizeof(data_type))


def _IOW(  # noqa: N802
    type_char: str,
    nr: int,
    data_type: type[Structure | ctypes.c_int],
) -> int:
    return _IOC(_IOC_WRITE, type_char, nr, ctypes.sizeof(data_type))


def _IOWR(  # noqa: N802
    type_char: str,
    nr: int,
    data_type: type[Structure],
) -> int:
    return _IOC(_IOC_READ | _IOC_WRITE, type_char, nr, ctypes.sizeof(data_type))


V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 1
V4L2_PIX_FMT_MJPEG = struct.unpack("<I", b"MJPG")[0]
V4L2_PIX_FMT_YUYV = struct.unpack("<I", b"YUYV")[0]
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_META_CAPTURE = 0x00800000
V4L2_CAP_STREAMING = 0x04000000
V4L2_CAP_DEVICE_CAPS = 0x80000000

V4L2_CTRL_FLAG_NEXT_CTRL = 0x80000000
V4L2_CTRL_FLAG_DISABLED = 0x0001
V4L2_CTRL_FLAG_GRABBED = 0x0002
V4L2_CTRL_FLAG_READ_ONLY = 0x0004
V4L2_CTRL_FLAG_UPDATE = 0x0008
V4L2_CTRL_FLAG_INACTIVE = 0x0010
V4L2_CTRL_FLAG_WRITE_ONLY = 0x0020
V4L2_CTRL_FLAG_VOLATILE = 0x0040
V4L2_CTRL_FLAG_SLIDER = 0x1000

V4L2_CTRL_TYPE_INTEGER = 1
V4L2_CTRL_TYPE_BOOLEAN = 2
V4L2_CTRL_TYPE_MENU = 3
V4L2_CTRL_TYPE_BUTTON = 4
V4L2_CTRL_TYPE_INTEGER64 = 5
V4L2_CTRL_TYPE_CTRL_CLASS = 6
V4L2_CTRL_TYPE_STRING = 7
V4L2_CTRL_TYPE_BITMASK = 8
V4L2_CTRL_TYPE_COMPOUND = 0x0100

CTRL_TYPE_NAMES: dict[int, str] = {
    V4L2_CTRL_TYPE_INTEGER: "integer",
    V4L2_CTRL_TYPE_BOOLEAN: "boolean",
    V4L2_CTRL_TYPE_MENU: "menu",
    V4L2_CTRL_TYPE_BUTTON: "button",
    V4L2_CTRL_TYPE_INTEGER64: "integer64",
    V4L2_CTRL_TYPE_CTRL_CLASS: "ctrl_class",
    V4L2_CTRL_TYPE_STRING: "string",
    V4L2_CTRL_TYPE_BITMASK: "bitmask",
    V4L2_CTRL_TYPE_COMPOUND: "compound",
}

CTRL_FLAG_BITS: list[tuple[int, str]] = [
    (V4L2_CTRL_FLAG_DISABLED, "disabled"),
    (V4L2_CTRL_FLAG_GRABBED, "grabbed"),
    (V4L2_CTRL_FLAG_READ_ONLY, "read_only"),
    (V4L2_CTRL_FLAG_UPDATE, "update"),
    (V4L2_CTRL_FLAG_INACTIVE, "inactive"),
    (V4L2_CTRL_FLAG_WRITE_ONLY, "write_only"),
    (V4L2_CTRL_FLAG_VOLATILE, "volatile"),
    (V4L2_CTRL_FLAG_SLIDER, "slider"),
]


def decode_ctrl_flags(flags: int) -> list[str]:
    """Decode a V4L2 control flags bitmask into human-readable names.

    Args:
        flags: V4L2 control flag bitmask.

    Returns:
        The decoded flag names present in ``flags``.
    """
    return [name for bit, name in CTRL_FLAG_BITS if flags & bit]


class v4l2_control(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("id", ctypes.c_uint32),
            ("value", ctypes.c_int32),
        ],
    )


class v4l2_queryctrl(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("id", ctypes.c_uint32),
            ("type", ctypes.c_uint32),
            ("name", ctypes.c_char * 32),
            ("minimum", ctypes.c_int32),
            ("maximum", ctypes.c_int32),
            ("step", ctypes.c_int32),
            ("default_value", ctypes.c_int32),
            ("flags", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32 * 2),
        ],
    )


class _v4l2_querymenu_union(ctypes.Union):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("name", ctypes.c_char * 32),
            ("value", ctypes.c_int64),
        ],
    )


class v4l2_querymenu(ctypes.Structure):  # noqa: N801
    _fields_ = cast(
        "Any",
        [
            ("id", ctypes.c_uint32),
            ("index", ctypes.c_uint32),
            ("u", _v4l2_querymenu_union),
            ("reserved", ctypes.c_uint32),
        ],
    )


VIDIOC_QUERYCAP = _IOR("V", 0, v4l2_capability)
VIDIOC_ENUM_FMT = _IOWR("V", 2, v4l2_fmtdesc)
VIDIOC_G_FMT = _IOWR("V", 4, v4l2_format)
VIDIOC_S_FMT = _IOWR("V", 5, v4l2_format)
VIDIOC_REQBUFS = _IOWR("V", 8, v4l2_requestbuffers)
VIDIOC_QUERYBUF = _IOWR("V", 9, v4l2_buffer)
VIDIOC_QBUF = _IOWR("V", 15, v4l2_buffer)
VIDIOC_DQBUF = _IOWR("V", 17, v4l2_buffer)
VIDIOC_STREAMON = _IOW("V", 18, ctypes.c_int)
VIDIOC_STREAMOFF = _IOW("V", 19, ctypes.c_int)
VIDIOC_S_PARM = _IOWR("V", 22, v4l2_streamparm)
VIDIOC_G_CTRL = _IOWR("V", 27, v4l2_control)
VIDIOC_S_CTRL = _IOWR("V", 28, v4l2_control)
VIDIOC_QUERYCTRL = _IOWR("V", 36, v4l2_queryctrl)
VIDIOC_QUERYMENU = _IOWR("V", 37, v4l2_querymenu)


def xioctl(fd: int, request: int, arg: Structure) -> int:
    """Run ioctl with automatic retry on interrupted syscalls.

    Args:
        fd: File descriptor of the V4L2 device.
        request: Encoded ioctl request.
        arg: Mutable ctypes struct argument passed to ioctl.

    Returns:
        Integer return code from ``fcntl.ioctl``.

    Raises:
        OSError: If ioctl fails with any errno other than EINTR.
    """
    while True:
        try:
            return fcntl.ioctl(fd, request, arg, True)  # noqa: FBT003
        except OSError as exc:
            if exc.errno != errno.EINTR:
                raise


__all__ = [
    "CTRL_FLAG_BITS",
    "CTRL_TYPE_NAMES",
    "V4L2_BUF_TYPE_VIDEO_CAPTURE",
    "V4L2_CAP_DEVICE_CAPS",
    "V4L2_CAP_META_CAPTURE",
    "V4L2_CAP_STREAMING",
    "V4L2_CAP_VIDEO_CAPTURE",
    "V4L2_CTRL_FLAG_DISABLED",
    "V4L2_CTRL_FLAG_GRABBED",
    "V4L2_CTRL_FLAG_INACTIVE",
    "V4L2_CTRL_FLAG_NEXT_CTRL",
    "V4L2_CTRL_FLAG_READ_ONLY",
    "V4L2_CTRL_FLAG_SLIDER",
    "V4L2_CTRL_FLAG_UPDATE",
    "V4L2_CTRL_FLAG_VOLATILE",
    "V4L2_CTRL_FLAG_WRITE_ONLY",
    "V4L2_CTRL_TYPE_BITMASK",
    "V4L2_CTRL_TYPE_BOOLEAN",
    "V4L2_CTRL_TYPE_BUTTON",
    "V4L2_CTRL_TYPE_COMPOUND",
    "V4L2_CTRL_TYPE_CTRL_CLASS",
    "V4L2_CTRL_TYPE_INTEGER",
    "V4L2_CTRL_TYPE_INTEGER64",
    "V4L2_CTRL_TYPE_MENU",
    "V4L2_CTRL_TYPE_STRING",
    "V4L2_FIELD_NONE",
    "V4L2_MEMORY_MMAP",
    "V4L2_PIX_FMT_MJPEG",
    "V4L2_PIX_FMT_YUYV",
    "VIDIOC_DQBUF",
    "VIDIOC_ENUM_FMT",
    "VIDIOC_G_CTRL",
    "VIDIOC_G_FMT",
    "VIDIOC_QBUF",
    "VIDIOC_QUERYBUF",
    "VIDIOC_QUERYCAP",
    "VIDIOC_QUERYCTRL",
    "VIDIOC_QUERYMENU",
    "VIDIOC_REQBUFS",
    "VIDIOC_STREAMOFF",
    "VIDIOC_STREAMON",
    "VIDIOC_S_CTRL",
    "VIDIOC_S_FMT",
    "VIDIOC_S_PARM",
    "decode_ctrl_flags",
    "v4l2_buffer",
    "v4l2_capability",
    "v4l2_captureparm",
    "v4l2_clip",
    "v4l2_control",
    "v4l2_fmtdesc",
    "v4l2_format",
    "v4l2_pix_format",
    "v4l2_queryctrl",
    "v4l2_querymenu",
    "v4l2_rect",
    "v4l2_requestbuffers",
    "v4l2_streamparm",
    "v4l2_timecode",
    "v4l2_timeval",
    "v4l2_window",
    "xioctl",
]
