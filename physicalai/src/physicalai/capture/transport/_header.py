# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Packed transport header and frame encode/decode helpers."""

from __future__ import annotations

import ctypes

import numpy as np

from physicalai.capture.camera import ColorMode
from physicalai.capture.errors import CaptureError
from physicalai.capture.frame import Frame


class FrameHeader(ctypes.Structure):
    """Binary protocol header prepended to each transport payload."""

    _pack_ = 1
    _fields_ = [
        ("version", ctypes.c_uint8),
        ("channels", ctypes.c_uint8),
        ("dtype", ctypes.c_uint8),
        ("color_mode", ctypes.c_uint8),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("sequence", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
        ("depth_offset", ctypes.c_uint32),
        ("depth_width", ctypes.c_uint32),
        ("depth_height", ctypes.c_uint32),
    ]


HEADER_SIZE: int = ctypes.sizeof(FrameHeader)
PROTOCOL_VERSION: int = 1

_NDIM_2D = 2
_NDIM_3D = 3

DTYPE_MAP = {
    0: np.uint8,
    1: np.uint16,
}

COLOR_MODE_MAP = {
    ColorMode.RGB: 0,
    ColorMode.BGR: 1,
    ColorMode.GRAY: 2,
    0: ColorMode.RGB,
    1: ColorMode.BGR,
    2: ColorMode.GRAY,
}


def encode_frame(
    frame: Frame,
    color_mode: ColorMode,
    depth_frame: Frame | None = None,
) -> tuple[FrameHeader, bytes]:
    """Encode colour (and optional depth) frames into transport payload bytes.

    Args:
        frame: Colour frame to encode.
        color_mode: Colour mode describing channel ordering in ``frame``.
        depth_frame: Optional depth frame appended to payload.

    Returns:
        Tuple of the populated header and payload bytes (without header bytes).

    Raises:
        CaptureError: If frame dtypes/shapes are unsupported.
    """
    if frame.data.ndim == _NDIM_2D:
        height, width = frame.data.shape
        channels = 1
    elif frame.data.ndim == _NDIM_3D:
        height, width, channels = frame.data.shape
    else:
        msg = f"Unsupported frame rank {frame.data.ndim}; expected 2D or 3D image data"
        raise CaptureError(msg)

    if frame.data.dtype == np.uint8:
        dtype_code = 0
    elif frame.data.dtype == np.uint16:
        dtype_code = 1
    else:
        msg = f"Unsupported frame dtype {frame.data.dtype}; expected uint8 or uint16"
        raise CaptureError(msg)

    color_code = COLOR_MODE_MAP[color_mode]
    header = FrameHeader(
        version=PROTOCOL_VERSION,
        channels=channels,
        dtype=dtype_code,
        color_mode=color_code,
        width=width,
        height=height,
        sequence=frame.sequence,
        timestamp_ns=int(frame.timestamp * 1e9),
        depth_offset=0,
        depth_width=0,
        depth_height=0,
    )

    rgb_bytes = frame.data.tobytes()
    payload_bytes = rgb_bytes

    if depth_frame is not None:
        if depth_frame.data.ndim != _NDIM_2D:
            msg = "Depth frame must be 2D"
            raise CaptureError(msg)
        if depth_frame.data.dtype != np.uint16:
            msg = f"Unsupported depth dtype {depth_frame.data.dtype}; expected uint16"
            raise CaptureError(msg)

        depth_height, depth_width = depth_frame.data.shape
        depth_bytes = depth_frame.data.tobytes()
        header.depth_offset = HEADER_SIZE + len(rgb_bytes)
        header.depth_width = depth_width
        header.depth_height = depth_height
        payload_bytes = rgb_bytes + depth_bytes

    return header, payload_bytes


def decode_header(payload: memoryview | bytes) -> FrameHeader:
    """Decode and validate the binary protocol header from payload bytes.

    Args:
        payload: Full transport payload with header at byte offset zero.

    Returns:
        Parsed :class:`FrameHeader`.

    Raises:
        CaptureError: If payload is too small or protocol version mismatches.
    """
    if len(payload) < HEADER_SIZE:
        msg = f"Payload too small for header: got {len(payload)} bytes, need {HEADER_SIZE}"
        raise CaptureError(msg)

    header = FrameHeader.from_buffer_copy(payload[:HEADER_SIZE])
    if header.version != PROTOCOL_VERSION:
        msg = f"Unsupported protocol version {header.version}; expected {PROTOCOL_VERSION}"
        raise CaptureError(msg)
    return header


def decode_rgb(header: FrameHeader, payload: memoryview | bytes) -> Frame:
    """Decode colour image data from payload into a :class:`Frame`.

    Args:
        header: Parsed protocol header.
        payload: Full transport payload including header bytes.

    Returns:
        Decoded colour :class:`Frame`.

    Raises:
        CaptureError: If the dtype code or color mode is unsupported,
            or the payload is too small.
    """
    dtype = DTYPE_MAP.get(header.dtype)
    if dtype is None:
        msg = f"Unsupported dtype code {header.dtype}"
        raise CaptureError(msg)

    dtype_size = np.dtype(dtype).itemsize
    rgb_size = header.width * header.height * header.channels * dtype_size
    rgb_start = HEADER_SIZE
    rgb_end = rgb_start + rgb_size

    if len(payload) < rgb_end:
        msg = f"Payload too small for RGB data: got {len(payload)} bytes, need {rgb_end}"
        raise CaptureError(msg)

    rgb_bytes = payload[rgb_start:rgb_end]
    color_mode = COLOR_MODE_MAP.get(header.color_mode)
    if color_mode is None:
        msg = f"Unsupported color mode code {header.color_mode}"
        raise CaptureError(msg)

    if color_mode == ColorMode.GRAY:
        arr = np.frombuffer(rgb_bytes, dtype=dtype).reshape((header.height, header.width)).copy()
    else:
        arr = np.frombuffer(rgb_bytes, dtype=dtype).reshape((header.height, header.width, header.channels)).copy()

    return Frame(
        data=arr,
        timestamp=header.timestamp_ns / 1e9,
        sequence=header.sequence,
    )


def decode_rgb_view(header: FrameHeader, payload: memoryview) -> Frame:
    """Decode colour image as a read-only zero-copy view into *payload*.

    The returned array shares memory with *payload* and becomes invalid
    when the underlying iceoryx2 sample is released.

    Args:
        header: Parsed protocol header.
        payload: Writable memoryview over the transport payload.

    Returns:
        Decoded colour :class:`Frame` backed by a read-only view.

    Raises:
        CaptureError: If the dtype code is unsupported or the payload
            is too small.
    """
    dtype = DTYPE_MAP.get(header.dtype)
    if dtype is None:
        msg = f"Unsupported dtype code {header.dtype}"
        raise CaptureError(msg)

    dtype_size = np.dtype(dtype).itemsize
    rgb_size = header.width * header.height * header.channels * dtype_size
    rgb_start = HEADER_SIZE
    rgb_end = rgb_start + rgb_size

    if len(payload) < rgb_end:
        msg = f"Payload too small for RGB data: got {len(payload)} bytes, need {rgb_end}"
        raise CaptureError(msg)

    rgb_bytes = payload[rgb_start:rgb_end]
    if header.channels == 1:
        arr = np.frombuffer(rgb_bytes, dtype=dtype).reshape((header.height, header.width))
    else:
        arr = np.frombuffer(rgb_bytes, dtype=dtype).reshape((header.height, header.width, header.channels))

    arr.flags.writeable = False
    return Frame(
        data=arr,
        timestamp=header.timestamp_ns / 1e9,
        sequence=header.sequence,
    )


def decode_depth(header: FrameHeader, payload: memoryview | bytes) -> Frame:
    """Decode depth image data from payload into a :class:`Frame`.

    Args:
        header: Parsed protocol header.
        payload: Full transport payload including header bytes.

    Returns:
        Decoded depth :class:`Frame`.

    Raises:
        CaptureError: If the payload is too small for depth data.
    """
    if header.depth_offset == 0:
        msg = "no depth data in this stream"
        raise NotImplementedError(msg)

    depth_size = header.depth_width * header.depth_height * np.dtype(np.uint16).itemsize
    depth_end = header.depth_offset + depth_size
    if len(payload) < depth_end:
        msg = f"Payload too small for depth data: got {len(payload)} bytes, need {depth_end}"
        raise CaptureError(msg)

    depth_bytes = payload[header.depth_offset : depth_end]
    arr = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((header.depth_height, header.depth_width)).copy()

    return Frame(
        data=arr,
        timestamp=header.timestamp_ns / 1e9,
        sequence=header.sequence,
    )
