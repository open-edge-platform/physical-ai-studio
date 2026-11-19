# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pickle-free request/response frames shared by the host and GR00T workers.

A frame is ``MAGIC | header length | JSON header | raw array buffers``. The
JSON header carries the message body with arrays replaced by indexed
placeholders. Only fixed-size numeric and boolean dtypes are accepted, so a
peer can never make the other side execute code while decoding.

This module runs on both sides of the process boundary: in the host
(Python 3.12+, NumPy 2) and in the GR00T worker environments (Python 3.10+,
NumPy 1.26). It must only use the standard library and NumPy.
"""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO

import numpy as np

MAGIC = b"PGW1"
_PREFIX = struct.Struct(">4sI")
_ARRAY_KEY = "__ndarray__"
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 4 * 1024 * 1024 * 1024
_ALLOWED_DTYPES = frozenset(
    {"|b1", "|u1", "|i1", "<u2", "<i2", "<u4", "<i4", "<u8", "<i8", "<f2", "<f4", "<f8"},
)


class ProtocolError(RuntimeError):
    """Raised when a frame is malformed or contains a disallowed value."""


def encode(message: dict[str, Any]) -> bytes:
    """Serialize *message* into one frame.

    Args:
        message: JSON-compatible mapping that may contain NumPy arrays.

    Returns:
        Encoded frame bytes.

    Raises:
        ProtocolError: If the message holds a disallowed value or is too large.
    """
    arrays: list[np.ndarray] = []
    body = _pack(message, arrays)
    header = json.dumps({"body": body, "sizes": [array.nbytes for array in arrays]}, separators=(",", ":"))
    encoded = header.encode("utf-8")
    if len(encoded) > _MAX_HEADER_BYTES:
        msg = "Frame header exceeds the protocol limit."
        raise ProtocolError(msg)
    return b"".join([_PREFIX.pack(MAGIC, len(encoded)), encoded, *(array.tobytes() for array in arrays)])


def write_frame(stream: BinaryIO, message: dict[str, Any]) -> None:
    """Write one frame to *stream* and flush it.

    Args:
        stream: Binary output stream.
        message: Message to send.
    """
    stream.write(encode(message))
    stream.flush()


def read_frame(stream: BinaryIO) -> dict[str, Any] | None:
    """Read one frame from *stream*.

    Args:
        stream: Binary input stream.

    Returns:
        The decoded message, or ``None`` when the peer closed the stream.

    Raises:
        ProtocolError: If the frame is malformed.
    """
    prefix = _read_prefix(stream)
    if prefix is None:
        return None
    magic, header_size = _PREFIX.unpack(prefix)
    if magic != MAGIC or header_size > _MAX_HEADER_BYTES:
        msg = "Invalid frame prefix."
        raise ProtocolError(msg)
    header = json.loads(_read_exact(stream, header_size).decode("utf-8"))
    if not isinstance(header, dict) or not isinstance(header.get("sizes"), list):
        msg = "Frame header must be an object with a 'sizes' list."
        raise ProtocolError(msg)
    sizes = header["sizes"]
    if not all(isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in sizes):
        msg = "Frame buffer sizes must be non-negative integers."
        raise ProtocolError(msg)
    if sum(sizes) > _MAX_PAYLOAD_BYTES:
        msg = "Frame payload exceeds the protocol limit."
        raise ProtocolError(msg)
    buffers = [_read_exact(stream, size) for size in sizes]
    body = _unpack(header.get("body"), buffers)
    if not isinstance(body, dict):
        msg = "Frame body must be an object."
        raise ProtocolError(msg)
    return body


def _pack(value: Any, arrays: list[np.ndarray]) -> Any:  # ruff: ignore[any-type]  # JSON-shaped values are dynamically typed.
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.byteorder == ">":
            array = array.astype(array.dtype.newbyteorder("<"))
        dtype = array.dtype.str
        if dtype not in _ALLOWED_DTYPES:
            msg = f"Arrays of dtype {array.dtype} cannot cross the worker boundary."
            raise ProtocolError(msg)
        arrays.append(array)
        return {_ARRAY_KEY: len(arrays) - 1, "dtype": dtype, "shape": list(array.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        packed = {}
        for key, item in value.items():
            if not isinstance(key, str) or key == _ARRAY_KEY:
                msg = f"Invalid message key {key!r}."
                raise ProtocolError(msg)
            packed[key] = _pack(item, arrays)
        return packed
    if isinstance(value, (list, tuple)):
        return [_pack(item, arrays) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    msg = f"Values of type {type(value).__name__} cannot cross the worker boundary."
    raise ProtocolError(msg)


def _unpack(value: Any, buffers: list[bytes]) -> Any:  # ruff: ignore[any-type]  # JSON-shaped values are dynamically typed.
    if isinstance(value, dict):
        if _ARRAY_KEY in value:
            return _array(value, buffers)
        return {key: _unpack(item, buffers) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item, buffers) for item in value]
    return value


def _array(spec: dict[str, Any], buffers: list[bytes]) -> np.ndarray:
    index, dtype, shape = spec.get(_ARRAY_KEY), spec.get("dtype"), spec.get("shape")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(buffers):
        msg = "Array placeholder references a missing buffer."
        raise ProtocolError(msg)
    if dtype not in _ALLOWED_DTYPES:
        msg = f"Disallowed array dtype {dtype!r}."
        raise ProtocolError(msg)
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        msg = "Array shape must be a list of non-negative integers."
        raise ProtocolError(msg)
    array = np.frombuffer(buffers[index], dtype=np.dtype(dtype))
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if array.size != expected:
        msg = f"Array buffer holds {array.size} items, but shape {shape} needs {expected}."
        raise ProtocolError(msg)
    return array.reshape(shape).copy()


def _read_prefix(stream: BinaryIO) -> bytes | None:
    first = stream.read(1)
    if not first:
        return None
    return first + _read_exact(stream, _PREFIX.size - 1)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            msg = "Peer closed the stream in the middle of a frame."
            raise ProtocolError(msg)
        chunks.extend(chunk)
    return bytes(chunks)
