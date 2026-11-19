# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pickle-free worker frame protocol."""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest

from physicalai_gr00t_policy_plugin._worker import gr00t_worker_protocol as protocol


def _roundtrip(message: dict) -> dict:
    stream = io.BytesIO()
    protocol.write_frame(stream, message)
    stream.seek(0)
    decoded = protocol.read_frame(stream)
    assert decoded is not None
    assert protocol.read_frame(stream) is None
    return decoded


def test_nested_messages_and_numeric_arrays_roundtrip() -> None:
    image = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    state = np.linspace(-1, 1, 8, dtype=np.float32).reshape(1, 8)
    message = {
        "op": "predict",
        "video": {"image": image},
        "state": {"x": state, "mask": np.array([True, False])},
        "steps": np.int64(7),
        "task": "pick up the mug",
        "seed": None,
        "values": [1, 2.5, {"nested": np.zeros((0, 3), dtype=np.int16)}],
    }
    decoded = _roundtrip(message)
    np.testing.assert_array_equal(decoded["video"]["image"], image)
    np.testing.assert_array_equal(decoded["state"]["x"], state)
    assert decoded["state"]["mask"].dtype == np.bool_
    assert decoded["steps"] == 7
    assert decoded["task"] == "pick up the mug"
    assert decoded["seed"] is None
    assert decoded["values"][2]["nested"].shape == (0, 3)
    assert decoded["video"]["image"].flags.writeable


def test_big_endian_arrays_are_sent_little_endian() -> None:
    value = np.arange(4, dtype=">f4")
    decoded = _roundtrip({"value": value})
    assert decoded["value"].dtype == np.dtype("<f4")
    np.testing.assert_array_equal(decoded["value"], value)


@pytest.mark.parametrize("value", [np.array([object()]), np.array(["text"]), {1: "non-string key"}, {1, 2}])
def test_values_that_could_carry_code_or_ambiguity_are_rejected(value: object) -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.encode({"value": value})


def test_decoder_rejects_object_dtype_frames_without_unpickling() -> None:
    header = json.dumps({"body": {"value": {"__ndarray__": 0, "dtype": "|O", "shape": [1]}}, "sizes": [8]}).encode()
    frame = struct.pack(">4sI", protocol.MAGIC, len(header)) + header + b"\x80\x04N." + b"\x00" * 4
    with pytest.raises(protocol.ProtocolError, match="Disallowed array dtype"):
        protocol.read_frame(io.BytesIO(frame))


def test_decoder_rejects_inconsistent_shapes_and_bad_prefixes() -> None:
    header = json.dumps({"body": {"value": {"__ndarray__": 0, "dtype": "<f4", "shape": [3]}}, "sizes": [4]}).encode()
    frame = struct.pack(">4sI", protocol.MAGIC, len(header)) + header + b"\x00" * 4
    with pytest.raises(protocol.ProtocolError, match="needs 3"):
        protocol.read_frame(io.BytesIO(frame))
    with pytest.raises(protocol.ProtocolError, match="Invalid frame prefix"):
        protocol.read_frame(io.BytesIO(struct.pack(">4sI", b"XXXX", 2) + b"{}"))


def test_truncated_frames_raise() -> None:
    frame = protocol.encode({"value": np.ones(16, dtype=np.float32)})
    with pytest.raises(protocol.ProtocolError, match="middle of a frame"):
        protocol.read_frame(io.BytesIO(frame[:-1]))
