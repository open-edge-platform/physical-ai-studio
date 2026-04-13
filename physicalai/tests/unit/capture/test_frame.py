# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Frame dataclass."""

import time

import numpy as np
import pytest

from physicalai.capture.frame import Frame


class TestFrame:
    """Frame construction and immutability."""

    def test_rgb_frame_shape(self) -> None:
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=data, timestamp=1.0, sequence=0)
        assert frame.data.shape == (480, 640, 3)
        assert frame.data.dtype == np.uint8

    def test_grayscale_frame_shape(self) -> None:
        data = np.zeros((480, 640), dtype=np.uint8)
        frame = Frame(data=data, timestamp=1.0, sequence=0)
        assert frame.data.shape == (480, 640)

    def test_frozen_metadata(self) -> None:
        frame = Frame(data=np.zeros((1, 1), dtype=np.uint8), timestamp=1.0, sequence=0)
        with pytest.raises(AttributeError):
            frame.timestamp = 2.0  # type: ignore[misc]

    def test_sequence_and_timestamp(self) -> None:
        t = time.monotonic()
        frame = Frame(data=np.zeros((1, 1), dtype=np.uint8), timestamp=t, sequence=42)
        assert frame.sequence == 42
        assert frame.timestamp == t
