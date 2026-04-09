# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Frame dataclass — the universal return type for all read operations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Frame:
    """A captured image with metadata.

    Every :meth:`~physicalai.capture.camera.Camera.read` call returns a
    ``Frame``. The frozen dataclass prevents accidental mutation of
    metadata; the underlying ``data`` buffer is still mutable.

    Attributes:
        data: Image array. ``(H, W, 3)`` uint8 for colour,
            ``(H, W)`` uint8 for grayscale, ``(H, W)`` uint16 for depth.
        timestamp: ``time.monotonic()`` at the moment of capture.
        sequence: Monotonically increasing counter per source (0, 1, 2, …).
            Gaps indicate dropped frames.
    """

    data: NDArray[np.uint8]
    timestamp: float
    sequence: int
