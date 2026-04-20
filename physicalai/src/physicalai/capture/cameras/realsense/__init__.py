# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RealSense camera type implementation.

Public exports:
  - :class:`~physicalai.capture.cameras.realsense.RealSenseCamera`
  - :func:`~physicalai.capture.cameras.realsense.discover_realsense`
"""

from __future__ import annotations

from ._camera import RealSenseCamera
from ._discover import discover_realsense

__all__ = ["RealSenseCamera", "discover_realsense"]
