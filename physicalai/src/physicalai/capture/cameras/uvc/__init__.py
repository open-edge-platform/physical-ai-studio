# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""UVC camera type implementation.

Public exports:
  - :class:`~physicalai.capture.cameras.uvc.UVCCamera`
  - :func:`~physicalai.capture.cameras.uvc.discover_uvc`
"""

from __future__ import annotations

from ._camera import UVCCamera
from ._camera_setting import CameraSetting
from ._discover import discover_uvc

__all__ = ["CameraSetting", "UVCCamera", "discover_uvc"]
