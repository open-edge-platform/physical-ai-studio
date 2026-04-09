# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Depth capture mixin for cameras with depth sensing."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame


class DepthMixin:
    """Adds depth capture capability.

    Classes using this mixin must also inherit from
    :class:`~physicalai.capture.camera.Camera` (which provides
    :meth:`read`).
    """

    @abstractmethod
    def read_depth(self) -> Frame:
        """Read a depth frame.

        Returns:
            Frame with ``data`` as ``(H, W)`` uint16 array in millimetres.
        """
        ...

    def read_rgbd(self) -> tuple[Frame, Frame]:
        """Read aligned RGB and depth frames.

        Returns:
            ``(rgb_frame, depth_frame)``.  RGB is uint8, depth is uint16.
            Default implementation calls :meth:`read` and
            :meth:`read_depth` sequentially; subclasses may override for
            hardware-aligned capture.
        """
        return self.read(), self.read_depth()  # type: ignore[attr-defined]
