# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""V4L2 camera controls via ioctl.

Stateless helper — no lifecycle management required by callers.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

from physicalai.capture.cameras.uvc._camera_setting import CameraSetting  # noqa: PLC2701
from physicalai.capture.errors import CaptureError

from ._ioctl import (
    CTRL_TYPE_NAMES,
    V4L2_CTRL_FLAG_DISABLED,
    V4L2_CTRL_FLAG_INACTIVE,
    V4L2_CTRL_FLAG_NEXT_CTRL,
    V4L2_CTRL_FLAG_READ_ONLY,
    V4L2_CTRL_TYPE_CTRL_CLASS,
    VIDIOC_G_CTRL,
    VIDIOC_QUERYCTRL,
    VIDIOC_QUERYMENU,
    VIDIOC_S_CTRL,
    v4l2_control,
    v4l2_queryctrl,
    v4l2_querymenu,
    xioctl,
)

if TYPE_CHECKING:
    from collections.abc import Generator


class V4L2CameraControls:
    """V4L2 ioctl-based camera controls. Stateless — no lifecycle management needed.

    Two construction patterns:
    - ``V4L2CameraControls(device_path)`` — each method call opens/closes its own fd.
    - ``V4L2CameraControls(device_path, fd=existing_fd)`` — reuses a caller-owned fd.
    """

    def __init__(self, device_path: str, *, fd: int | None = None) -> None:
        self._device_path = device_path
        self._shared_fd = fd

    @contextlib.contextmanager
    def _fd_scope(self) -> Generator[int, None, None]:
        if self._shared_fd is not None:
            yield self._shared_fd
        else:
            fd = os.open(self._device_path, os.O_RDWR | os.O_NONBLOCK)
            try:
                yield fd
            finally:
                os.close(fd)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_controls(self) -> list[CameraSetting]:
        """Enumerate all supported V4L2 controls on this device.

        Walks the control ID space using ``V4L2_CTRL_FLAG_NEXT_CTRL``
        and returns metadata for each non-disabled control.  Control
        class group headers are excluded.

        Returns:
            The supported controls exposed by the device.
        """
        with self._fd_scope() as fd:
            controls: list[CameraSetting] = []
            next_id = V4L2_CTRL_FLAG_NEXT_CTRL

            while True:
                qc = v4l2_queryctrl()
                qc.id = next_id
                try:
                    xioctl(fd, VIDIOC_QUERYCTRL, qc)
                except OSError:
                    break

                next_id = qc.id | V4L2_CTRL_FLAG_NEXT_CTRL

                if qc.flags & V4L2_CTRL_FLAG_DISABLED:
                    continue

                if qc.type == V4L2_CTRL_TYPE_CTRL_CLASS:
                    continue

                controls.append(self._query_control_with_fd(fd, qc.id))

            return controls

    def get_control(self, control_id: int) -> int:
        """Read the current value of a V4L2 control.

        Args:
            control_id: V4L2 control ID.

        Returns:
            The current integer value of the control.

        Raises:
            CaptureError: If the control value cannot be read.
        """
        with self._fd_scope() as fd:
            ctrl = v4l2_control()
            ctrl.id = control_id
            try:
                xioctl(fd, VIDIOC_G_CTRL, ctrl)
            except OSError as exc:
                msg = f"Failed to get control 0x{control_id:08x}: {exc}"
                raise CaptureError(msg) from exc
            return ctrl.value

    def set_control(self, control_id: int, value: int | bool | str) -> None:  # noqa: FBT001
        """Set a V4L2 control value.

        Args:
            control_id: V4L2 control ID.
            value: Value to set (int for integer/menu, bool for boolean,
                   str menu label for menu controls).
        """
        with self._fd_scope() as fd:
            self._set_control_with_fd(fd, control_id, value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_control_with_fd(fd: int, control_id: int, value: int | bool | str) -> None:  # noqa: FBT001
        ctrl = v4l2_control()
        ctrl.id = control_id
        ctrl.value = value
        try:
            xioctl(fd, VIDIOC_S_CTRL, ctrl)
        except OSError as exc:
            msg = f"Failed to set control 0x{control_id:08x} to {value}: {exc}"
            raise CaptureError(msg) from exc

    def _query_control_with_fd(self, fd: int, control_id: int) -> CameraSetting:
        qc = v4l2_queryctrl()
        qc.id = control_id
        try:
            xioctl(fd, VIDIOC_QUERYCTRL, qc)
        except OSError as exc:
            msg = f"Failed to query control 0x{control_id:08x}: {exc}"
            raise CaptureError(msg) from exc

        type_name = CTRL_TYPE_NAMES.get(qc.type, f"unknown({qc.type})")
        is_rangeless = type_name in {"boolean", "button"}

        ctrl = v4l2_control()
        ctrl.id = control_id
        try:
            xioctl(fd, VIDIOC_G_CTRL, ctrl)
        except OSError:
            ctrl.value = qc.default_value

        if type_name == "boolean":
            coerced_default: int | bool = bool(qc.default_value)
            coerced_value: int | bool | None = bool(ctrl.value)
        elif type_name == "button":
            coerced_default = 0
            coerced_value = None
        else:
            coerced_default = qc.default_value
            coerced_value = ctrl.value

        return CameraSetting(
            id=qc.id,
            name=qc.name.decode(errors="replace").rstrip("\x00"),
            setting_type=type_name,
            min=None if is_rangeless else qc.minimum,
            max=None if is_rangeless else qc.maximum,
            step=None if is_rangeless else qc.step,
            default=coerced_default,
            value=coerced_value,
            inactive=bool(qc.flags & V4L2_CTRL_FLAG_INACTIVE),
            read_only=bool(qc.flags & V4L2_CTRL_FLAG_READ_ONLY),
            menu_items=self._query_menu_items(fd, qc.id, qc.minimum, qc.maximum) if type_name == "menu" else None,
        )

    @staticmethod
    def _query_menu_items(fd: int, control_id: int, min_index: int, max_index: int) -> dict[int, str]:
        menu_items: dict[int, str] = {}
        for index in range(min_index, max_index + 1):
            menu = v4l2_querymenu()
            menu.id = control_id
            menu.index = index
            try:
                xioctl(fd, VIDIOC_QUERYMENU, menu)
            except OSError:
                continue
            menu_items[index] = menu.u.name.decode(errors="replace").rstrip("\x00")

        return menu_items


__all__ = ["V4L2CameraControls"]
