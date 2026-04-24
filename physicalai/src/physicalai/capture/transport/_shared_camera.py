# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory camera subscriber transport based on iceoryx2."""

from __future__ import annotations

import contextlib
import ctypes
import time
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from physicalai.capture.camera import Camera, CameraType, ColorMode
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError

from ._header import decode_header, decode_rgb

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.capture.frame import Frame
    from physicalai.capture.transport._publisher import CameraPublisher


_SERVICE_NAME_EXPECTED_PARTS = 5


def _probe_service(service_name: str) -> bool:
    """Check if a publisher is serving *service_name*.

    Returns:
        ``True`` if a publisher is reachable, ``False`` otherwise.
    """
    try:
        iox2 = cast("Any", import_module("iceoryx2"))

        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        try:
            svc = (
                node.service_builder(
                    iox2.ServiceName.new(service_name),
                )
                .publish_subscribe(iox2.Slice[ctypes.c_uint8])
                .open()
            )
        except Exception:  # noqa: BLE001
            return False
        else:
            del svc
            return True
        finally:
            del node
    except Exception:  # noqa: BLE001
        return False


def _probe_with_retry(service_name: str, timeout: float, interval: float = 0.1) -> bool:
    """Poll ``_probe_service`` until it returns True or *timeout* elapses.

    Returns:
        ``True`` if a publisher is reachable within timeout, ``False`` otherwise.
    """
    deadline = time.monotonic() + timeout
    while True:
        if _probe_service(service_name):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def _derive_service_name(camera_type: str, camera_kwargs: Mapping[str, object]) -> str:
    device_id = camera_kwargs.get("serial_number", camera_kwargs.get("device", 0))
    return f"physicalai/camera/{camera_type}/{device_id}/frame"


class SharedCamera(Camera):
    """Camera subscriber that reads frames from shared memory via iceoryx2.

    Connects to a publisher process that owns the physical camera device.
    Multiple SharedCamera instances can subscribe to the same publisher
    for zero-copy fan-out.

    Args:
        camera_type: Logical camera type (auto-spawn mode), or ``None`` to
            subscribe to an existing publisher only.
        color_mode: Pixel format for returned frames.
    """

    def __init__(
        self,
        camera_type: CameraType | str | None,
        *,
        color_mode: ColorMode = ColorMode.RGB,
        zero_copy: bool = False,
        service_name: str | None = None,
        **camera_kwargs: object,
    ) -> None:
        try:
            camera_type = CameraType(camera_type) if camera_type is not None else None
        except ValueError as exc:
            valid = ", ".join(m.value for m in CameraType)
            msg = f"unknown camera_type {camera_type!r}; expected one of: {valid}"
            raise ValueError(msg) from exc

        if camera_type is None and service_name is None:
            msg = "must provide camera_type or service_name"
            raise ValueError(msg)

        if camera_type is not None:
            service_name = _derive_service_name(camera_type, camera_kwargs)
        elif service_name is None:
            msg = "service_name must be provided if camera_type is None"
            raise ValueError(msg)

        super().__init__(color_mode=color_mode)
        self._camera_type = camera_type
        self._camera_kwargs = camera_kwargs
        self._service_name: str = service_name
        self._zero_copy = zero_copy
        self._publisher: CameraPublisher | None = None
        self._connected = False
        self._latest: Frame | None = None
        self._held_sample: Any = None
        self._node: Any | None = None
        self._subscriber: Any | None = None
        self._listener: Any | None = None

    @classmethod
    def from_publisher(
        cls,
        service_name: str,
        *,
        color_mode: ColorMode = ColorMode.RGB,
        zero_copy: bool = False,
    ) -> SharedCamera:
        return cls(None, color_mode=color_mode, zero_copy=zero_copy, service_name=service_name)

    def connect(self, timeout: float = 5.0) -> None:
        if self._connected:
            return

        if self._camera_type is not None and not _probe_service(self._service_name):
            from ._publisher import CameraPublisher  # noqa: PLC0415
            from ._spec import CameraSpec  # noqa: PLC0415

            spec = CameraSpec(self._camera_type, self._camera_kwargs)
            publisher = CameraPublisher(spec, self._service_name)
            try:
                publisher.start()
            except Exception as exc:
                if _probe_with_retry(self._service_name, timeout=3.0):
                    logger.debug(f"Lost publisher race for {self._service_name} — using existing")
                else:
                    msg = f"failed to start camera publisher for {self._service_name}"
                    raise CaptureError(msg) from exc
            else:
                self._publisher = publisher

        iox2 = cast("Any", import_module("iceoryx2"))

        # Silence iceoryx2's kHz-rate ``FailedToDeliverSignal`` warnings.
        # They fire whenever a listener's unix-datagram wake-up socket is
        # full — which happens for any subscriber using non-blocking reads
        # (e.g. ``read_latest``) under camera-rate notifications. The
        # pub-sub payload itself still delivers reliably.
        with contextlib.suppress(Exception):
            iox2.set_log_level(iox2.LogLevel.Error)

        self._node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

        pub_sub = (
            self._node.service_builder(iox2.ServiceName.new(self._service_name))
            .publish_subscribe(iox2.Slice[ctypes.c_uint8])
            .open()
        )
        self._subscriber = pub_sub.subscriber_builder().create()

        event_svc = self._node.service_builder(iox2.ServiceName.new(f"{self._service_name}/notify")).event().open()
        self._listener = event_svc.listener_builder().create()

        logger.debug(f"Connecting SharedCamera subscriber to {self._service_name}")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sample = self._subscriber.receive()
            if sample is not None:
                self._latest = self._decode_sample(sample)
                self._connected = True
                return

            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._listener.timed_wait_one(
                    iox2.Duration.from_secs_f64(min(remaining, 0.5)),
                )

        logger.debug(f"SharedCamera connect timeout for {self._service_name}")
        self._do_disconnect()
        msg = "no publisher responded within timeout"
        raise CaptureTimeoutError(msg)

    def read(self, timeout: float | None = None) -> Frame:
        if not self._connected or self._subscriber is None or self._listener is None:
            msg = "shared camera is not connected"
            raise NotConnectedError(msg)

        iox2 = cast("Any", import_module("iceoryx2"))

        wait_timeout = 3600.0 if timeout is None else timeout
        event = self._listener.timed_wait_one(iox2.Duration.from_secs_f64(wait_timeout))
        if event is None:
            msg = "timed out waiting for frame"
            raise CaptureTimeoutError(msg)

        self._held_sample = None  # release previous borrow before draining
        newest_sample = None
        while True:
            sample = self._subscriber.receive()
            if sample is None:
                break
            newest_sample = sample

        if newest_sample is not None:
            self._latest = self._decode_sample(newest_sample)

        if self._latest is None:
            msg = "no frame available"
            raise CaptureTimeoutError(msg)

        return self._latest

    def read_latest(self) -> Frame:
        if not self._connected or self._subscriber is None:
            msg = "shared camera is not connected"
            raise NotConnectedError(msg)

        self._held_sample = None  # release previous borrow before draining
        newest_sample = None
        while True:
            sample = self._subscriber.receive()
            if sample is None:
                break
            newest_sample = sample

        if newest_sample is not None:
            self._latest = self._decode_sample(newest_sample)

        if self._latest is None:
            msg = "no frame available"
            raise CaptureTimeoutError(msg)

        return self._latest

    def _decode_sample(self, sample: Any) -> Frame:  # noqa: ANN401
        import ctypes as _ct  # noqa: PLC0415

        slc = sample.payload()
        buf = (_ct.c_uint8 * slc.number_of_elements).from_address(slc.data_ptr)
        header = decode_header(bytes(buf))
        if self._zero_copy:
            from ._header import decode_rgb_view  # noqa: PLC0415

            self._held_sample = sample
            # ``toreadonly()`` marks the buffer read-only at the buffer-protocol
            # level. ``np.frombuffer`` then produces an array whose ``writeable``
            # flag cannot be flipped back to True, so consumer writes — including
            # accidental in-place ops like ``arr += 1`` or OpenCV drawing calls —
            # fail fast instead of silently corrupting the iceoryx2 SHM segment
            # (which would be seen by every other subscriber). Bounds metadata on
            # the memoryview also prevents out-of-bounds indexing past the
            # payload. Consumers that need to modify the frame must ``.copy()``.
            return decode_rgb_view(header, memoryview(buf).toreadonly())
        return decode_rgb(header, bytes(buf))

    def _do_disconnect(self) -> None:
        self._held_sample = None
        self._publisher = None
        self._subscriber = None
        self._listener = None
        self._node = None
        self._connected = False
        self._latest = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def service_name(self) -> str:
        return self._service_name

    @property
    def device_id(self) -> str:
        # service_name format: physicalai/camera/<type>/<device_id>/frame
        parts = self._service_name.split("/")
        if (
            len(parts) >= _SERVICE_NAME_EXPECTED_PARTS
            and parts[0] == "physicalai"
            and parts[1] == "camera"
            and parts[4] == "frame"
        ):
            return parts[3]
        # Custom service_name (from_publisher) — no embedded device id.
        return ""
