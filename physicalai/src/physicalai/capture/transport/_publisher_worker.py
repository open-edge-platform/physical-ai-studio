# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standalone publisher worker (``python -m physicalai.capture.transport._publisher_worker``)."""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import json
import os
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from types import FrameType

    from physicalai.capture.camera import Camera

shutdown = threading.Event()


def sigterm_handler(_signum: int, _frame: FrameType | None) -> None:
    shutdown.set()


def signal_ready() -> None:
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    sys.stdout.close()


def signal_error(msg: str) -> None:
    sys.stdout.write(f"ERROR:{json.dumps(msg)}\n")
    sys.stdout.flush()
    sys.stdout.close()


def suppress_stdout() -> int:
    """Redirect fd 1 to /dev/null.

    Operates at the OS file-descriptor level (not ``sys.stdout``) so that
    native libraries writing directly to the C stdout fd (e.g.
    ``omni_camera``/Nokhwa via ``printf``) are also silenced. This keeps
    the single-line ``READY``/``ERROR`` IPC protocol on stdout uncorrupted
    during camera startup.

    Returns:
        The saved original fd 1, to be passed to :func:`restore_stdout`.
    """
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    return saved_fd


def restore_stdout(saved_fd: int) -> None:
    """Restore fd 1 from *saved_fd* and rewrap ``sys.stdout``.

    Undoes :func:`suppress_stdout` by pointing fd 1 back at the original
    stdout and rebuilding the Python ``sys.stdout`` text wrapper so that
    subsequent ``print`` calls reach the parent process again.

    Args:
        saved_fd: The fd returned by :func:`suppress_stdout`.
    """
    os.dup2(saved_fd, 1)
    os.close(saved_fd)
    sys.stdout = os.fdopen(1, "w")


def build_camera(config: dict) -> Camera:
    factory_override = config.get("_factory_override")
    if factory_override:
        module_path, _, attr = factory_override.rpartition(":")
        mod = importlib.import_module(module_path)
        factory = getattr(mod, attr)
        return factory(**config.get("camera_kwargs", {}))

    from physicalai.capture.transport._spec import CameraSpec  # noqa: PLC0415, PLC2701

    spec = CameraSpec.from_json_dict(config)
    return spec.build()


def main() -> int:  # noqa: PLR0912, PLR0914, PLR0915
    """Entry point for the publisher worker process.

    Returns:
        Exit code: 0 on success, 1 on startup failure.
    """
    signal.signal(signal.SIGTERM, sigterm_handler)

    raw = sys.stdin.read()
    sys.stdin.close()
    try:
        config = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        signal_error(f"invalid JSON config: {exc}")
        return 1

    service_name: str = config["service_name"]
    idle_timeout: float = config.get("idle_timeout", 5.0)
    max_subscribers: int = config.get("max_subscribers", 32)

    # Redirect fd 1 (stdout) to /dev/null during setup so that native
    # libraries (e.g. omni_camera/Nokhwa) that write directly to the C-level
    # stdout fd don't corrupt the single-line IPC protocol used by
    # CameraPublisher.start().
    saved_stdout_fd: int | None = suppress_stdout()

    camera = None
    try:
        iox2 = importlib.import_module("iceoryx2")

        # Silence iceoryx2's internal log warnings. The notifier logs a
        # ``FailedToDeliverSignal`` warning at WARN level whenever a
        # listener's unix-datagram socket is full (common under load —
        # the pub-sub payload still delivers reliably, only the wake-up
        # hint is dropped). These warnings can flood stderr at kHz
        # rates. Error level keeps genuine errors visible.
        with contextlib.suppress(Exception):
            iox2.set_log_level(iox2.LogLevel.Error)

        camera = build_camera(config)
        camera.connect()

        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        max_nodes = max_subscribers + 2
        service = (
            node.service_builder(iox2.ServiceName.new(service_name))
            .publish_subscribe(iox2.Slice[ctypes.c_uint8])
            .max_publishers(1)
            .max_subscribers(max_subscribers)
            .max_nodes(max_nodes)
            .open_or_create()
        )

        from physicalai.capture.transport._header import HEADER_SIZE, encode_frame  # noqa: PLC0415, PLC2701

        first_frame = camera.read_latest()
        max_slice_len = HEADER_SIZE + first_frame.data.nbytes
        publisher = (
            service.publisher_builder()
            .initial_max_slice_len(max_slice_len)
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
            .create()
        )

        event_service = (
            node.service_builder(iox2.ServiceName.new(f"{service_name}/notify"))
            .event()
            .max_listeners(max_subscribers)
            .max_nodes(max_nodes)
            .open_or_create()
        )
        notifier = event_service.notifier_builder().create()
    except Exception as exc:  # noqa: BLE001
        if saved_stdout_fd is not None:
            restore_stdout(saved_stdout_fd)
            saved_stdout_fd = None
        signal_error(str(exc))
        if camera is not None:
            try:
                camera.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception("camera disconnect failed during error cleanup")
        return 1
    finally:
        if saved_stdout_fd is not None:
            restore_stdout(saved_stdout_fd)

    signal_ready()

    from physicalai.capture.errors import CaptureError  # noqa: PLC0415

    node_check_interval = 1.0

    try:
        idle_since: float | None = None
        last_node_check = 0.0
        while not shutdown.is_set():
            try:
                frame = camera.read(timeout=1.0)
            except CaptureError:
                continue

            header, payload_bytes = encode_frame(frame, camera.color_mode)
            header_bytes = bytes(header)
            total_size = HEADER_SIZE + len(payload_bytes)

            sample = publisher.loan_slice_uninit(total_size)
            shm_ptr = sample.payload().data_ptr
            ctypes.memmove(shm_ptr, header_bytes, HEADER_SIZE)
            ctypes.memmove(shm_ptr + HEADER_SIZE, payload_bytes, len(payload_bytes))
            sample.assume_init().send()

            with contextlib.suppress(Exception):
                notifier.notify_with_custom_event_id(iox2.EventId.new(0))

            now = time.monotonic()
            if now - last_node_check >= node_check_interval:
                last_node_check = now
                # service.nodes includes the publisher's own node — subtract 1.
                sub_count = max(0, len(service.nodes) - 1)

                if sub_count == 0:
                    if idle_since is None:
                        idle_since = now
                    elif now - idle_since > idle_timeout:
                        logger.info(
                            f"No subscribers for {idle_timeout}s -- shutting down publisher for {service_name}",
                        )
                        break
                else:
                    idle_since = None
    except Exception:  # noqa: BLE001
        logger.exception(f"publisher loop failed for service {service_name}")
    finally:
        try:
            camera.disconnect()
        except Exception:  # noqa: BLE001
            logger.exception(f"camera disconnect failed for service {service_name}")
        # Release iceoryx2 FFI resources deterministically (not via GC).
        with contextlib.suppress(NameError):
            del publisher, service, event_service, notifier, node

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
