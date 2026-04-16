# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Multi-camera synchronisation utilities."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from physicalai.capture.errors import CaptureTimeoutError

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera
    from physicalai.capture.frame import Frame


@dataclass(frozen=True)
class SyncedFrames:
    """Temporally aligned frames from multiple cameras.

    Attributes:
        frames: Mapping of camera name to captured frame.
        max_skew_ms: Worst-case temporal skew across all frames,
            in milliseconds (``max(timestamps) - min(timestamps)``).
    """

    frames: dict[str, Frame]
    max_skew_ms: float


def read_cameras(
    cameras: dict[str, Camera],
    timeout: float = 1.0,
    *,
    latest: bool = True,
) -> SyncedFrames:
    """Read one frame from each camera in parallel.

    Spawns one thread per camera and reads simultaneously to minimise
    temporal skew between captures.

    Args:
        cameras: Mapping of name to connected ``Camera`` instance.
        timeout: Maximum seconds to wait for all cameras.
        latest: If ``True`` (default), use ``read_latest()``.
            If ``False``, use ``read(timeout=timeout)``.

    Returns:
        Temporally aligned frames from all cameras.

    Raises:
        CaptureTimeoutError: One or more cameras did not respond in time.
    """
    frames: dict[str, Frame] = {}

    with ThreadPoolExecutor(max_workers=len(cameras)) as pool:
        if latest:
            futures = {pool.submit(cam.read_latest): name for name, cam in cameras.items()}
        else:
            futures = {pool.submit(cam.read, timeout): name for name, cam in cameras.items()}

        for future in as_completed(futures, timeout=timeout):
            name = futures[future]
            frames[name] = future.result()

    if len(frames) != len(cameras):
        missing = set(cameras) - set(frames)
        msg = f"Cameras did not respond within {timeout}s: {missing}"
        raise CaptureTimeoutError(msg)

    timestamps = [f.timestamp for f in frames.values()]
    skew_ms = (max(timestamps) - min(timestamps)) * 1000.0

    return SyncedFrames(frames=frames, max_skew_ms=skew_ms)


async def async_read_cameras(
    cameras: dict[str, Camera],
    timeout: float = 1.0,  # noqa: ASYNC109
    *,
    latest: bool = True,
) -> SyncedFrames:
    """Async version of :func:`read_cameras`.

    Uses ``asyncio.gather`` to read all cameras concurrently via
    each camera's :meth:`~Camera.async_read`.

    Args:
        cameras: Mapping of name to connected ``Camera`` instance.
        timeout: Maximum seconds to wait for all cameras.
        latest: If ``True`` (default), use ``read_latest()`` (offloaded
            to executor).  If ``False``, use ``async_read(timeout=timeout)``.

    Returns:
        Temporally aligned frames from all cameras.
    """
    names = list(cameras.keys())
    loop = asyncio.get_running_loop()

    if latest:
        tasks = [loop.run_in_executor(None, cam.read_latest) for cam in cameras.values()]
    else:
        tasks = [cam.async_read(timeout=timeout) for cam in cameras.values()]

    async with asyncio.timeout(timeout):
        results = await asyncio.gather(*tasks)

    frames = dict(zip(names, results, strict=True))

    timestamps = [f.timestamp for f in frames.values()]
    skew_ms = (max(timestamps) - min(timestamps)) * 1000.0

    return SyncedFrames(frames=frames, max_skew_ms=skew_ms)
