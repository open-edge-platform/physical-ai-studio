# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Byte-transfer formatting and throttled progress heartbeats.

Shared by the remote backend's snapshot upload and model download paths so both
render identical human-readable sizes, throughput, and heartbeat log lines.
"""

from __future__ import annotations

import time

from loguru import logger

# Minimum wall-clock gap between intermediate transfer progress log lines.
_TRANSFER_LOG_INTERVAL_S = 15.0


def format_bytes(num_bytes: float) -> str:
    """Render a byte count as a human-readable string (e.g. ``1.5 GiB``)."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def format_throughput(num_bytes: int, elapsed_s: float) -> str:
    """Render an average transfer rate as ``<size>/s`` for the elapsed window."""
    if elapsed_s <= 0:
        return "n/a"
    return f"{format_bytes(num_bytes / elapsed_s)}/s"


class TransferProgressLogger:
    """Emit throttled ``INFO`` heartbeats while bytes stream in or out.

    Logs at most once per :data:`_TRANSFER_LOG_INTERVAL_S` and always logs the
    first heartbeat once any bytes have moved, so even short transfers surface a
    line. When the total size is known, the message includes percent, ETA, and
    instantaneous throughput; otherwise it reports transferred bytes and rate.
    """

    def __init__(self, verb: str, total_bytes: int | None) -> None:
        self._verb = verb
        self._total = total_bytes if total_bytes and total_bytes > 0 else None
        now = time.monotonic()
        self._start = now
        self._last_log = now
        self._last_bytes = 0

    def update(self, transferred: int) -> None:
        """Consider logging a heartbeat given ``transferred`` total bytes so far."""
        now = time.monotonic()
        if now - self._last_log < _TRANSFER_LOG_INTERVAL_S:
            return
        window_s = now - self._last_log
        window_bytes = transferred - self._last_bytes
        rate = format_throughput(window_bytes, window_s)
        if self._total is not None:
            percent = min(100, round(transferred / self._total * 100))
            remaining = self._total - transferred
            avg_rate = window_bytes / window_s if window_s > 0 else 0
            eta = f"{remaining / avg_rate:.0f}s" if avg_rate > 0 else "n/a"
            logger.info(
                "{} progress: {}/{} ({}%) at {}, ETA {}",
                self._verb,
                format_bytes(transferred),
                format_bytes(self._total),
                percent,
                rate,
                eta,
            )
        else:
            logger.info(
                "{} progress: {} transferred at {}",
                self._verb,
                format_bytes(transferred),
                rate,
            )
        self._last_log = now
        self._last_bytes = transferred
