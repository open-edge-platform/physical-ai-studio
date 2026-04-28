# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Throughput monitor callback for inference performance tracking.

Tracks predictions per second over a time-based sliding window.
Useful for monitoring sustained throughput in deployment and
detecting degradation.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, override

from physicalai.inference.callbacks.base import Callback

_MIN_SAMPLES_FOR_THROUGHPUT = 2


class ThroughputMonitor(Callback):
    """Track inference throughput (predictions/sec) over a time-based window.

    Timestamps older than *window_seconds* are pruned on each
    prediction, so throughput always reflects a recent time span
    rather than a fixed sample count.

    Args:
        window_seconds: Duration of the sliding window in seconds.
            Defaults to 10.0.

    Attributes:
        throughput: Current throughput in predictions per second,
            computed over timestamps within the window.
        total_calls: Total number of predictions recorded.

    Examples:
        >>> monitor = ThroughputMonitor(window_seconds=5.0)
        >>> model = InferenceModel.load("./exports/act", callbacks=[monitor])
        >>> for obs in observations:
        ...     model.select_action(obs)
        >>> print(f"{monitor.throughput:.1f} predictions/sec")
    """

    def __init__(self, window_seconds: float = 10.0) -> None:
        """Initialise the throughput monitor.

        Args:
            window_seconds: Duration of the sliding window in seconds.
                Defaults to 10.0.
        """
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self.total_calls: int = 0
        self.throughput: float = 0.0

    @override
    def on_reset(self) -> None:
        """Clear all timestamps and reset throughput to zero."""
        self._timestamps.clear()
        self.total_calls = 0
        self.throughput = 0.0

    def _prune_window(self, now: float) -> None:
        """Remove timestamps older than the window boundary."""
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    @override
    def on_predict_end(self, outputs: dict[str, Any]) -> None:
        """Record prediction timestamp and recompute throughput."""
        now = time.perf_counter()
        self._timestamps.append(now)
        self.total_calls += 1

        self._prune_window(now)

        if len(self._timestamps) >= _MIN_SAMPLES_FOR_THROUGHPUT:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self.throughput = (len(self._timestamps) - 1) / elapsed
            else:
                self.throughput = 0.0
        else:
            self.throughput = 0.0

    @override
    def __repr__(self) -> str:
        return (
            f"ThroughputMonitor("
            f"throughput={self.throughput:.1f}/s, "
            f"total={self.total_calls}, "
            f"window={self._window_seconds}s)"
        )
