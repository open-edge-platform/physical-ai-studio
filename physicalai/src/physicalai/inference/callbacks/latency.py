# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Latency monitor callback for inference performance profiling.

Records wall-clock latency of each ``select_action`` / ``__call__``
invocation over a sliding window and exposes summary statistics
(mean, min, max, p95).  Useful for benchmarking, detecting latency
regressions, and monitoring production deployments.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, override

from physicalai.inference.callbacks.base import Callback

_DEFAULT_WINDOW_SIZE = 100


class LatencyMonitor(Callback):
    """Record prediction latency in milliseconds over a sliding window.

    The window retains the most recent *window_size* measurements and
    exposes lightweight summary statistics as properties.  Older
    samples are automatically discarded.

    Args:
        window_size: Number of recent predictions to keep.
            Defaults to 100.

    Attributes:
        latest_ms: Duration of the most recent prediction (ms).
        total_calls: Total number of predictions recorded since
            creation (not bounded by window).

    Examples:
        >>> latency = LatencyMonitor(window_size=200)
        >>> model = InferenceModel.load("./exports/act", callbacks=[latency])
        >>> for obs in observations:
        ...     action = model.select_action(obs)
        >>> print(
        ...     f"p95={latency.p95_ms:.1f}ms  avg={latency.avg_ms:.1f}ms  "
        ...     f"min={latency.min_ms:.1f}ms  max={latency.max_ms:.1f}ms"
        ... )
    """

    def __init__(self, window_size: int = _DEFAULT_WINDOW_SIZE) -> None:
        """Initialise timing state with a bounded sliding window.

        Args:
            window_size: Maximum number of latency samples to retain.
        """
        self._window: deque[float] = deque(maxlen=window_size)
        self.latest_ms: float = 0.0
        self.total_calls: int = 0
        self._start_time: float = 0.0

    @override
    def on_reset(self) -> None:
        """Clear all recorded samples and reset counters."""
        self._window.clear()
        self.latest_ms = 0.0
        self.total_calls = 0
        self._start_time = 0.0

    @override
    def on_predict_start(self, inputs: dict[str, Any]) -> None:
        """Record the prediction start time."""
        self._start_time = time.perf_counter()

    @override
    def on_predict_end(self, outputs: dict[str, Any]) -> None:
        """Compute and store the prediction duration."""
        ms_per_second = 1000.0
        self.latest_ms = (time.perf_counter() - self._start_time) * ms_per_second
        self._window.append(self.latest_ms)
        self.total_calls += 1

    @property
    def avg_ms(self) -> float:
        """Mean latency over the current window (ms).

        Returns ``0.0`` when no samples have been recorded.
        """
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def min_ms(self) -> float:
        """Minimum latency over the current window (ms).

        Returns ``0.0`` when no samples have been recorded.
        """
        if not self._window:
            return 0.0
        return min(self._window)

    @property
    def max_ms(self) -> float:
        """Maximum latency over the current window (ms).

        Returns ``0.0`` when no samples have been recorded.
        """
        if not self._window:
            return 0.0
        return max(self._window)

    @property
    def p95_ms(self) -> float:
        """95th-percentile latency over the current window (ms).

        Uses the *nearest-rank* method.  Returns ``0.0`` when no
        samples have been recorded.
        """
        if not self._window:
            return 0.0
        sorted_window = sorted(self._window)
        # Nearest-rank: index = ceil(p * n) - 1
        rank_index = math.ceil(0.95 * len(sorted_window)) - 1
        return sorted_window[rank_index]

    @override
    def __repr__(self) -> str:
        """Return string representation with latest timing and stats."""
        return (
            f"LatencyMonitor("
            f"latest={self.latest_ms:.1f}ms, "
            f"avg={self.avg_ms:.1f}ms, "
            f"p95={self.p95_ms:.1f}ms, "
            f"calls={self.total_calls})"
        )
