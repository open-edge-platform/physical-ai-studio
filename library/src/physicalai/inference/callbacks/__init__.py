# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference callbacks for cross-cutting concerns.

Callbacks hook into the inference lifecycle to provide timing,
throughput monitoring, safety checks, and other instrumentation
without modifying model or runner code.
"""

from physicalai.inference.callbacks.base import Callback
from physicalai.inference.callbacks.latency import LatencyMonitor
from physicalai.inference.callbacks.throughput import ThroughputMonitor

__all__ = [
    "Callback",
    "LatencyMonitor",
    "ThroughputMonitor",
]
