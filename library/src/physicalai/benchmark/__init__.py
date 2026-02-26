# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark classes for evaluating policies across multiple environments.

This module provides the `Benchmark` class - a concrete, directly usable class
for evaluating policies.

Examples:
    Direct usage with explicit gyms:

        >>> from physicalai.benchmark import Benchmark
        >>> from physicalai.gyms import LiberoGym

        >>> gyms = [LiberoGym(task_id=i) for i in range(10)]
        >>> benchmark = Benchmark(gyms=gyms, num_episodes=20, max_steps=300)
        >>> results = benchmark.evaluate(policy)
        >>> print(results.summary())

    Specialized LIBERO benchmark:

        >>> from physicalai.benchmark import LiberoBenchmark

        >>> benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
        >>> results = benchmark.evaluate(policy)
        >>> results.to_json("libero_10_results.json")

    Multi-policy comparison:

        >>> results = benchmark.evaluate([act, pi0, groot])
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.overall_success_rate:.1%}")
"""

from physicalai.benchmark.benchmark import Benchmark
from physicalai.benchmark.libero import LiberoBenchmark
from physicalai.benchmark.results import BenchmarkResults, TaskResult
from physicalai.eval.video import RecordMode, VideoRecorder

__all__ = [
    "Benchmark",
    "BenchmarkResults",
    "LiberoBenchmark",
    "RecordMode",
    "TaskResult",
    "VideoRecorder",
]
