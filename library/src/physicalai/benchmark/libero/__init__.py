# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LIBERO benchmark for evaluating policies on LIBERO task suites.

Provides `LiberoBenchmark`, a convenience wrapper around `Benchmark` that
auto-creates `LiberoGym` instances for all tasks in a given LIBERO suite.

Available task suites:
    - ``libero_spatial`` — 10 tasks testing spatial reasoning
    - ``libero_object`` — 10 tasks testing object manipulation
    - ``libero_goal`` — 10 tasks testing goal-conditioned behavior
    - ``libero_10`` — 10 mixed long-horizon tasks
    - ``libero_90`` — 90 tasks for large-scale evaluation

Example:
    Run a full LIBERO-10 benchmark and print per-task results:

        >>> from physicalai.benchmark.libero import LiberoBenchmark

        >>> benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
        >>> results = benchmark.evaluate(policy)
        >>> print(results.summary())

    Evaluate only the first three spatial tasks with video recording:

        >>> benchmark = LiberoBenchmark(
        ...     task_suite="libero_spatial",
        ...     task_ids=[0, 1, 2],
        ...     num_episodes=5,
        ...     video_dir="videos/",
        ...     record_mode="failures",
        ... )
        >>> results = benchmark.evaluate(policy)
        >>> results.to_json("spatial_results.json")
"""

from physicalai.benchmark.libero.libero import LiberoBenchmark

__all__ = [
    "LiberoBenchmark",
]
