# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LIBERO benchmark - specialized benchmark for LIBERO task suites.

This module provides `LiberoBenchmark`, a convenience class that auto-creates
gyms for LIBERO task suites with sensible defaults.

Example:
    >>> benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
    >>> results = benchmark.evaluate(policy)
    >>> print(results.summary())

    # Compare multiple policies
    >>> results = {p.name: benchmark.evaluate(p) for p in [act, pi0, groot]}
    >>> for name, r in results.items():
    ...     print(f"{name}: {r.overall_success_rate:.1%}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.benchmark.benchmark import Benchmark

if TYPE_CHECKING:
    from pathlib import Path


class LiberoBenchmark(Benchmark):
    """Specialized benchmark for LIBERO task suites.

    Auto-creates `LiberoGym` instances for all tasks in the specified suite.
    Provides sensible defaults for LIBERO evaluation.

    Args:
        task_suite: LIBERO task suite name. Options:
            - "libero_spatial" (10 tasks)
            - "libero_object" (10 tasks)
            - "libero_goal" (10 tasks)
            - "libero_10" (10 tasks, mixed)
            - "libero_90" (90 tasks, long-horizon)
        task_ids: Specific task IDs to evaluate. None means all tasks.
        num_episodes: Number of episodes per task (default: 20).
        max_steps: Maximum steps per episode (default: 300).
        seed: Random seed for reproducibility (default: 42).
        observation_height: Height of observation images (default: 256).
        observation_width: Width of observation images (default: 256).
        video_dir: Directory to save videos. None disables recording.
        record_mode: Video recording mode - "all", "successes", "failures", "none".

    Example:
        >>> # Full LIBERO-10 benchmark
        >>> benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
        >>> results = benchmark.evaluate(policy)

        >>> # Quick test on specific tasks
        >>> benchmark = LiberoBenchmark(
        ...     task_suite="libero_spatial",
        ...     task_ids=[0, 1, 2],
        ...     num_episodes=5,
        ... )
        >>> results = benchmark.evaluate(policy)
    """

    # Default max steps for LIBERO tasks
    DEFAULT_MAX_STEPS = 300

    def __init__(
        self,
        task_suite: str = "libero_10",
        task_ids: list[int] | None = None,
        num_episodes: int = 20,
        max_steps: int | None = None,
        seed: int = 42,
        observation_height: int = 256,
        observation_width: int = 256,
        video_dir: str | Path | None = None,
        record_mode: str = "failures",
    ) -> None:
        """Initialize LIBERO benchmark with task suite configuration."""
        self.task_suite = task_suite
        self.task_ids = task_ids
        self.observation_height = observation_height
        self.observation_width = observation_width

        # Use LIBERO default max_steps if not specified
        if max_steps is None:
            max_steps = self.DEFAULT_MAX_STEPS

        # Create gyms for the task suite
        gyms = self._create_gyms()

        super().__init__(
            gyms=gyms,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
            video_dir=video_dir,
            record_mode=record_mode,
        )

    def _create_gyms(self) -> list:
        """Create LiberoGym instances for the task suite.

        Returns:
            List of LiberoGym instances.
        """
        from physicalai.gyms import create_libero_gyms  # noqa: PLC0415

        return create_libero_gyms(
            task_suites=self.task_suite,
            task_ids=self.task_ids,
            observation_height=self.observation_height,
            observation_width=self.observation_width,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        task_info = f"task_ids={self.task_ids}" if self.task_ids else "all tasks"
        return (
            f"LiberoBenchmark("
            f"task_suite={self.task_suite!r}, "
            f"{task_info}, "
            f"num_episodes={self.num_episodes}, "
            f"max_steps={self.max_steps})"
        )
