# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark class for evaluating policies across multiple gym environments.

This module provides the `Benchmark` class - a concrete, directly usable class
for evaluating policies.

Example:
    Direct usage with explicit gyms:

        benchmark = Benchmark(
            gyms=[LiberoGym(task_id=i) for i in range(10)],
            num_episodes=20,
            max_steps=300,
        )
        results = benchmark.evaluate(policy)

    Specialized benchmark:

        benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
        results = benchmark.evaluate(policy)

    Multi-policy comparison:

        results = benchmark.evaluate([act, pi0, groot])
        for name, result in results.items():
            print(f"{name}: {result.overall_success_rate:.1%}")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from getiaction.benchmark.results import BenchmarkResults, TaskResult
from getiaction.eval.rollout import evaluate_policy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from getiaction.eval.video import VideoRecorder
    from getiaction.gyms import Gym
    from getiaction.policies.base import Policy

logger = logging.getLogger(__name__)


class Benchmark:
    """Concrete class for evaluating policies across multiple gym environments.

    `Benchmark` orchestrates evaluation across multiple gym environments,
    runs multiple episodes per gym, aggregates results, and optionally
    records videos of episodes.

    This class follows the same pattern as other geti-action core classes:
    - Direct usage: `Benchmark(gyms=[...], num_episodes=20)`
    - Specialized subclass: `LiberoBenchmark(task_suite="libero_10")`

    Args:
        gyms: List of gym environments to evaluate on.
        num_episodes: Number of episodes per gym (default: 20).
        max_steps: Maximum steps per episode. None uses gym default.
        seed: Random seed for reproducibility (default: 42).
        video_dir: Directory to save videos. None disables recording.
        record_mode: Video recording mode - "all", "successes", "failures", "none".

    Example:
        >>> from getiaction.benchmark import Benchmark
        >>> from getiaction.gyms import LiberoGym

        >>> gyms = [LiberoGym(task_suite="libero_10", task_id=i) for i in range(10)]
        >>> benchmark = Benchmark(gyms=gyms, num_episodes=20, max_steps=300)
        >>> results = benchmark.evaluate(policy)
        >>> print(results.summary())
    """

    def __init__(
        self,
        gyms: list[Gym],
        num_episodes: int = 20,
        max_steps: int | None = None,
        seed: int = 42,
        video_dir: str | Path | None = None,
        record_mode: str = "failures",
    ) -> None:
        """Initialize benchmark with gyms and evaluation parameters.

        Raises:
            ValueError: If gyms list is empty.
        """
        if not gyms:
            msg = "At least one gym is required"
            raise ValueError(msg)

        self.gyms = gyms
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.seed = seed
        self.video_dir = Path(video_dir) if video_dir else None
        self.record_mode = record_mode

    def evaluate(
        self,
        policy: Policy | Sequence[Policy],
        *,
        continue_on_error: bool = True,
    ) -> BenchmarkResults | dict[str, BenchmarkResults]:
        """Evaluate one or more policies on all benchmark gyms.

        Args:
            policy: Single policy or sequence of policies to evaluate.
            continue_on_error: Whether to continue if a task fails.

        Returns:
            If single policy: BenchmarkResults for that policy.
            If multiple policies: Dict mapping policy name to BenchmarkResults.

        Example:
            Single policy:
                >>> results = benchmark.evaluate(my_policy)
                >>> print(results.overall_success_rate)

            Multiple policies:
                >>> results = benchmark.evaluate([act, pi0, groot])
                >>> for name, r in results.items():
                ...     print(f"{name}: {r.overall_success_rate:.1%}")
        """
        # Handle multi-policy evaluation
        if isinstance(policy, (list, tuple)):
            return self._evaluate_multiple(
                policies=policy,  # type: ignore[arg-type]
                continue_on_error=continue_on_error,
            )

        return self._evaluate_single(
            policy=policy,  # type: ignore[arg-type]
            continue_on_error=continue_on_error,
        )

    def _evaluate_single(
        self,
        policy: Policy,
        *,
        continue_on_error: bool = True,
    ) -> BenchmarkResults:
        """Evaluate a single policy on all gyms.

        Returns:
            BenchmarkResults containing evaluation metrics.

        Raises:
            RuntimeError: If all tasks fail during evaluation.
        """
        # Build metadata
        metadata = self._build_metadata(policy)

        # Initialize results
        results = BenchmarkResults(metadata=metadata)

        total_gyms = len(self.gyms)
        failed_tasks: list[str] = []

        logger.info(
            "Starting benchmark: %d gyms, %d episodes each",
            total_gyms,
            self.num_episodes,
        )

        start_time = time.time()

        for gym_idx, gym in enumerate(self.gyms):
            task_id = _get_task_id(gym, gym_idx)
            task_name = _get_task_name(gym)

            logger.info("Evaluating task %d/%d: %s", gym_idx + 1, total_gyms, task_id)

            # Create video recorder for this task if enabled
            video_recorder = self._create_video_recorder(policy, task_id)

            try:
                eval_result = evaluate_policy(
                    env=gym,
                    policy=policy,
                    n_episodes=self.num_episodes,
                    start_seed=self.seed,
                    max_steps=self.max_steps,
                    video_recorder=video_recorder,
                )

                aggregated = eval_result["aggregated"]
                per_episode = eval_result.get("per_episode", [])

                task_result = TaskResult(
                    task_id=task_id,
                    task_name=task_name,
                    n_episodes=self.num_episodes,
                    success_rate=aggregated.get("pc_success", 0.0),
                    avg_reward=aggregated["avg_sum_reward"],
                    avg_episode_length=aggregated["avg_episode_length"],
                    avg_fps=aggregated.get("avg_fps", 0.0),
                    per_episode_data=per_episode,
                )

                results.task_results.append(task_result)

                logger.info(
                    "  Task %s: success=%.1f%%, reward=%.4f",
                    task_id,
                    task_result.success_rate,
                    task_result.avg_reward,
                )

            except Exception:
                logger.exception("Error evaluating task %s", task_id)
                failed_tasks.append(task_id)

                if not continue_on_error:
                    raise

        # Record timing
        elapsed = time.time() - start_time
        results.metadata["elapsed_seconds"] = elapsed
        results.metadata["failed_tasks"] = failed_tasks

        if failed_tasks:
            logger.warning("Failed tasks: %s", failed_tasks)

        if not results.task_results:
            msg = "All tasks failed during evaluation"
            raise RuntimeError(msg)

        logger.info(
            "Benchmark complete: %.1f%% success rate, %.1f seconds",
            results.overall_success_rate,
            elapsed,
        )

        return results

    def _evaluate_multiple(
        self,
        policies: Sequence[Policy],
        *,
        continue_on_error: bool = True,
    ) -> dict[str, BenchmarkResults]:
        """Evaluate multiple policies and return results dict.

        Returns:
            Dict mapping policy names to their BenchmarkResults.
        """
        all_results: dict[str, BenchmarkResults] = {}
        total_policies = len(policies)

        for policy_idx, policy in enumerate(policies):
            policy_name = _get_policy_name(policy, policy_idx)
            logger.info(
                "Evaluating policy %d/%d: %s",
                policy_idx + 1,
                total_policies,
                policy_name,
            )

            results = self._evaluate_single(
                policy=policy,
                continue_on_error=continue_on_error,
            )

            all_results[policy_name] = results

        return all_results

    def _create_video_recorder(
        self,
        policy: Policy,
        task_id: str,
    ) -> VideoRecorder | None:
        """Create a VideoRecorder for the current task if video recording is enabled.

        Args:
            policy: Policy being evaluated (used for naming).
            task_id: Task identifier (used for naming).

        Returns:
            VideoRecorder instance or None if recording is disabled.
        """
        if not self.video_dir or self.record_mode == "none":
            return None

        from getiaction.eval.video import VideoRecorder  # noqa: PLC0415

        policy_name = _get_policy_name(policy, 0)
        video_path = self.video_dir / policy_name / task_id

        return VideoRecorder(
            output_dir=video_path,
            fps=30,
            record_mode=self.record_mode,  # type: ignore[arg-type]
        )

    def _build_metadata(self, policy: Policy) -> dict[str, Any]:
        """Build metadata dict for results.

        Returns:
            Dict with benchmark and policy metadata.
        """
        return {
            "benchmark_class": type(self).__name__,
            "policy_class": type(policy).__name__,
            "num_episodes": self.num_episodes,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "num_gyms": len(self.gyms),
            "video_dir": str(self.video_dir) if self.video_dir else None,
            "record_mode": self.record_mode,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{type(self).__name__}("
            f"gyms={len(self.gyms)}, "
            f"num_episodes={self.num_episodes}, "
            f"max_steps={self.max_steps}, "
            f"seed={self.seed})"
        )


# Helper functions
def _get_task_id(gym: Gym, index: int) -> str:
    """Extract task ID from gym or use index.

    Returns:
        Task identifier string.
    """
    if hasattr(gym, "task_suite_name") and hasattr(gym, "task_id"):
        return f"{gym.task_suite_name}_{gym.task_id}"
    if hasattr(gym, "task_id"):
        return str(gym.task_id)
    return f"task_{index}"


def _get_task_name(gym: Gym) -> str:
    """Extract task name from gym.

    Returns:
        Task name string or empty string.
    """
    if hasattr(gym, "task_name"):
        return gym.task_name
    if hasattr(gym, "task_description"):
        return gym.task_description
    return ""


def _get_policy_name(policy: Policy, _index: int = 0) -> str:
    """Extract policy name for results dict key.

    Returns:
        Policy name string.
    """
    if hasattr(policy, "name") and policy.name:
        return str(policy.name)
    return type(policy).__name__
