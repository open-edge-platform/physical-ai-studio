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

    Evaluate exported inference models:

        from getiaction.inference import InferenceModel
        model = InferenceModel.load("./exports/act_policy")
        results = benchmark.evaluate(model)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from getiaction.benchmark.results import BenchmarkResults, TaskResult
from getiaction.eval.rollout import evaluate_policy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from getiaction.eval.video import VideoRecorder
    from getiaction.gyms import Gym
    from getiaction.policies.base import PolicyLike

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
        *,
        show_progress: bool | Literal["auto"] = "auto",
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
        self.show_progress = show_progress

    def evaluate(
        self,
        policy: PolicyLike | Sequence[PolicyLike],
        *,
        continue_on_error: bool = True,
    ) -> BenchmarkResults | dict[str, BenchmarkResults]:
        """Evaluate one or more policies on all benchmark gyms.

        Supports both PyTorch `Policy` objects and exported `InferenceModel`
        objects, enabling benchmarking of production inference performance.

        Args:
            policy: Single policy/model or sequence of policies/models to evaluate.
                Accepts Policy (PyTorch), InferenceModel (exported), or any
                object implementing the PolicyLike protocol (select_action, reset).
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

            Exported inference model:
                >>> from getiaction.inference import InferenceModel
                >>> model = InferenceModel.load("./exports/act_policy")
                >>> results = benchmark.evaluate(model)
        """
        # Normalize to list for uniform processing
        is_single_policy = not isinstance(policy, (list, tuple))
        policies: Sequence[PolicyLike] = (
            [policy] if is_single_policy else policy  # type: ignore[list-item,assignment]
        )

        all_results: dict[str, BenchmarkResults] = {}
        total_policies = len(policies)

        for policy_idx, current_policy in enumerate(policies):
            policy_name = _get_policy_name(current_policy, policy_idx)

            if total_policies > 1:
                logger.info(
                    "Evaluating policy %d/%d: %s",
                    policy_idx + 1,
                    total_policies,
                    policy_name,
                )

            results = self._evaluate_single_policy(
                current_policy,
                policy_name=policy_name,
                show_policy_in_progress=total_policies > 1,
                continue_on_error=continue_on_error,
            )
            all_results[policy_name] = results

        # Return single result or dict based on input type
        if is_single_policy:
            return next(iter(all_results.values()))
        return all_results

    def _evaluate_single_policy(
        self,
        policy: PolicyLike,
        *,
        policy_name: str,
        show_policy_in_progress: bool,
        continue_on_error: bool = True,
    ) -> BenchmarkResults:
        """Evaluate a single policy across all benchmark gyms.

        Args:
            policy: The policy to evaluate.
            policy_name: Name for logging and progress display.
            show_policy_in_progress: Whether to show policy name in progress bar.
            continue_on_error: Whether to continue if a task fails.

        Returns:
            BenchmarkResults for this policy.

        Raises:
            RuntimeError: If all tasks fail during evaluation.
        """
        metadata = self._build_metadata(policy)
        results = BenchmarkResults(metadata=metadata)
        failed_tasks: list[str] = []
        total_gyms = len(self.gyms)

        logger.info(
            "Starting benchmark: %d gyms, %d episodes each",
            total_gyms,
            self.num_episodes,
        )

        start_time = time.time()

        gym_iterator = self._wrap_with_progress(
            enumerate(self.gyms),
            total=total_gyms,
            desc=f"Benchmark ({policy_name})" if show_policy_in_progress else "Benchmark",
        )

        for gym_idx, gym in gym_iterator:
            task_result = self._evaluate_gym(
                gym=gym,
                gym_idx=gym_idx,
                total_gyms=total_gyms,
                policy=policy,
                failed_tasks=failed_tasks,
                continue_on_error=continue_on_error,
            )
            if task_result is not None:
                results.task_results.append(task_result)

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

    def _evaluate_gym(
        self,
        gym: Gym,
        gym_idx: int,
        total_gyms: int,
        policy: PolicyLike,
        failed_tasks: list[str],
        *,
        continue_on_error: bool,
    ) -> TaskResult | None:
        """Evaluate policy on a single gym environment.

        Args:
            gym: The gym environment to evaluate on.
            gym_idx: Index of the gym in the benchmark.
            total_gyms: Total number of gyms (for logging).
            policy: The policy to evaluate.
            failed_tasks: List to append failed task IDs to.
            continue_on_error: Whether to continue if evaluation fails.

        Returns:
            TaskResult if successful, None if failed and continue_on_error=True.
        """
        task_id = _get_task_id(gym, gym_idx)
        task_name = _get_task_name(gym)

        logger.info("Evaluating task %d/%d: %s", gym_idx + 1, total_gyms, task_id)

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
        except Exception:
            logger.exception("Error evaluating task %s", task_id)
            failed_tasks.append(task_id)

            if not continue_on_error:
                raise

            return None

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

        logger.info(
            "  Task %s: success=%.1f%%, reward=%.4f",
            task_id,
            task_result.success_rate,
            task_result.avg_reward,
        )

        return task_result

    def _create_video_recorder(
        self,
        policy: PolicyLike,
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

    def _build_metadata(self, policy: PolicyLike) -> dict[str, Any]:
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

    def _should_show_progress(self) -> bool:
        if self.show_progress == "auto":
            return sys.stderr.isatty()
        return bool(self.show_progress)

    def _wrap_with_progress(
        self,
        iterable: Any,  # noqa: ANN401
        total: int,
        desc: str,
    ) -> Any:  # noqa: ANN401
        if not self._should_show_progress():
            return iterable

        try:
            from tqdm import tqdm  # noqa: PLC0415

            return tqdm(iterable, total=total, desc=desc)
        except ImportError:
            logger.debug("tqdm not installed, progress bar disabled")
            return iterable

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


def _get_policy_name(policy: PolicyLike, _index: int = 0) -> str:
    """Extract policy name for results dict key.

    Handles both Policy objects and InferenceModel objects.

    Returns:
        Policy name string.
    """
    if hasattr(policy, "name") and policy.name:
        return str(policy.name)
    if hasattr(policy, "policy_name") and policy.policy_name:
        # InferenceModel uses policy_name attribute
        return str(policy.policy_name)
    return type(policy).__name__
