# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark result containers with export capabilities.

This module provides dataclasses for storing and exporting benchmark results
in multiple formats (JSON, CSV).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Results for a single task (gym environment).

    Attributes:
        task_id: Identifier for the task (e.g., "libero_spatial_0").
        task_name: Human-readable task name or description.
        n_episodes: Number of episodes evaluated.
        success_rate: Percentage of successful episodes (0-100).
        avg_reward: Average cumulative reward per episode.
        avg_episode_length: Average number of steps per episode.
        avg_fps: Average frames per second during evaluation.
        per_episode_data: Detailed per-episode results.
    """

    task_id: str
    task_name: str
    n_episodes: int
    success_rate: float
    avg_reward: float
    avg_episode_length: float
    avg_fps: float = 0.0
    per_episode_data: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(
        self,
        *,
        include_per_episode: bool = False,
    ) -> dict[str, Any]:
        """Convert to dictionary representation.

        Args:
            include_per_episode: Whether to include per-episode data.

        Returns:
            Dictionary with task results.
        """
        result = {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "n_episodes": self.n_episodes,
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "avg_episode_length": self.avg_episode_length,
            "avg_fps": self.avg_fps,
        }
        if include_per_episode:
            result["per_episode_data"] = self.per_episode_data
        return result


@dataclass
class BenchmarkResults:
    """Container for benchmark evaluation results across multiple tasks.

    Provides methods for computing aggregate statistics and exporting
    results to various formats.

    Attributes:
        task_results: List of per-task results.
        metadata: Additional metadata (policy name, timestamp, etc.).

    Example:
        >>> results = BenchmarkResults()
        >>> results.task_results.append(task_result)
        >>> print(results.summary())
        >>> results.to_json("results.json")
    """

    task_results: list[TaskResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize metadata with timestamp if not provided."""
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now(UTC).isoformat()

    @property
    def n_tasks(self) -> int:
        """Total number of tasks evaluated."""
        return len(self.task_results)

    @property
    def n_episodes(self) -> int:
        """Total number of episodes across all tasks."""
        return sum(tr.n_episodes for tr in self.task_results)

    @property
    def aggregate_success_rate(self) -> float:
        """Mean success rate across all tasks (0-100)."""
        if not self.task_results:
            return 0.0
        return sum(tr.success_rate for tr in self.task_results) / len(self.task_results)

    # Alias for consistency with design document
    overall_success_rate: ClassVar = aggregate_success_rate

    @property
    def aggregate_reward(self) -> float:
        """Mean reward across all tasks."""
        if not self.task_results:
            return 0.0
        return sum(tr.avg_reward for tr in self.task_results) / len(self.task_results)

    # Alias for consistency
    mean_reward: ClassVar = aggregate_reward

    @property
    def aggregate_episode_length(self) -> float:
        """Mean episode length across all tasks."""
        if not self.task_results:
            return 0.0
        return sum(tr.avg_episode_length for tr in self.task_results) / len(self.task_results)

    @property
    def aggregate_fps(self) -> float:
        """Mean FPS across all tasks."""
        if not self.task_results:
            return 0.0
        return sum(tr.avg_fps for tr in self.task_results) / len(self.task_results)

    _MIN_TASKS_FOR_STD = 2

    @property
    def std_reward(self) -> float:
        """Standard deviation of reward across tasks."""
        if len(self.task_results) < self._MIN_TASKS_FOR_STD:
            return 0.0
        mean = self.aggregate_reward
        variance = sum((tr.avg_reward - mean) ** 2 for tr in self.task_results) / len(self.task_results)
        return variance**0.5

    @property
    def total_episodes(self) -> int:
        """Total number of episodes (alias for n_episodes)."""
        return self.n_episodes

    def summary(self) -> str:
        """Generate human-readable summary of results.

        Returns:
            Formatted string with benchmark summary.
        """
        lines = [
            "=" * 60,
            "BENCHMARK RESULTS SUMMARY",
            "=" * 60,
            f"Tasks evaluated: {self.n_tasks}",
            f"Total episodes: {self.n_episodes}",
            "",
            "AGGREGATE METRICS:",
            f"  Success Rate: {self.aggregate_success_rate:.1f}%",
            f"  Avg Reward: {self.aggregate_reward:.4f}",
            f"  Avg Episode Length: {self.aggregate_episode_length:.1f}",
            f"  Avg FPS: {self.aggregate_fps:.1f}",
            "",
            "PER-TASK RESULTS:",
        ]

        lines.extend(
            f"  {tr.task_id}: success={tr.success_rate:.1f}%, "
            f"reward={tr.avg_reward:.4f}, steps={tr.avg_episode_length:.1f}"
            for tr in self.task_results
        )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(
        self,
        *,
        include_per_episode: bool = False,
    ) -> dict[str, Any]:
        """Convert to dictionary representation.

        Args:
            include_per_episode: Whether to include per-episode data.

        Returns:
            Dictionary with all benchmark results.
        """
        return {
            "metadata": self.metadata,
            "aggregate": {
                "n_tasks": self.n_tasks,
                "n_episodes": self.n_episodes,
                "success_rate": self.aggregate_success_rate,
                "avg_reward": self.aggregate_reward,
                "avg_episode_length": self.aggregate_episode_length,
                "avg_fps": self.aggregate_fps,
            },
            "task_results": [tr.to_dict(include_per_episode=include_per_episode) for tr in self.task_results],
        }

    def to_json(
        self,
        path: str | Path,
        *,
        include_per_episode: bool = True,
        indent: int = 2,
    ) -> Path:
        """Export results to JSON file.

        Args:
            path: Output file path.
            include_per_episode: Whether to include per-episode data.
            indent: JSON indentation level.

        Returns:
            Path to the created JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(
                self.to_dict(include_per_episode=include_per_episode),
                f,
                indent=indent,
            )

        logger.info("Saved benchmark results to %s", path)
        return path

    def to_csv(self, path: str | Path) -> Path:
        """Export per-task results to CSV file.

        Args:
            path: Output file path.

        Returns:
            Path to the created CSV file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "task_id",
            "task_name",
            "n_episodes",
            "success_rate",
            "avg_reward",
            "avg_episode_length",
            "avg_fps",
        ]

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for tr in self.task_results:
                writer.writerow(tr.to_dict(include_per_episode=False))

        logger.info("Saved benchmark results CSV to %s", path)
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> BenchmarkResults:
        """Load results from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            BenchmarkResults instance.
        """
        path = Path(path)
        with path.open() as f:
            data = json.load(f)

        results = cls(metadata=data.get("metadata", {}))
        for tr_data in data.get("task_results", []):
            results.task_results.append(
                TaskResult(
                    task_id=tr_data["task_id"],
                    task_name=tr_data["task_name"],
                    n_episodes=tr_data["n_episodes"],
                    success_rate=tr_data["success_rate"],
                    avg_reward=tr_data["avg_reward"],
                    avg_episode_length=tr_data["avg_episode_length"],
                    avg_fps=tr_data.get("avg_fps", 0.0),
                    per_episode_data=tr_data.get("per_episode_data", []),
                ),
            )

        return results


__all__ = ["BenchmarkResults", "TaskResult"]
