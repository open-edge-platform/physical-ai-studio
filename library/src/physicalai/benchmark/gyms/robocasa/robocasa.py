# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa benchmark - specialized benchmark for RoboCasa task groups.

This module provides `RoboCasaBenchmark`, a convenience class that auto-creates
gyms for RoboCasa task groups with sensible defaults.

Example:
    >>> benchmark = RoboCasaBenchmark(task="atomic_seen", num_episodes=20)
    >>> results = benchmark.evaluate(policy)
    >>> print(results.summary())

    # Compare multiple policies
    >>> results = {p.name: benchmark.evaluate(p) for p in [act, pi0, groot]}
    >>> for name, r in results.items():
    ...     print(f"{name}: {r.overall_success_rate:.1%}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.benchmark.gyms.benchmark import Benchmark

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.gyms.robocasa import FieldOrder


class RoboCasaBenchmark(Benchmark):
    """Specialized benchmark for RoboCasa task groups.

    Auto-creates `RoboCasaGym` instances for all tasks in the specified group.
    Provides sensible defaults for RoboCasa evaluation.

    Args:
        task: RoboCasa task group keyword, single task name, or
            comma-separated task names. Group keywords:
            - "atomic_seen" (18 atomic tasks, target split)
            - "composite_seen" (composite tasks, target split)
            - "composite_unseen" (composite tasks, target split)
            - "pretrain50" / "pretrain100" / "pretrain200" / "pretrain300"
              (pretrain splits of increasing size)
        num_episodes: Number of episodes per task (default: 20).
        max_steps: Maximum steps per episode. When ``None`` (default), each
            task uses its own official horizon from robocasa's dataset_registry
            (horizons vary per task, roughly 200-4800 steps). Pass an explicit
            value to apply the same cap to every task instead.
        seed: Random seed for reproducibility (default: 42).
        observation_height: Height of observation images (default: 256).
        observation_width: Width of observation images (default: 256).
        video_dir: Directory to save videos. None disables recording.
        record_mode: Video recording mode - "all", "successes", "failures", "none".
        split: RoboCasa dataset split override (``None``/``"all"``/
            ``"pretrain"``/``"target"``). Only meaningful when ``task`` is an
            explicit task name rather than a group keyword -- group keywords
            already imply their natural split (e.g. ``"atomic_seen"`` implies
            ``"target"``) unless overridden here.
        state_order: Forwarded to each ``RoboCasaGym``. Ordered
            ``(name, dim)`` schema for the flat ``agent_pos`` vector;
            defaults to the native PandaOmron order. Pass a
            checkpoint-derived order to match a policy trained with a
            different field order.
        action_order: Forwarded to each ``RoboCasaGym``, analogous to
            `state_order` but for the flat action vector.

    Example:
        >>> # Full atomic_seen benchmark
        >>> benchmark = RoboCasaBenchmark(task="atomic_seen", num_episodes=20)
        >>> results = benchmark.evaluate(policy)

        >>> # Quick test on specific tasks
        >>> benchmark = RoboCasaBenchmark(
        ...     task="CloseFridge,OpenDrawer",
        ...     num_episodes=5,
        ... )
        >>> results = benchmark.evaluate(policy)
    """

    def __init__(
        self,
        task: str = "atomic_seen",
        num_episodes: int = 20,
        max_steps: int | None = None,
        seed: int = 42,
        observation_height: int = 256,
        observation_width: int = 256,
        video_dir: str | Path | None = None,
        record_mode: str = "failures",
        split: str | None = None,
        state_order: FieldOrder | None = None,
        action_order: FieldOrder | None = None,
    ) -> None:
        """Initialize RoboCasa benchmark with task group configuration."""
        self.task = task
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.split = split
        self.state_order = state_order
        self.action_order = action_order

        # Create gyms for the task group
        gyms = self._create_gyms()

        super().__init__(
            gyms=gyms,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
            video_dir=video_dir,
            record_mode=record_mode,
        )

        # RoboCasa keys images by raw camera name, not the base "image" default.
        # Stack all 3 default views (left, right, wrist) horizontally in recorded videos.
        self.frame_key = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

    def _create_gyms(self) -> list:
        """Create RoboCasaGym instances for the task group.

        Bridges this class's human-friendly ``str`` ``task``/``split`` to the
        typed ``RoboCasaTaskGroup``/``RoboCasaSplit`` API that
        ``create_robocasa_gyms`` requires. Sets ``task_id`` and
        ``task_suite_name`` on each gym so the base class
        ``_get_task_id``/``_get_task_name`` protocol can build per-task
        result keys.

        Returns:
            List of RoboCasaGym instances.

        Raises:
            ValueError: If ``self.task`` is empty.
        """
        from physicalai.gyms import RoboCasaSplit, RoboCasaTaskGroup, create_robocasa_gyms  # noqa: PLC0415

        tasks: RoboCasaTaskGroup | list[str]
        try:
            tasks = RoboCasaTaskGroup(self.task)
        except ValueError:
            tasks = [t.strip() for t in self.task.split(",") if t.strip()]
            if not tasks:
                msg = "`task` must contain at least one RoboCasa task name."
                raise ValueError(msg) from None

        gyms = create_robocasa_gyms(
            tasks=tasks,
            observation_height=self.observation_height,
            observation_width=self.observation_width,
            split=RoboCasaSplit(self.split) if self.split is not None else None,
            state_order=self.state_order,
            action_order=self.action_order,
        )
        for gym in gyms:
            gym.task_id = gym.task  # type: ignore[attr-defined]
            gym.task_suite_name = self.task  # type: ignore[attr-defined]
        return gyms

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RoboCasaBenchmark(task={self.task!r}, num_episodes={self.num_episodes}, max_steps={self.max_steps})"
