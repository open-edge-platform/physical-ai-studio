# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa benchmark for evaluating policies on RoboCasa task groups.

Provides `RoboCasaBenchmark`, a convenience wrapper around `Benchmark` that
auto-creates `RoboCasaGym` instances for all tasks in a given RoboCasa group.

Available task group keywords:
    - ``atomic_seen`` — 18 atomic tasks, target split
    - ``composite_seen`` — composite tasks, target split
    - ``composite_unseen`` — composite tasks, target split
    - ``pretrain50`` — 50-task pretrain split
    - ``pretrain100`` — 100-task pretrain split
    - ``pretrain200`` — 200-task pretrain split
    - ``pretrain300`` — 300-task pretrain split

Example:
    Run a full atomic_seen benchmark and print per-task results:

        >>> from physicalai.benchmark.gyms.robocasa import RoboCasaBenchmark

        >>> benchmark = RoboCasaBenchmark(task="atomic_seen", num_episodes=20)
        >>> results = benchmark.evaluate(policy)
        >>> print(results.summary())

    Evaluate two specific tasks with video recording:

        >>> benchmark = RoboCasaBenchmark(
        ...     task="CloseFridge,OpenDrawer",
        ...     num_episodes=5,
        ...     video_dir="videos/",
        ...     record_mode="failures",
        ... )
        >>> results = benchmark.evaluate(policy)
        >>> results.to_json("robocasa_results.json")
"""

from physicalai.benchmark.gyms.robocasa.robocasa import RoboCasaBenchmark
from physicalai.gyms import RoboCasaSplit, RoboCasaTaskGroup

__all__ = ["RoboCasaBenchmark", "RoboCasaSplit", "RoboCasaTaskGroup"]
