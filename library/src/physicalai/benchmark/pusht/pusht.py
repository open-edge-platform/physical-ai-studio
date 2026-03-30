# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PushT benchmark following the Diffusion Policy paper protocol.

Paper: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
       (Chi et al., 2023)

Evaluation protocol:
    - 50 environment initializations per seed
    - Base seed: 100000
    - Report average of last 10 checkpoints (saved every 50 epochs)
    - Averaged across 3 training seeds

Example:
    Single-process:
        >>> benchmark = PushTBenchmark()
        >>> results = benchmark.evaluate(policy)
        >>> print(results.overall_success_rate)

    Vectorized (faster):
        >>> benchmark = PushTBenchmark(num_envs=8)
        >>> results = benchmark.evaluate(policy)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gym_pusht  # noqa: F401

from physicalai.benchmark.benchmark import Benchmark
from physicalai.gyms.pusht import PushTGym

if TYPE_CHECKING:
    from pathlib import Path

    import torch


# Paper protocol constants
BASE_SEED = 100_000
NUM_EPISODES = 50
MAX_STEPS = 300


class PushTBenchmark(Benchmark):
    """PushT benchmark following the Diffusion Policy paper evaluation protocol.

    Defaults match the paper exactly:
    - ``num_episodes=50``  (50 environment initializations)
    - ``seed=100000``      (base seed from the paper's codebase)
    - ``max_steps=300``    (standard PushT episode length)

    Set ``num_envs > 1`` to run environments in parallel worker processes
    (``AsyncVectorEnv``), which significantly reduces wall-clock evaluation time.

    Args:
        num_episodes: Episodes per evaluation (default: 50).
        num_envs: Parallel environment workers. 1 = single process (default: 1).
        seed: Base environment seed (default: 100000, matches paper).
        max_steps: Max steps per episode (default: 300).
        obs_type: Observation type for the underlying gym.
        device: Torch device.
        video_dir: Directory to save episode videos. None disables recording.
        record_mode: "all", "successes", "failures", or "none".
        show_progress: Show tqdm progress bar. "auto" = only when stderr is a tty.

    Example:
        >>> benchmark = PushTBenchmark()
        >>> results = benchmark.evaluate(policy)
        >>> print(f"Success rate: {results.overall_success_rate:.1%}")

        Parallel workers:
        >>> benchmark = PushTBenchmark(num_envs=8)
        >>> results = benchmark.evaluate(policy)
    """

    def __init__(
        self,
        num_episodes: int = NUM_EPISODES,
        num_envs: int = 1,
        seed: int = BASE_SEED,
        max_steps: int = MAX_STEPS,
        obs_type: str = "pixels_agent_pos",
        device: str | torch.device = "cpu",
        video_dir: str | Path | None = None,
        record_mode: str = "failures",
        *,
        show_progress: bool | str = "auto",
    ) -> None:
        """Initialize PushT benchmark with paper-protocol defaults."""
        if num_envs > 1:
            gym = PushTGym.vectorize(
                "gym_pusht/PushT-v0",
                num_envs,
                async_mode=True,
                obs_type=obs_type,
                device=device,
            )
        else:
            gym = PushTGym(obs_type=obs_type, device=device)

        super().__init__(
            gyms=[gym],
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
            video_dir=video_dir,
            record_mode=record_mode,
            show_progress=show_progress,
        )

        self.num_envs = num_envs
        self.obs_type = obs_type
        self.frame_key = "top"

    def _build_metadata(self, policy: Any) -> dict[str, Any]:  # noqa: ANN401
        meta = super()._build_metadata(policy)
        meta.update(
            {
                "seed": BASE_SEED,
                "num_episodes": NUM_EPISODES,
                "num_envs": self.num_envs,
                "obs_type": self.obs_type,
            },
        )
        return meta
