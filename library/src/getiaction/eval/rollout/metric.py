# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TorchMetrics-based rollout evaluation metrics for gym environments.

This module provides a TorchMetrics.Metric implementation for policy evaluation
that supports distributed training, automatic state synchronization, and
seamless integration with PyTorch Lightning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torchmetrics import Metric

from .functional import rollout

if TYPE_CHECKING:
    from getiaction.eval.video import VideoRecorder
    from getiaction.gyms import Gym
    from getiaction.policies.base import Policy


class Rollout(Metric):
    """TorchMetrics-based rollout evaluation metric for gym environments.

    This metric computes rollout statistics across multiple episodes with
    automatic distributed synchronization for multi-GPU training. It wraps
    the existing rollout functionality while providing torchmetrics benefits:
    - Automatic state synchronization across distributed processes
    - Integration with Lightning's metric logging system
    - Efficient tensor-based accumulation
    - Consistent API with other torchmetrics

    The metric tracks per-episode statistics and computes aggregated metrics
    across all episodes when compute() is called.

    Args:
        max_steps: Maximum steps per episode. If None, uses environment default.
        dist_sync_on_step: Whether to sync state across processes on each update.
            Default is False (sync only when compute() is called).
        **kwargs: Additional arguments passed to torchmetrics.Metric.

    Attributes:
        sum_rewards: Accumulated sum of all episode rewards
        max_rewards: Accumulated sum of max rewards per episode
        episode_lengths: Accumulated sum of episode lengths
        num_successes: Total number of successful episodes
        num_episodes: Total number of episodes evaluated
        all_sum_rewards: List of individual episode sum rewards
        all_max_rewards: List of individual episode max rewards
        all_episode_lengths: List of individual episode lengths
        all_successes: List of success flags for each episode

    Example:
        >>> from getiaction.eval.rollout import Rollout
        >>> metric = Rollout()
        >>> # During validation
        >>> for gym_env in val_dataloader:
        ...     metric.update(gym_env, policy, seed=0)
        >>> results = metric.compute()
        >>> print(f"Success rate: {results['pc_success']:.1f}%")
        >>> metric.reset()  # Prepare for next epoch
    """

    full_state_update: bool | None = False

    # Threshold for considering a success value as True (stored as float)
    _SUCCESS_THRESHOLD = 0.5

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        video_recorder: VideoRecorder | None = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the Rollout.

        Args:
            max_steps: Maximum steps per episode rollout.
            video_recorder: Optional video recorder for capturing rollout videos.
                Can also be set later via the video_recorder attribute.
            dist_sync_on_step: Whether to sync distributed state on each update.
            **kwargs: Additional torchmetrics.Metric arguments.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)

        self.max_steps = max_steps
        self.video_recorder = video_recorder  # Can be set/changed later

        # Aggregate states (synced via sum)
        self.add_state("sum_rewards", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("max_rewards", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("episode_lengths", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_episodes", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_successes", default=torch.tensor(0), dist_reduce_fx="sum")

        # Per-episode tracking (concatenated)
        self.add_state("all_sum_rewards", default=[], dist_reduce_fx="cat")
        self.add_state("all_max_rewards", default=[], dist_reduce_fx="cat")
        self.add_state("all_episode_lengths", default=[], dist_reduce_fx="cat")
        self.add_state("all_successes", default=[], dist_reduce_fx="cat")

        # Episode counter for naming videos
        self._episode_counter = 0

    def update(self, env: Gym, policy: Policy, seed: int | None = None) -> dict[str, Tensor]:
        """Run a rollout and update metric state.

        This method executes one complete episode in the gym environment using
        the provided policy, then updates the internal state with the results.
        The state is automatically synchronized across distributed processes
        when compute() is called.

        If video_recorder is set, recording happens DURING the rollout loop.

        Args:
            env: Gym environment instance to evaluate in.
            policy: Policy to use for action selection.
            seed: Optional seed for reproducible rollouts.

        Returns:
            Dictionary with this episode's metrics:
                - sum_reward: Total reward accumulated in the episode
                - max_reward: Maximum single-step reward in the episode
                - episode_length: Number of steps taken
                - success: Whether the episode succeeded (if available)
        """
        # Start video recording if enabled
        if self.video_recorder is not None:
            episode_name = f"rollout_{self._episode_counter:03d}"
            self.video_recorder.start_episode(episode_name)

        result = rollout(
            env=env,
            policy=policy,
            seed=seed,
            max_steps=self.max_steps,
            return_observations=False,
            video_recorder=self.video_recorder,
        )

        # Finish video recording
        success_tensor = result.get("success")
        success_bool = bool(success_tensor[0].item()) if success_tensor is not None else False
        if self.video_recorder is not None:
            self.video_recorder.finish_episode(success=success_bool)

        self._episode_counter += 1

        # (B,) tensors
        sum_reward = result["sum_reward"]  # (batch_size,)
        max_reward = result["max_reward"]  # (batch_size,)
        batch_size = sum_reward.shape[0]
        episode_length = torch.tensor(result["episode_length"], dtype=torch.float32, device=self.device)

        # Aggregate (sum over batch)
        # Promote to this device
        sum_reward = sum_reward.to(self.device)
        max_reward = max_reward.to(self.device)

        self.sum_rewards += sum_reward.sum()  # type: ignore[has-type]
        self.max_rewards += max_reward.sum()  # type: ignore[has-type]
        self.episode_lengths += episode_length * batch_size  # type: ignore[has-type]
        self.num_episodes += batch_size  # type: ignore[has-type]

        # Track success if available
        if success_tensor is not None:
            success_tensor = success_tensor.to(self.device)
            self.num_successes += success_tensor.sum().int()  # type: ignore[has-type]
            self.all_successes.extend(success_tensor.float())  # type: ignore[union-attr]

        # Store individual per-episode data
        self.all_sum_rewards.extend(sum_reward)  # type: ignore[union-attr]
        self.all_max_rewards.extend(max_reward)  # type: ignore[union-attr]
        self.all_episode_lengths.extend([episode_length] * batch_size)  # type: ignore[union-attr]

        episode_result = {
            "sum_reward": sum_reward.mean(),  # averaged just for convenience
            "max_reward": max_reward.mean(),
            "episode_length": episode_length,
        }
        if success_tensor is not None:
            episode_result["success"] = success_tensor.float().mean()

        return episode_result

    # averages across all episodes
    def compute(self) -> dict[str, Tensor]:
        """Compute aggregated metrics across all episodes.

        This method is called at the end of an epoch (or when explicitly requested)
        to compute final aggregated statistics. In distributed settings, the state
        is automatically synchronized across all processes before computation.

        The returned metrics match the format of evaluate_policy() from rollout.py
        to ensure numerical equivalence with the previous implementation.

        Returns:
            Dictionary of aggregated metrics:
                - avg_sum_reward: Mean total reward per episode
                - avg_max_reward: Mean maximum reward per episode
                - pc_success: Success rate as percentage (0-100)
                - avg_episode_length: Mean episode length
                - n_episodes: Total number of episodes evaluated

        Note:
            Returns zero values if no episodes have been evaluated yet.
        """
        if self.num_episodes == 0:
            return {
                "avg_sum_reward": torch.tensor(0.0, device=self.device),
                "avg_max_reward": torch.tensor(0.0, device=self.device),
                "pc_success": torch.tensor(0.0, device=self.device),
                "avg_episode_length": torch.tensor(0.0, device=self.device),
                "n_episodes": torch.tensor(0, device=self.device),
            }

        avg_sum_reward = self.sum_rewards / self.num_episodes  # type: ignore[operator]
        avg_max_reward = self.max_rewards / self.num_episodes  # type: ignore[operator]
        avg_episode_length = self.episode_lengths / self.num_episodes  # type: ignore[operator]
        pc_success = (self.num_successes.float() / self.num_episodes.float()) * 100.0  # type: ignore[union-attr]

        return {
            "avg_sum_reward": avg_sum_reward,
            "avg_max_reward": avg_max_reward,
            "pc_success": pc_success,
            "avg_episode_length": avg_episode_length,
            "n_episodes": self.num_episodes,  # type: ignore[dict-item]
        }

    def get_per_episode_data(self) -> list[dict[str, float]]:
        """Get detailed per-episode data.

        Returns:
            List of dictionaries, one per episode, containing:
                - sum_reward: Total reward for that episode
                - max_reward: Maximum reward for that episode
                - episode_length: Length of that episode
                - success: Whether that episode succeeded (if tracked)

        Note:
            This data is synchronized across processes when called in distributed mode.
        """
        if not self.all_sum_rewards:
            return []

        # Check if we have success data
        has_success = bool(self.all_successes)

        results = []
        for i, (sum_r, max_r, ep_len) in enumerate(  # type: ignore[misc]
            zip(
                self.all_sum_rewards,  # type: ignore[arg-type]
                self.all_max_rewards,  # type: ignore[arg-type]
                self.all_episode_lengths,  # type: ignore[arg-type]
                strict=True,
            ),
        ):
            record = {
                "sum_reward": float(sum_r.item()),
                "max_reward": float(max_r.item()),
                "episode_length": float(ep_len.item()),
            }
            if has_success and i < len(self.all_successes):  # type: ignore[arg-type]
                record["success"] = bool(self.all_successes[i].item() > self._SUCCESS_THRESHOLD)  # type: ignore[index]
            results.append(record)

        return results
