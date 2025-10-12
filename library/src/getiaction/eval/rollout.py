# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Policy evaluation utilities for gym environments.

This module provides functions for evaluating policies in gym environments,
collecting metrics, and generating evaluation reports. The evaluation approach
is inspired by LeRobot's evaluation framework but adapted for getiaction's
architecture using Observation dataclass and Lightning integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from getiaction.gyms import Gym
    from getiaction.policies.base import Policy


def rollout(
    env: Gym,
    policy: Policy,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
    return_observations: bool = False,
    render_callback: Callable[[Gym], None] | None = None,
) -> dict[str, Any]:
    """Run a complete policy rollout in a gym environment.

    Executes the policy in the environment until termination, truncation, or
    reaching max_steps. Collects actions, rewards, success flags, and optionally
    observations throughout the episode.

    Args:
        env: The gym environment to run the rollout in.
        policy: The policy to use for action selection. Must have a `select_action`
            method that takes an Observation and returns an action tensor.
        seed: Random seed for environment reset. If None, no seed is set.
        max_steps: Maximum number of steps to run. If None, runs until done.
            If the environment has a built-in max_episode_steps, that takes precedence.
        return_observations: Whether to include observations in the returned data.
            This increases memory usage but enables advanced analysis.
        render_callback: Optional callback function called after reset and each step,
            useful for rendering or recording videos. Receives the environment as input.

    Returns:
        Dictionary containing:
            - "action": (sequence, action_dim) tensor of actions taken
            - "reward": (sequence,) tensor of rewards received
            - "success": (sequence,) tensor of success flags (True only at termination)
            - "done": (sequence,) tensor of cumulative done flags
            - "observation": (optional) Dictionary of (sequence + 1, *) observation tensors
            - "episode_length": Integer number of steps in the episode
            - "sum_reward": Float sum of all rewards
            - "max_reward": Float maximum reward in the episode
            - "is_success": Boolean whether the episode succeeded

    Example:
        >>> from getiaction.gyms import PushTGym
        >>> from getiaction.policies.lerobot import ACT
        >>> policy = ACT.from_pretrained("path/to/checkpoint")
        >>> env = PushTGym()
        >>> result = rollout(env, policy, seed=42, max_steps=300)
        >>> print(f"Success: {result['is_success']}, Reward: {result['sum_reward']:.2f}")

    Note:
        The policy's `reset()` method is called before starting the rollout to
        clear any internal state (action queues, observation history, etc.).
    """
    # Determine max steps from environment or argument
    if max_steps is None:
        # Use Gym's get_max_episode_steps() method if available
        max_steps = env.get_max_episode_steps() if hasattr(env, "get_max_episode_steps") else None
        # Fallback to default if still None
        if max_steps is None:
            max_steps = 1000

    # Reset policy state (clears action queues, observation histories, etc.)
    if hasattr(policy, "reset") and callable(policy.reset):
        policy.reset()

    # Reset environment - returns (observation, info) tuple
    observation, info = env.reset(seed=seed)

    if render_callback is not None:
        render_callback(env)

    # Storage for episode data
    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    done = False

    while not done and step < max_steps:
        # Store initial observation if requested
        if return_observations:
            # Convert observation to dict format for storage
            if hasattr(observation, "__dict__"):
                obs_dict = {k: v for k, v in observation.__dict__.items() if v is not None}
            elif isinstance(observation, dict):
                obs_dict = observation
            else:
                obs_dict = {"observation": observation}
            all_observations.append(obs_dict)

        # Get action from policy
        # Policy.select_action expects the observation in the format it was trained on
        with torch.inference_mode():
            policy.eval()
            action = policy.select_action(observation)

        # Convert action to numpy for environment
        if isinstance(action, Tensor):
            action_numpy = action.detach().cpu().numpy()
        else:
            action_numpy = np.asarray(action)

        # Ensure action is 1D for single environment
        if action_numpy.ndim > 1:
            action_numpy = action_numpy.squeeze(0)

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action_numpy)

        if render_callback is not None:
            render_callback(env)

        # Check for success (usually in info dict at termination)
        is_success = info.get("is_success", False) if (terminated or truncated) else False

        # Update done flag
        done = terminated or truncated

        # Store step data
        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.tensor(reward, dtype=torch.float32))
        all_successes.append(torch.tensor(is_success, dtype=torch.bool))
        all_dones.append(torch.tensor(done, dtype=torch.bool))

        step += 1

    # Store final observation if requested
    if return_observations:
        if hasattr(observation, "__dict__"):
            obs_dict = {k: v for k, v in observation.__dict__.items() if v is not None}
        elif isinstance(observation, dict):
            obs_dict = observation
        else:
            obs_dict = {"observation": observation}
        all_observations.append(obs_dict)

    # Stack tensors along sequence dimension
    ret = {
        "action": torch.stack(all_actions, dim=0),
        "reward": torch.stack(all_rewards, dim=0),
        "success": torch.stack(all_successes, dim=0),
        "done": torch.stack(all_dones, dim=0),
        "episode_length": step,
        "sum_reward": torch.stack(all_rewards).sum().item(),
        "max_reward": torch.stack(all_rewards).max().item() if all_rewards else 0.0,
        "is_success": any(s.item() for s in all_successes),
    }

    # Add observations if requested
    if return_observations:
        # Stack observations by key
        stacked_obs = {}
        if all_observations:
            keys = all_observations[0].keys()
            for key in keys:
                values = [obs.get(key) for obs in all_observations if obs.get(key) is not None]
                if values and isinstance(values[0], (Tensor, np.ndarray)):
                    tensors = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in values]
                    stacked_obs[key] = torch.stack(tensors, dim=0)
        ret["observation"] = stacked_obs

    return ret


def evaluate_policy(
    env: Gym,
    policy: Policy,
    n_episodes: int,
    *,
    start_seed: int | None = None,
    max_steps: int | None = None,
    return_episode_data: bool = False,
) -> dict[str, Any]:
    """Evaluate a policy over multiple episodes in a gym environment.

    Runs the policy for n_episodes, collecting success rates, rewards,
    and episode lengths. Useful for benchmarking policy performance.

    Args:
        env: The gym environment to evaluate in.
        policy: The policy to evaluate.
        n_episodes: Number of episodes to run.
        start_seed: Starting seed for episodes. Each episode uses seed + episode_idx.
            If None, episodes are not seeded.
        max_steps: Maximum steps per episode. If None, uses environment default.
        return_episode_data: Whether to return detailed per-episode data including
            observations and actions. Increases memory usage significantly.

    Returns:
        Dictionary containing:
            - "per_episode": List of dicts with per-episode metrics (sum_reward,
              max_reward, success, episode_length, seed)
            - "aggregated": Dict with aggregate metrics:
                * "avg_sum_reward": Mean cumulative reward
                * "avg_max_reward": Mean maximum reward
                * "pc_success": Success rate as percentage
                * "avg_episode_length": Mean episode length
            - "episodes": (optional) List of full episode data if return_episode_data=True

    Example:
        >>> results = evaluate_policy(env, policy, n_episodes=50, start_seed=1000)
        >>> print(f"Success rate: {results['aggregated']['pc_success']:.1f}%")
        >>> print(f"Avg reward: {results['aggregated']['avg_sum_reward']:.2f}")
    """
    # Collect metrics
    per_episode = []
    episode_data_list = [] if return_episode_data else None

    for episode_idx in range(n_episodes):
        # Set seed if provided
        seed = None if start_seed is None else start_seed + episode_idx

        # Run rollout
        rollout_result = rollout(
            env,
            policy,
            seed=seed,
            max_steps=max_steps,
            return_observations=return_episode_data,
        )

        # Store per-episode metrics
        per_episode.append({
            "episode_idx": episode_idx,
            "sum_reward": rollout_result["sum_reward"],
            "max_reward": rollout_result["max_reward"],
            "success": rollout_result["is_success"],
            "episode_length": rollout_result["episode_length"],
            "seed": seed,
        })

        # Store full episode data if requested
        if return_episode_data:
            episode_data_list.append(rollout_result)

    # Compute aggregate metrics
    sum_rewards = [ep["sum_reward"] for ep in per_episode]
    max_rewards = [ep["max_reward"] for ep in per_episode]
    successes = [ep["success"] for ep in per_episode]
    episode_lengths = [ep["episode_length"] for ep in per_episode]

    aggregated = {
        "avg_sum_reward": float(np.mean(sum_rewards)),
        "avg_max_reward": float(np.mean(max_rewards)),
        "pc_success": float(np.mean(successes) * 100),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "n_episodes": n_episodes,
    }

    result = {
        "per_episode": per_episode,
        "aggregated": aggregated,
    }

    if return_episode_data:
        result["episodes"] = episode_data_list

    return result
