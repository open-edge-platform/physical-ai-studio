# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Policy evaluation utilities for gym environments.

This module provides functions for evaluating policies in gym environments,
collecting metrics, and generating evaluation reports. The evaluation approach
is inspired by LeRobot's evaluation framework but adapted for getiaction's
architecture using Observation dataclass and Lightning integration.

TODO: Refactor this module to further improve code organization:
  - Consider splitting helper functions into a separate utils module
  - Extract episode data collection into a dedicated class
  - Add more comprehensive type hints for observation formats
  - Improve error handling for edge cases (empty episodes, invalid observations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from getiaction.data.utils import infer_batch_size

if TYPE_CHECKING:
    from collections.abc import Callable

    from getiaction.gyms import Gym
    from getiaction.policies.base import Policy


@dataclass
class _EpisodeData:
    """Container for storing episode data during rollout."""

    observations: list[dict[str, Any]] = field(default_factory=list)
    actions: list[Tensor] = field(default_factory=list)
    rewards: list[Tensor] = field(default_factory=list)
    successes: list[Tensor] = field(default_factory=list)
    dones: list[Tensor] = field(default_factory=list)


def _convert_observation_to_dict(observation: Any) -> dict[str, Any]:  # noqa: ANN401
    """Convert observation to dictionary format for storage.

    Args:
        observation: The observation to convert (can be object with __dict__, dict, or other).

    Returns:
        Dictionary representation of the observation.
    """
    if hasattr(observation, "__dict__"):
        return {k: v for k, v in observation.__dict__.items() if v is not None}
    if isinstance(observation, dict):
        return observation
    return {"observation": observation}


def _stack_observations(all_observations: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Stack observation dictionaries into tensors.

    Args:
        all_observations: List of observation dictionaries.

    Returns:
        Dictionary mapping keys to stacked tensors.
    """
    stacked_obs: dict[str, Tensor] = {}
    if not all_observations:
        return stacked_obs

    keys = all_observations[0].keys()
    for key in keys:
        values = [obs.get(key) for obs in all_observations if obs.get(key) is not None]
        if values and isinstance(values[0], (Tensor, np.ndarray)):
            tensors = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in values if v is not None]
            if tensors:
                stacked_obs[key] = torch.stack(tensors, dim=0)  # type: ignore[arg-type]
    return stacked_obs


def _get_max_steps(env: Gym, max_steps: int | None) -> int:
    """Get maximum steps if available from Gym env.

    Args:
        env (Gym): the Gym environment to call.
        max_steps (int | None, optional): return these max_steps instead.

    Returns:
        max_steps (int): maximum number of steps for episode.
    """
    if max_steps is not None:
        return max_steps
    if hasattr(env, "max_steps"):
        env_max = int(env.max_steps)
        if env_max is not None:
            return env_max
    return 1000  # Default fallback


def rollout(  # noqa: PLR0914
    env: Gym,
    policy: Policy,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
    return_observations: bool = False,
    render_callback: Callable[[Gym], None] | None = None,
) -> dict[str, Any]:
    """Runs a policy in an environment for a single episode.

    Args:
        env (Gym): Environment to interact with.
        policy (Policy): Policy used to select actions.
        seed (int | None, optional): RNG seed for the environment. Defaults to None.
        max_steps (int | None, optional): Maximum number of steps before termination.
            If None, runs until the episode ends. Defaults to None.
        return_observations (bool, optional): Whether to include the observation
            sequence in the output. Defaults to False.
        render_callback (Callable[[Gym], None] | None, optional): Optional callback
            invoked each step for rendering. Defaults to None.

    Returns:
        dict[str, Any]: Episode information, including rewards, actions, and optionally
        observations.
    """
    max_steps = _get_max_steps(env, max_steps)

    # Reset environment → batched observation
    observation, _ = env.reset(seed=seed)
    batch_size = infer_batch_size(observation)

    # Reset policy if needed
    if hasattr(policy, "reset") and callable(policy.reset):
        policy.reset()

    if render_callback is not None:
        render_callback(env)

    episode_data = _EpisodeData()
    step = 0

    # Track per-env done
    done_mask = torch.zeros(batch_size, dtype=torch.bool)

    # run loop for max steps or until all done
    while step < max_steps and not torch.all(done_mask):
        # Store observations if requested
        if return_observations:
            episode_data.observations.append(_convert_observation_to_dict(observation))

        # Policy forward (already batched)
        with torch.inference_mode():
            policy.eval()
            action = policy(observation)  # shape: (B, action_dim)

        # Step environment (env expects batched action)
        observation, reward, terminated, truncated, _info = env.step(action)

        if render_callback is not None:
            render_callback(env)

        # Convert arrays -> tensors
        reward_t = torch.as_tensor(reward, dtype=torch.float32).reshape(batch_size)
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool).reshape(batch_size)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool).reshape(batch_size)

        # Per-env step done
        done = torch.logical_or(terminated_t, truncated_t)
        done_mask = torch.logical_or(done_mask, done)

        # Store step data
        episode_data.actions.append(action)
        episode_data.rewards.append(reward_t)
        episode_data.dones.append(done_mask.clone())

        step += 1

    # Store final observation
    if return_observations:
        episode_data.observations.append(_convert_observation_to_dict(observation))

    # Stack (T, B, ..)
    actions = torch.stack(episode_data.actions, dim=0)
    rewards = torch.stack(episode_data.rewards, dim=0)
    dones = torch.stack(episode_data.dones, dim=0)

    # Episode-level stats per env
    sum_reward = rewards.sum(dim=0)  # (B,)
    max_reward = rewards.max(dim=0).values  # (B,)

    ret = {
        "action": actions,
        "reward": rewards,
        "done": dones,
        "episode_length": step,
        "sum_reward": sum_reward,
        "max_reward": max_reward,
    }

    if return_observations:
        ret["observation"] = _stack_observations(episode_data.observations)

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
    """Evaluates a policy over multiple episodes.

    Args:
        env (Gym): Environment used for evaluation.
        policy (Policy): Policy to evaluate.
        n_episodes (int): Number of episodes to run.
        start_seed (int | None, optional): Initial seed; incremented per episode
            if provided. Defaults to None.
        max_steps (int | None, optional): Maximum steps per episode. Defaults to None.
        return_episode_data (bool, optional): Whether to include per-episode rollout
            data in the result. Defaults to False.

    Returns:
        dict[str, Any]: Aggregate evaluation results, including mean reward and
        optionally episode-level data.
    """
    per_episode = []
    episode_data_list: list[dict[str, Any]] | None = [] if return_episode_data else None

    episodes_collected = 0
    rollout_idx = 0

    while episodes_collected < n_episodes:
        # Seed per rollout
        seed = None if start_seed is None else start_seed + rollout_idx
        rollout_idx += 1

        rollout_result = rollout(
            env,
            policy,
            seed=seed,
            max_steps=max_steps,
            return_observations=return_episode_data,
        )

        # Extract batch size
        batch_size = rollout_result["sum_reward"].shape[0]

        # How many episodes still needed?
        # Could be vectorized hence the take
        remaining = n_episodes - episodes_collected
        take = min(batch_size, remaining)

        # Record per-env results
        for env_i in range(take):
            per_episode.append({
                "episode_idx": episodes_collected,
                "sum_reward": float(rollout_result["sum_reward"][env_i]),
                "max_reward": float(rollout_result["max_reward"][env_i]),
                "episode_length": rollout_result["episode_length"],
                "seed": seed,
            })
            episodes_collected += 1

        # Store full rollout if requested
        if return_episode_data and episode_data_list is not None:
            episode_data_list.append(rollout_result)

    # Aggregate metrics
    sum_rewards = [ep["sum_reward"] for ep in per_episode]
    max_rewards = [ep["max_reward"] for ep in per_episode]
    episode_lengths = [ep["episode_length"] for ep in per_episode]

    aggregated = {
        "avg_sum_reward": float(np.mean(sum_rewards)),
        "avg_max_reward": float(np.mean(max_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "n_episodes": n_episodes,
    }

    result = {
        "per_episode": per_episode,
        "aggregated": aggregated,
    }

    if return_episode_data and episode_data_list is not None:
        result["episodes"] = episode_data_list

    return result
