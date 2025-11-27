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

if TYPE_CHECKING:
    from collections.abc import Callable

    from getiaction.data import Observation
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

    def add_step(
        self,
        action: Tensor,
        reward: Tensor,
        done_mask: Tensor,
        success: Tensor | None = None,
    ) -> None:
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done_mask.clone())
        if success is not None:
            self.successes.append(success)

    def add_observation(self, obs: Observation) -> None:
        self.observations.append(_convert_observation_to_dict(obs))


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


def setup_rollout(env: Gym, policy: Policy, seed: int | None, max_steps: int | None) -> tuple[Observation, int]:
    """Set up rollout by attaching max_steps, seed, resetting policy and providing first observation.

    Args:
        env (Gym): environment to probe max_steps and init.
        policy (Policy): policy to reset if it has attribute.
        seed (int | None): seed to init reset.
        max_steps (int | None): maximum number of steps

    Returns:
        tuple[Observation, int]: First Observation and maximum number of steps of rollout
    """
    max_steps = _get_max_steps(env, max_steps)

    # Reset environment → batched observation
    observation, _ = env.reset(seed=seed)

    # Reset policy if needed
    if hasattr(policy, "reset") and callable(policy.reset):
        policy.reset()

    return observation, max_steps


def run_rollout_loop(
    env: Gym,
    policy: Policy,
    initial_observation: Observation,
    max_steps: int,
    *,
    render_callback: Callable[[Gym], None] | None,
    return_observations: bool = False,
) -> tuple[_EpisodeData, int]:
    """Run a full rollout loop.

    Args:
        env (Gym): Gym environment the policy interacts with.
        policy (Policy): The policy to interact with the environment.
        initial_observation (Observation): First inital observation of the environmnet
        max_steps (int): Truncate rollout if maximum number of steps is reached.
        render_callback (Callable[[Gym], None] | None, optional): Optional callback for gym to render.
        return_observations (bool, optional): Optional save observations and return after rollout.

    Returns:
        tuple[_EpisodeData, int]: The data collected through the rollout and the step terminated on.
    """
    # set initial observation to re-usable variable name
    observation = initial_observation
    # Episode recorder
    episode_data = _EpisodeData()
    # Track per-env done
    batch_size = observation.batch_size
    done_mask = torch.zeros(batch_size, dtype=torch.bool)

    # run loop for max steps or until all done
    step = 0
    while step < max_steps and not torch.all(done_mask):
        # Store observations if requested
        if return_observations:
            episode_data.add_observation(observation)

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

        # zero out actions for gyms which have finished.
        if done.any():
            action = action.masked_fill(done_mask.unsqueeze(-1), 0)

        # Store step data
        episode_data.add_step(
            action=action,
            reward=reward_t,
            done_mask=done,
        )

        step += 1

    # Store final observation
    if return_observations:
        episode_data.observations.append(_convert_observation_to_dict(observation))

    return episode_data, step


def finalize_rollout(episode_data: _EpisodeData, step: int) -> dict[str, torch.Tensor | float]:
    """Stack metrics from episode_data for final metric dict.

    Args:
        episode_data (_EpisodeData): Full episode data after rollout.
        step (int): Number of steps in environment before termination.

    Returns:
        dict[str, torch.Tensor | float]: _description_
    """
    actions = torch.stack(episode_data.actions, dim=0)
    rewards = torch.stack(episode_data.rewards, dim=0)
    dones = torch.stack(episode_data.dones, dim=0)

    result = {
        "action": actions,
        "reward": rewards,
        "done": dones,
        "episode_length": step,
        "sum_reward": rewards.sum(dim=0),
        "max_reward": rewards.max(dim=0).values,
    }

    if episode_data.observations is not None:
        result["observation"] = _stack_observations(episode_data.observations)

    return result


def rollout(
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
    # init rollout and policy
    initial_observation, max_steps = setup_rollout(env, policy, seed, max_steps)

    # if render callback, call for first observation
    if render_callback:
        render_callback(env)

    # run episode loops
    recorder, step = run_rollout_loop(
        env=env,
        policy=policy,
        initial_observation=initial_observation,
        max_steps=max_steps,
        render_callback=render_callback,
        return_observations=return_observations,
    )

    # finalize
    return finalize_rollout(recorder, step)


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
