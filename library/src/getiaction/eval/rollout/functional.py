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

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from getiaction.data import Observation
    from getiaction.eval.video import VideoRecorder
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
    frames: list[np.ndarray] = field(default_factory=list)
    step_times: list[float] = field(default_factory=list)

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

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a rendered frame for visualization."""
        self.frames.append(frame)


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


def _extract_success_from_info(
    info: dict[str, Any] | list[dict[str, Any]],
    batch_size: int,
) -> Tensor | None:
    """Extract success information from step info dict(s).

    Handles both single-env and vectorized env info formats.
    Looks for common success keys: 'is_success', 'success', 'task_success'.

    Args:
        info: Info dict from env.step() (single dict or list of dicts for vectorized).
        batch_size: Expected batch size for output tensor.

    Returns:
        Boolean tensor of shape (batch_size,), or None if no success info found.
    """
    success_keys = ("is_success", "success", "task_success")

    # Handle list of info dicts (vectorized envs)
    if isinstance(info, list):
        successes = []
        for single_info in info:
            found = False
            for key in success_keys:
                if key in single_info:
                    successes.append(bool(single_info[key]))
                    found = True
                    break
            if not found:
                return None  # Missing success info in at least one env
        return torch.tensor(successes, dtype=torch.bool)

    # Handle single info dict
    for key in success_keys:
        if key in info:
            val = info[key]
            # Could be a single bool or array for batched envs
            if isinstance(val, (list, np.ndarray)):
                return torch.as_tensor(val, dtype=torch.bool).reshape(batch_size)
            return torch.tensor([bool(val)] * batch_size, dtype=torch.bool)

    return None


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
        env_max = int(env.max_steps)  # type: ignore[attr-defined]
        if env_max is not None:
            return env_max
    return 1000  # Default fallback


@dataclass
class _StepResult:
    """Result from a single environment step."""

    reward: Tensor
    done: Tensor
    success: Tensor | None


def _process_step_outputs(
    reward: Any,  # noqa: ANN401
    terminated: Any,  # noqa: ANN401
    truncated: Any,  # noqa: ANN401
    info: Any,  # noqa: ANN401
    batch_size: int,
) -> _StepResult:
    """Process raw step outputs into tensors.

    Args:
        reward: Raw reward from env.step()
        terminated: Raw terminated flag from env.step()
        truncated: Raw truncated flag from env.step()
        info: Info dict from env.step()
        batch_size: Expected batch size

    Returns:
        _StepResult with processed tensors
    """
    reward_t = torch.as_tensor(reward, dtype=torch.float32).reshape(batch_size)
    terminated_t = torch.as_tensor(terminated, dtype=torch.bool).reshape(batch_size)
    truncated_t = torch.as_tensor(truncated, dtype=torch.bool).reshape(batch_size)
    done = torch.logical_or(terminated_t, truncated_t)
    success = _extract_success_from_info(info, batch_size)  # type: ignore[arg-type]

    return _StepResult(reward=reward_t, done=done, success=success)


def _collect_frame(
    observation: Observation,
    frame_key: str,
) -> np.ndarray | None:
    """Extract a frame from observation for visualization.

    Args:
        observation: Current observation
        frame_key: Key to extract from observation.images

    Returns:
        Frame as numpy array (H, W, C) or None if not available
    """
    if not hasattr(observation, "images") or observation.images is None:
        return None
    if frame_key not in observation.images:  # type: ignore[operator]
        return None
    img = observation.images[frame_key]  # type: ignore[index]
    return img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # type: ignore[union-attr]


def _get_policy_action(policy: Policy, observation: Observation) -> Tensor:
    """Get action from policy, handling chunked outputs.

    Args:
        policy: Policy to query
        observation: Current observation

    Returns:
        Action tensor of shape (batch, action_dim)
    """
    policy.eval()
    action = policy(observation)

    # Handle chunked outputs (ACT returns [batch, chunk_size, action_dim])
    chunked_action_ndim = 3
    if action.dim() == chunked_action_ndim:
        action = action[:, 0, :]

    return action


def setup_rollout(
    env: Gym,
    policy: Policy,
    seed: int | None,
    max_steps: int | None,
) -> tuple[Observation, int]:
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


def run_rollout_loop(  # noqa: PLR0914
    env: Gym,
    policy: Policy,
    initial_observation: Observation,
    max_steps: int,
    *,
    render_callback: Callable[[Gym], None] | None,
    return_observations: bool = False,
    return_frames: bool = False,
    frame_key: str = "image",
    video_recorder: VideoRecorder | None = None,
) -> tuple[_EpisodeData, int, float]:
    """Run a full rollout loop.

    Args:
        env (Gym): Gym environment the policy interacts with.
        policy (Policy): The policy to interact with the environment.
        initial_observation (Observation): First initial observation of the environment.
        max_steps (int): Truncate rollout if maximum number of steps is reached.
        render_callback (Callable[[Gym], None] | None, optional): Optional callback for gym to render.
        return_observations (bool, optional): Optional save observations and return after rollout.
        return_frames (bool, optional): Whether to collect rendered frames for visualization.
        frame_key (str, optional): Key for extracting frames from observation images.
        video_recorder (VideoRecorder | None, optional): Video recorder for capturing frames.
            Recording happens DURING the loop - no separate pass needed.

    Returns:
        tuple[_EpisodeData, int, float]: The data collected, steps taken, and elapsed time.
    """
    start_time = time.perf_counter()
    observation = initial_observation
    episode_data = _EpisodeData()
    batch_size = observation.batch_size
    done_mask = torch.zeros(batch_size, dtype=torch.bool)
    success_mask = torch.zeros(batch_size, dtype=torch.bool)

    step = 0
    while step < max_steps and not torch.all(done_mask):
        # Store observations if requested
        if return_observations:
            episode_data.add_observation(observation)

        # Store frames for visualization if requested
        if return_frames:
            frame = _collect_frame(observation, frame_key)
            if frame is not None:
                episode_data.add_frame(frame)

        # Record frame for video (happens DURING the loop, not after)
        if video_recorder is not None:
            frame = _collect_frame(observation, frame_key)
            if frame is not None:
                video_recorder.record_frame(frame)

        # Policy forward
        with torch.inference_mode():
            action = _get_policy_action(policy, observation)

        # For non-vectorized envs (batch_size=1), squeeze the batch dimension
        # LiberoGym and similar envs expect action shape (action_dim,) not (1, action_dim)
        step_action = action.squeeze(0) if batch_size == 1 else action

        # Step environment
        observation, reward, terminated, truncated, info = env.step(step_action)

        if render_callback is not None:
            render_callback(env)

        # Process step outputs
        step_result = _process_step_outputs(reward, terminated, truncated, info, batch_size)
        if step_result.success is not None:
            success_mask = torch.logical_or(success_mask, step_result.success)
        done_mask = torch.logical_or(done_mask, step_result.done)

        # Store step data
        episode_data.add_step(
            action=action,
            reward=step_result.reward,
            done_mask=step_result.done,
            success=step_result.success,
        )

        step += 1

    # Store final success state
    episode_data.successes.append(success_mask)

    # Store final observation
    if return_observations:
        episode_data.observations.append(_convert_observation_to_dict(observation))

    elapsed_time = time.perf_counter() - start_time
    return episode_data, step, elapsed_time


def finalize_rollout(
    episode_data: _EpisodeData,
    step: int,
    elapsed_time: float = 0.0,
) -> dict[str, Any]:
    """Stack metrics from episode_data for final metric dict.

    Args:
        episode_data (_EpisodeData): Full episode data after rollout.
        step (int): Number of steps in environment before termination.
        elapsed_time (float): Time elapsed during rollout in seconds.

    Returns:
        dict containing:
            - action: Stacked actions (steps, batch, action_dim)
            - reward: Stacked rewards (steps, batch)
            - done: Stacked done flags (steps, batch)
            - episode_length: Number of steps taken
            - sum_reward: Total reward per environment (batch,)
            - max_reward: Maximum reward per environment (batch,)
            - success: Final success state per environment (batch,)
            - fps: Frames per second achieved
            - frames: Optional numpy array of rendered frames (steps, H, W, C)
            - observation: Optional stacked observations
    """
    actions = torch.stack(episode_data.actions, dim=0)
    rewards = torch.stack(episode_data.rewards, dim=0)
    dones = torch.stack(episode_data.dones, dim=0)

    result: dict[str, Any] = {
        "action": actions,
        "reward": rewards,
        "done": dones,
        "episode_length": step,
        "sum_reward": rewards.sum(dim=0),
        "max_reward": rewards.max(dim=0).values,
        "fps": step / elapsed_time if elapsed_time > 0 else 0.0,
    }

    # Add success info if available
    if episode_data.successes:
        # Take the last success state (accumulated over episode)
        result["success"] = episode_data.successes[-1]

    # Add frames if collected
    if episode_data.frames:
        result["frames"] = np.array(episode_data.frames)

    if episode_data.observations:
        result["observation"] = _stack_observations(episode_data.observations)

    return result


def rollout(
    env: Gym,
    policy: Policy,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
    return_observations: bool = False,
    return_frames: bool = False,
    frame_key: str = "image",
    render_callback: Callable[[Gym], None] | None = None,
    video_recorder: VideoRecorder | None = None,
) -> dict[str, Any]:
    """Runs a policy in an environment for a single episode.

    This function is equivalent to the notebook's run_episode() but with
    additional features for distributed evaluation and batch support.

    Recording happens DURING the episode loop - no separate pass needed.
    The video_recorder.start_episode() must be called before this function,
    and video_recorder.finish_episode() should be called after.

    Args:
        env (Gym): Environment to interact with.
        policy (Policy): Policy used to select actions.
        seed (int | None, optional): RNG seed for the environment. Defaults to None.
        max_steps (int | None, optional): Maximum number of steps before termination.
            If None, runs until the episode ends. Defaults to None.
        return_observations (bool, optional): Whether to include the observation
            sequence in the output. Defaults to False.
        return_frames (bool, optional): Whether to collect rendered frames for
            visualization. Defaults to False.
        frame_key (str, optional): Key for extracting frames from observation.images.
            Defaults to "image".
        render_callback (Callable[[Gym], None] | None, optional): Optional callback
            invoked each step for rendering. Defaults to None.
        video_recorder (VideoRecorder | None, optional): Video recorder for capturing
            frames during the rollout. Call start_episode() before and finish_episode()
            after. Defaults to None.

    Returns:
        dict[str, Any]: Episode information containing:
            - action: Actions taken (steps, batch, action_dim)
            - reward: Rewards received (steps, batch)
            - done: Done flags (steps, batch)
            - episode_length: Number of steps
            - sum_reward: Total reward per env (batch,)
            - max_reward: Max reward per env (batch,)
            - success: Success flag per env (batch,) - if available from env
            - fps: Frames per second achieved
            - frames: Rendered frames (steps, H, W, C) - if return_frames=True
            - observation: Observations - if return_observations=True
    """
    # init rollout and policy
    initial_observation, max_steps = setup_rollout(env, policy, seed, max_steps)

    # if render callback, call for first observation
    if render_callback:
        render_callback(env)

    # run episode loops
    recorder, step, elapsed_time = run_rollout_loop(
        env=env,
        policy=policy,
        initial_observation=initial_observation,
        max_steps=max_steps,
        render_callback=render_callback,
        return_observations=return_observations,
        return_frames=return_frames,
        frame_key=frame_key,
        video_recorder=video_recorder,
    )

    # finalize
    return finalize_rollout(recorder, step, elapsed_time)


def _aggregate_episode_metrics(per_episode: list[dict[str, Any]], n_episodes: int) -> dict[str, Any]:
    """Aggregate metrics across multiple episodes.

    Args:
        per_episode: List of per-episode metric dicts
        n_episodes: Total number of episodes

    Returns:
        Aggregated metrics dictionary
    """
    aggregated: dict[str, Any] = {
        "avg_sum_reward": float(np.mean([ep["sum_reward"] for ep in per_episode])),
        "avg_max_reward": float(np.mean([ep["max_reward"] for ep in per_episode])),
        "avg_episode_length": float(np.mean([ep["episode_length"] for ep in per_episode])),
        "avg_fps": float(np.mean([ep["fps"] for ep in per_episode])),
        "n_episodes": n_episodes,
    }

    # Add success rate if available
    if per_episode and "success" in per_episode[0]:
        successes = [ep["success"] for ep in per_episode]
        aggregated["pc_success"] = float(np.mean(successes)) * 100.0
        aggregated["num_successes"] = sum(successes)

    return aggregated


def _extract_episode_records(
    rollout_result: dict[str, Any],
    take: int,
    episode_idx_start: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    """Extract per-episode records from a rollout result.

    Args:
        rollout_result: Result from rollout()
        take: Number of episodes to extract
        episode_idx_start: Starting episode index
        seed: Seed used for the rollout

    Returns:
        List of episode record dicts
    """
    records = []
    success_tensor = rollout_result.get("success")

    for env_i in range(take):
        record = {
            "episode_idx": episode_idx_start + env_i,
            "sum_reward": float(rollout_result["sum_reward"][env_i]),
            "max_reward": float(rollout_result["max_reward"][env_i]),
            "episode_length": rollout_result["episode_length"],
            "fps": rollout_result.get("fps", 0.0),
            "seed": seed,
        }
        if success_tensor is not None:
            record["success"] = bool(success_tensor[env_i].item())
        records.append(record)

    return records


def evaluate_policy(
    env: Gym,
    policy: Policy,
    n_episodes: int,
    *,
    start_seed: int | None = None,
    max_steps: int | None = None,
    return_episode_data: bool = False,
    video_recorder: VideoRecorder | None = None,
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
        video_recorder (VideoRecorder | None, optional): Video recorder for capturing
            frames during evaluation. Defaults to None.

    Returns:
        dict[str, Any]: Aggregate evaluation results containing:
            - per_episode: List of per-episode metrics
            - aggregated: Dict with avg_sum_reward, avg_max_reward, pc_success, etc.
            - episodes: Optional full rollout data
    """
    per_episode: list[dict[str, Any]] = []
    episode_data_list: list[dict[str, Any]] | None = [] if return_episode_data else None
    episodes_collected, rollout_idx = 0, 0

    while episodes_collected < n_episodes:
        seed = None if start_seed is None else start_seed + rollout_idx
        rollout_idx += 1

        # Start video recording for this episode
        if video_recorder is not None:
            episode_name = f"episode_{episodes_collected:03d}"
            video_recorder.start_episode(episode_name)

        rollout_result = rollout(
            env,
            policy,
            seed=seed,
            max_steps=max_steps,
            return_observations=return_episode_data,
            video_recorder=video_recorder,
        )

        # Finish video recording
        if video_recorder is not None:
            success_tensor = rollout_result.get("success")
            success = bool(success_tensor[0].item()) if success_tensor is not None else False
            video_recorder.finish_episode(success=success)

        batch_size = rollout_result["sum_reward"].shape[0]
        take = min(batch_size, n_episodes - episodes_collected)

        per_episode.extend(_extract_episode_records(rollout_result, take, episodes_collected, seed))
        episodes_collected += take

        if return_episode_data and episode_data_list is not None:
            episode_data_list.append(rollout_result)

    result: dict[str, Any] = {
        "per_episode": per_episode,
        "aggregated": _aggregate_episode_metrics(per_episode, n_episodes),
    }

    if return_episode_data and episode_data_list is not None:
        result["episodes"] = episode_data_list

    return result
