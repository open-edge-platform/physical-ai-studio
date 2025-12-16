# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for computing delta timestamps from LeRobot policy configs."""

from __future__ import annotations

from lerobot.policies.factory import make_policy_config


def get_delta_timestamps_from_policy(
    policy_name: str,
    fps: int = 10,
    obs_image_key: str = "observation.images.top",
    obs_state_key: str = "observation.state",
) -> dict[str, list[float]]:
    """Derive delta timestamps configuration from LeRobot policy config.

    This extracts n_obs_steps and action chunk/horizon size from the policy's
    default configuration to automatically compute the correct delta timestamps
    for use with LeRobotDataModule.

    For policies like Groot that have action_delta_indices with a capped horizon,
    we use the length of action_delta_indices rather than chunk_size to ensure
    the generated delta timestamps match what the policy expects.

    Args:
        policy_name: Name of the LeRobot policy (e.g., "act", "diffusion", "groot").
        fps: Frames per second of the dataset.
        obs_image_key: Key for image observations in the dataset.
        obs_state_key: Key for state observations in the dataset.

    Returns:
        Dictionary with delta timestamps for observation and action keys.

    Example:
        >>> from getiaction.data.lerobot import get_delta_timestamps_from_policy
        >>> from getiaction.data.lerobot import LeRobotDataModule

        >>> delta_timestamps = get_delta_timestamps_from_policy("act", fps=10)
        >>> datamodule = LeRobotDataModule(
        ...     repo_id="lerobot/aloha_sim_insertion_human",
        ...     delta_timestamps=delta_timestamps,
        ... )
    """
    config = make_policy_config(policy_name)

    n_obs_steps: int = getattr(config, "n_obs_steps", 1)

    # Initialize delta_timestamps dictionary
    delta_timestamps: dict[str, list[float]] = {}

    # For policies with action_delta_indices (e.g., Groot), use that length as the source of truth
    # This respects the model's capped horizon (e.g., Groot caps at 16 steps even though chunk_size=50)
    action_delta_indices = getattr(config, "action_delta_indices", None)
    if action_delta_indices is not None:
        # Observation timestamps: indices from -(n_obs_steps-1) to 0
        if n_obs_steps > 1:
            obs_indices = list(range(-(n_obs_steps - 1), 1))  # e.g., [-1, 0] for n_obs_steps=2
            delta_timestamps[obs_image_key] = [i / fps for i in obs_indices]
            delta_timestamps[obs_state_key] = [i / fps for i in obs_indices]

        # Action timestamps: use the action_delta_indices directly
        delta_timestamps["action"] = [i / fps for i in action_delta_indices]

        return delta_timestamps

    # Fallback for policies without action_delta_indices (ACT, Diffusion)
    # Get action sequence length - different policies use different attribute names
    action_length_raw = (
        getattr(config, "chunk_size", None)
        or getattr(config, "horizon", None)
        or getattr(config, "action_chunk_size", None)
        or getattr(config, "n_action_steps", None)
    )
    action_length: int = int(action_length_raw) if action_length_raw is not None else 1

    # Observation timestamps: indices from -(n_obs_steps-1) to 0
    if n_obs_steps > 1:
        obs_indices = list(range(-(n_obs_steps - 1), 1))  # e.g., [-1, 0] for n_obs_steps=2
        delta_timestamps[obs_image_key] = [i / fps for i in obs_indices]
        delta_timestamps[obs_state_key] = [i / fps for i in obs_indices]

    # Action timestamps: depends on policy type (diffusion starts from -1)
    action_indices = list(range(-1, action_length - 1)) if policy_name == "diffusion" else list(range(action_length))

    delta_timestamps["action"] = [i / fps for i in action_indices]

    return delta_timestamps
