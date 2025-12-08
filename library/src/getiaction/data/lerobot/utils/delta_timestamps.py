# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for computing delta timestamps from LeRobot policy configs."""

from __future__ import annotations


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
    from lerobot.policies.factory import make_policy_config

    config = make_policy_config(policy_name)

    n_obs_steps = getattr(config, "n_obs_steps", 1)

    # Get action sequence length - different policies use different attribute names
    action_length = (
        getattr(config, "chunk_size", None)
        or getattr(config, "horizon", None)
        or getattr(config, "action_chunk_size", None)
        or getattr(config, "n_action_steps", 1)
    )

    delta_timestamps: dict[str, list[float]] = {}

    # Observation timestamps: indices from -(n_obs_steps-1) to 0
    if n_obs_steps > 1:
        obs_indices = list(range(-(n_obs_steps - 1), 1))  # e.g., [-1, 0] for n_obs_steps=2
        delta_timestamps[obs_image_key] = [i / fps for i in obs_indices]
        delta_timestamps[obs_state_key] = [i / fps for i in obs_indices]

    # Action timestamps: depends on policy type
    if policy_name == "diffusion":
        # Diffusion predicts horizon steps starting from -1
        action_indices = list(range(-1, action_length - 1))
    else:
        # Other policies predict chunk_size steps starting from 0
        action_indices = list(range(action_length))

    delta_timestamps["action"] = [i / fps for i in action_indices]

    return delta_timestamps
