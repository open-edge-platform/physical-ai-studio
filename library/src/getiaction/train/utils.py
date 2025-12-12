# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Util functions for Training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning_utilities.core.imports import module_available

from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from getiaction.data import DataModule
    from getiaction.policies.base.policy import Policy


def _get_delta_indices(model: Any, attr_name: str) -> list[int] | None:  # noqa: ANN401
    """Get delta indices from a model, handling both first-party and LeRobot policies.

    Args:
        model: The model to extract delta indices from
        attr_name: Name of the delta indices attribute (e.g., 'observation_delta_indices')

    Returns:
        List of delta indices or None if not available/not needed.
    """
    # Try direct attribute access (first-party policies like ACT)
    if hasattr(model, attr_name):
        return getattr(model, attr_name)

    # Try config attribute (LeRobot policies)
    if hasattr(model, "config"):
        config = model.config

        # First, check if config has the property directly (e.g., Pi0Config.action_delta_indices)
        # This is preferred as it uses the correct logic for each policy
        if hasattr(config, attr_name):
            result = getattr(config, attr_name)
            if result is not None:
                return result

        # Fallback: Convert observation_delta_indices -> n_obs_steps
        # action_delta_indices -> chunk_size (for VLA policies)
        if attr_name == "observation_delta_indices" and hasattr(config, "n_obs_steps"):
            n_steps = config.n_obs_steps
            # Only add time dimension if n_obs_steps > 1 (i.e., we need history)
            # n_obs_steps=1 means just current observation, no time dimension needed
            return list(range(-n_steps + 1, 1)) if n_steps > 1 else None
        if attr_name == "action_delta_indices" and hasattr(config, "n_action_steps"):
            n_steps = config.n_action_steps
            return list(range(n_steps)) if n_steps > 0 else None
        # Try direct config attribute
        if hasattr(config, attr_name):
            return getattr(config, attr_name)

    return None


def reformat_dataset_to_match_policy(policy: Policy, datamodule: DataModule) -> None:
    """Reformat dataset to have correct deltas and parameters depending on policy.

    This function auto-computes delta_timestamps from the policy's configuration
    (e.g., chunk_size, n_obs_steps) and applies them to the dataset. This eliminates
    the need to manually specify delta_timestamps in YAML configs.

    Works with both:
    - _LeRobotDatasetAdapter (data_format="getiaction")
    - LeRobotDataset (data_format="lerobot")
    """
    if not module_available("lerobot"):
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset  # noqa: PLC0415
    from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices  # noqa: PLC0415

    # Get the underlying LeRobot dataset
    lerobot_dataset: LeRobotDataset | None = None
    if isinstance(datamodule.train_dataset, _LeRobotDatasetAdapter):
        lerobot_dataset = datamodule.train_dataset._lerobot_dataset  # noqa: SLF001
    elif isinstance(datamodule.train_dataset, _LeRobotDataset):
        lerobot_dataset = datamodule.train_dataset

    if lerobot_dataset is None:
        return

    # Skip if delta_indices already set (user provided delta_timestamps in config)
    if lerobot_dataset.delta_indices is not None:
        return

    delta_timestamps = {}

    # Get the LeRobot policy model for delta indices
    # For policies with lerobot_policy attribute, use that; otherwise use policy.model
    lerobot_model = getattr(policy, "lerobot_policy", None) or getattr(policy, "model", None)
    if lerobot_model is None:
        return

    for key in lerobot_dataset.meta.features:
        reward_delta_indices = _get_delta_indices(lerobot_model, "reward_delta_indices")
        if key == "next.reward" and reward_delta_indices is not None:
            delta_timestamps[key] = [i / lerobot_dataset.fps for i in reward_delta_indices]

        action_delta_indices = _get_delta_indices(lerobot_model, "action_delta_indices")
        if key == "action" and action_delta_indices is not None:
            delta_timestamps[key] = [i / lerobot_dataset.fps for i in action_delta_indices]

        observation_delta_indices = _get_delta_indices(lerobot_model, "observation_delta_indices")
        if key.startswith("observation.") and observation_delta_indices is not None:
            delta_timestamps[key] = [i / lerobot_dataset.fps for i in observation_delta_indices]

    # Apply delta_timestamps to the dataset
    if delta_timestamps:
        check_delta_timestamps(delta_timestamps, lerobot_dataset.fps, lerobot_dataset.tolerance_s)
        lerobot_dataset.delta_indices = get_delta_indices(delta_timestamps, lerobot_dataset.fps)
