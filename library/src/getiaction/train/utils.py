# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Util functions for Training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices

from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter

if TYPE_CHECKING:
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
        # Convert observation_delta_indices -> n_obs_steps
        # action_delta_indices -> n_action_steps, etc.
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
    """Reformat dataset to have correct deltas and parametrs depending on policy."""
    # if lerobot dataset, set delta timesteps correctly
    # https://github.com/huggingface/lerobot/blob/33cad37054c2b594ceba57463e8f11ee374fa93c/src/lerobot/datasets/factory.py#L37
    if isinstance(datamodule.train_dataset, _LeRobotDatasetAdapter):
        delta_timestamps = {}
        lerobot_dataset = datamodule.train_dataset

        # Get the LeRobot policy model for delta indices
        # For policies with lerobot_policy attribute, use that; otherwise use policy.model
        lerobot_model = getattr(policy, "lerobot_policy", None) or policy.model

        for key in lerobot_dataset.raw_features:
            reward_delta_indices = _get_delta_indices(lerobot_model, "reward_delta_indices")
            if key == "next.reward" and reward_delta_indices is not None:
                delta_timestamps[key] = [i / lerobot_dataset.fps for i in reward_delta_indices]

            action_delta_indices = _get_delta_indices(lerobot_model, "action_delta_indices")
            if key == "action" and action_delta_indices is not None:
                delta_timestamps[key] = [i / lerobot_dataset.fps for i in action_delta_indices]

            observation_delta_indices = _get_delta_indices(lerobot_model, "observation_delta_indices")
            if key.startswith("observation.") and observation_delta_indices is not None:
                delta_timestamps[key] = [i / lerobot_dataset.fps for i in observation_delta_indices]

        # in place change the lerobot dataset
        if delta_timestamps:
            check_delta_timestamps(delta_timestamps, lerobot_dataset.fps, lerobot_dataset.tolerance_s)
            lerobot_dataset.delta_indices = get_delta_indices(delta_timestamps, lerobot_dataset.fps)
