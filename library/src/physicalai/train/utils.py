# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Util functions for Training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lerobot.datasets.feature_utils import check_delta_timestamps, get_delta_indices

from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC2701

if TYPE_CHECKING:
    from physicalai.data import DataModule
    from physicalai.policies.base.policy import Policy


def _set_dataset_delta_indices(dataset: Any, delta_indices: dict[str, list[int]]) -> None:  # noqa: ANN401
    """Apply delta indices to both wrapped and raw LeRobot datasets."""
    if isinstance(dataset, _LeRobotDatasetAdapter):
        dataset.delta_indices = delta_indices
        return

    reader = getattr(dataset, "reader", None)
    if reader is not None:
        reader.delta_indices = delta_indices
        return

    dataset.delta_indices = delta_indices


def _get_lerobot_delta_source(policy: Policy) -> Any:  # noqa: ANN401
    """Return the initialized LeRobot policy or its pending config."""
    policy_dict = getattr(policy, "__dict__", {})

    lerobot_policy = policy_dict.get("_lerobot_policy")
    if lerobot_policy is not None:
        return lerobot_policy

    config = policy_dict.get("_config") or policy_dict.get("_provided_config")
    if config is not None:
        return config

    policy_name = getattr(policy, "policy_name", None)
    policy_config = policy_dict.get("_policy_config")
    if isinstance(policy_name, str) and isinstance(policy_config, dict):
        from lerobot.policies.factory import make_policy_config  # noqa: PLC0415

        clean_policy_config = {k: v for k, v in policy_config.items() if k != "dataset_stats"}
        return make_policy_config(policy_name, **clean_policy_config)

    return getattr(policy, "model", None)


def _get_delta_indices(model: Any, attr_name: str) -> list[int] | None:  # noqa: ANN401
    """Get delta indices from a model, handling both first-party and LeRobot policies.

    Args:
        model: The model to extract delta indices from
        attr_name: Name of the delta indices attribute (e.g., 'observation_delta_indices')

    Returns:
        List of delta indices or None if not available/not needed.
    """
    value = getattr(model, attr_name, None)
    if value is not None:
        return value

    config = getattr(model, "config", None)
    if config is not None:
        value = getattr(config, attr_name, None)
        if value is not None:
            return value
    else:
        config = model

    if attr_name == "observation_delta_indices":
        n_steps = getattr(config, "n_obs_steps", None)
        if isinstance(n_steps, int):
            return list(range(-n_steps + 1, 1)) if n_steps > 1 else None
    if attr_name == "action_delta_indices":
        n_steps = getattr(config, "n_action_steps", None)
        if isinstance(n_steps, int):
            return list(range(n_steps)) if n_steps > 0 else None

    return None


def reformat_dataset_to_match_policy(policy: Policy, datamodule: DataModule) -> None:
    """Reformat dataset to have correct deltas and parametrs depending on policy."""
    # if lerobot dataset, set delta timesteps correctly
    # https://github.com/huggingface/lerobot/blob/33cad37054c2b594ceba57463e8f11ee374fa93c/src/lerobot/datasets/factory.py#L37
    datasets = [datamodule.train_dataset]
    if getattr(datamodule, "val_eval_dataset", None) is not None:
        datasets.append(datamodule.val_eval_dataset)

    for lerobot_dataset in datasets:
        if isinstance(lerobot_dataset, _LeRobotDatasetAdapter):
            raw_features = lerobot_dataset.raw_features
            fps = lerobot_dataset.fps
            tolerance_s = lerobot_dataset.tolerance_s
        elif hasattr(lerobot_dataset, "features") and (
            hasattr(lerobot_dataset, "reader") or hasattr(lerobot_dataset, "delta_indices")
        ):
            raw_features = lerobot_dataset.features
            fps = lerobot_dataset.fps
            tolerance_s = lerobot_dataset.tolerance_s
        else:
            continue

        delta_timestamps = {}

        lerobot_model = _get_lerobot_delta_source(policy)

        for key in raw_features:
            reward_delta_indices = _get_delta_indices(lerobot_model, "reward_delta_indices")
            if key == "next.reward" and reward_delta_indices is not None:
                delta_timestamps[key] = [i / fps for i in reward_delta_indices]

            action_delta_indices = _get_delta_indices(lerobot_model, "action_delta_indices")
            if key == "action" and action_delta_indices is not None:
                delta_timestamps[key] = [i / fps for i in action_delta_indices]

            observation_delta_indices = _get_delta_indices(lerobot_model, "observation_delta_indices")
            if key.startswith("observation.") and observation_delta_indices is not None:
                delta_timestamps[key] = [i / fps for i in observation_delta_indices]

        # in place change the lerobot dataset
        if delta_timestamps:
            check_delta_timestamps(delta_timestamps, fps, tolerance_s)
            _set_dataset_delta_indices(lerobot_dataset, get_delta_indices(delta_timestamps, fps))
