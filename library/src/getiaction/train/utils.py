# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Util functions for Training."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices

from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter

if TYPE_CHECKING:
    from getiaction.data import DataModule
    from getiaction.policies.base.policy import Policy


def reformat_dataset_to_match_policy(policy: Policy, datamodule: DataModule) -> None:
    """Reformat dataset to have correct deltas and parametrs depending on policy."""
    # if lerobot dataset, set delta timesteps correctly
    # https://github.com/huggingface/lerobot/blob/33cad37054c2b594ceba57463e8f11ee374fa93c/src/lerobot/datasets/factory.py#L37
    if isinstance(datamodule.train_dataset, _LeRobotDatasetAdapter):
        delta_timestamps = {}
        lerobot_dataset = datamodule.train_dataset
        for key in lerobot_dataset.raw_features:
            if key == "next.reward" and policy.model.reward_delta_indices is not None:
                delta_timestamps[key] = [i / lerobot_dataset.fps for i in policy.model.reward_delta_indices]
            if key == "action" and policy.model.action_delta_indices is not None:
                delta_timestamps[key] = [i / lerobot_dataset.fps for i in policy.model.action_delta_indices]
            if key.startswith("observation.") and policy.model.observation_delta_indices is not None:
                delta_timestamps[key] = [i / lerobot_dataset.fps for i in policy.model.observation_delta_indices]
        # in place change the lerobot dataset
        if delta_timestamps:
            check_delta_timestamps(delta_timestamps, lerobot_dataset.fps, lerobot_dataset.tolerance_s)
            lerobot_dataset.delta_indices = get_delta_indices(delta_timestamps, lerobot_dataset.fps)
