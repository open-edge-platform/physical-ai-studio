# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
LeRobotDataset standard
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from action_trainer.data import ActionDataset, Observation

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import torch


def _convert_lerobot_item_to_observation(lerobot_item: dict) -> Observation:
    """Function converts item from lerobot to our internal representation, observation.

    Expect these keys are present in sample from lerobot:
        - an observation either image or state or both (tensor)
            - if it has images we assume LeRobot auto converts to CHW format
        - action (tensor)
        - task (str)
        - episode_index (tensor[int])
        - frame_index (tensor[int])
        - index (tensor[int])
        - task_index (tensor[int])
        - timestamp (tensor[float])

    Args:
        lerobot_item (dict): LeRobotDataset output sample from dataset.

    Returns:
        observation (Observation): Internal representation of a observation.
    Raises:
        AssertionError: If required keys are not present in the lerobot_item.
    """
    # Define the required keys for all items
    required_keys = [
        "action",
        "task",
        "episode_index",
        "frame_index",
        "index",
        "task_index",
        "timestamp",
    ]

    # Check for presence of all required keys
    lerobot_item_keys = lerobot_item.keys()
    for key in required_keys:
        assert key in lerobot_item_keys, f"Missing required key: {key}. Available keys: {lerobot_item_keys}"

    # Check if any observation keys are present
    has_image = any(key.startswith("observation.images.") for key in lerobot_item_keys)
    has_image_single = "observation.image" in lerobot_item_keys
    has_state = "observation.state" in lerobot_item_keys

    assert has_image or has_image_single or has_state, (
        f"Sample must contain some form of observation. Sample keys {lerobot_item_keys}"
    )

    # Process observations
    images: dict[str, torch.Tensor] = {}
    if has_image_single:
        # Handle the case with a single 'observation.image' key
        images["image"] = lerobot_item["observation.image"]
    elif has_image:
        # Handle multiple images, e.g., 'observation.images.image' and 'observation.images.wrist_image'
        for key, value in lerobot_item.items():
            if key.startswith("observation.images."):
                camera_name = key.split("observation.images.")[1]
                images[camera_name] = value

    state = lerobot_item.get("observation.state")

    # Create and return the LeRobotObservation object
    return Observation(
        images=images,
        state=state,
        action=lerobot_item.get("action"),
        task=lerobot_item.get("task"),
        episode_index=lerobot_item.get("episode_index"),
        frame_index=lerobot_item.get("frame_index"),
        index=lerobot_item.get("index"),
        task_index=lerobot_item.get("task_index"),
        timestamp=lerobot_item.get("timestamp"),
    )


# NOTE: Eventually we will need properties from action and lerobot datasets
class LeRobotActionDataset(ActionDataset):
    """A wrapper class that enables using a LeRobotDataset for action training.

    This class uses the **composition** pattern, holding a `LeRobotDataset` instance
    and delegating all data loading operations to it. This design allows for
    the addition of action-specific logic without modifying the original
    `LeRobotDataset` class.

    Args:
        repo_id (str): The Hugging Face Hub repository ID.
        root (str | Path | None, optional): Local directory for caching files. Defaults to None.
        episodes (list[int] | None, optional): A list of episode indices to load. Defaults to None.
        image_transforms (Callable | None, optional): A callable to apply to image data. Defaults to None.
        delta_timestamps (dict[str, list[float]] | None, optional): A dict of timestamps. Defaults to None.
        tolerance_s (float, optional): Tolerance in seconds for timestamp synchronization. Defaults to 1e-4.
        revision (str | None, optional): A Git revision ID for the dataset. Defaults to None.
        force_cache_sync (bool, optional): Flag to force synchronization with the Hugging Face Hub. Defaults to False.
        download_videos (bool, optional): Flag to download video files. Defaults to True.
        video_backend (str | None, optional): The video decoding backend to use. Defaults to None.
        batch_encoding_size (int, optional): The number of episodes to encode in a batch. Defaults to 1.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ):
        super().__init__()

        # All arguments are passed
        self._lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )

    def __len__(self):
        return len(self._lerobot_dataset)

    def __getitem__(self, idx) -> Observation:
        return _convert_lerobot_item_to_observation(self._lerobot_dataset[idx])

    @staticmethod
    def from_lerobot(lerobot_dataset: LeRobotDataset) -> LeRobotActionDataset:
        """Creates an instance of LeRobotActionDataset from an existing LeRobotDataset instance.

        This static method is useful when you already have a `LeRobotDataset` object
        that you want to wrap for use in action training.

        Args:
            lerobot_dataset (LeRobotDataset): The existing LeRobotDataset instance to be wrapped.

        Returns:
            LeRobotActionDataset: A new LeRobotActionDataset instance that uses the provided dataset.
        """
        instance = LeRobotActionDataset.__new__(LeRobotActionDataset)
        # Bypassing __init__ to set the internal dataset
        instance._lerobot_dataset = lerobot_dataset  # noqa: SLF001
        return instance
