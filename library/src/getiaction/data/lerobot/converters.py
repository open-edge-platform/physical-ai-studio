# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Format conversion utilities for LeRobot data formats."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from getiaction.data.observation import Observation

if TYPE_CHECKING:
    import torch


class DataFormat(StrEnum):
    """Supported data formats for LeRobot datasets."""

    GETIACTION = "getiaction"
    LEROBOT = "lerobot"


def _collect_field(
    item: dict,
    base_key: str,
    prefix: str | None = None,
) -> tuple[dict[str, torch.Tensor] | torch.Tensor | None, set[str]]:
    """Collect fields from `item` based on `base_key` and `prefix`.

    Returns:
        - Either a single `torch.Tensor`, a `dict`, or `None`
        - The set of keys that were consumed
    """
    if prefix is None:
        prefix = base_key + "."

    collected: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()

    # exact single key
    if base_key in item:
        collected[base_key] = item[base_key]
        used_keys.add(base_key)

    # prefixed subkeys
    for key, value in item.items():
        if key.startswith(prefix):
            subkey = key.split(prefix, 1)[1]
            collected[subkey] = value
            used_keys.add(key)

    if not collected:
        return None, used_keys
    if len(collected) == 1 and base_key in collected:
        return collected[base_key], used_keys
    return collected, used_keys


def _convert_lerobot_dict_to_observation(lerobot_dict: dict) -> Observation:
    """Convert dict from LeRobot format to our internal Observation format.

    Args:
        lerobot_dict: Dictionary in LeRobot format with flattened keys.

    Returns:
        Observation: The observation in our internal format.

    Raises:
        ValueError: If the item is missing a required key.
    """
    required_keys = [
        "episode_index",
        "frame_index",
        "index",
        "task_index",
        "timestamp",
    ]
    lerobot_item_keys = lerobot_dict.keys()
    for key in required_keys:
        if key not in lerobot_item_keys:
            msg = f"Missing required key: {key}. Available keys: {lerobot_item_keys}"
            raise ValueError(msg)

    if not any(k.startswith("observation") for k in lerobot_item_keys):
        msg = f"Sample must contain some form of observation. Sample keys {lerobot_item_keys}"
        raise ValueError(msg)
    if not any(k.startswith("action") for k in lerobot_item_keys):
        msg = f"Sample must contain an action. Sample keys {lerobot_item_keys}"
        raise ValueError(msg)

    used_keys: set[str] = set()

    # Observation images
    images, used = _collect_field(lerobot_dict, "observation.image", "observation.images.")
    used_keys |= used

    # Observation states
    state, used = _collect_field(lerobot_dict, "observation.state", "observation.state.")
    used_keys |= used

    # Actions
    action, used = _collect_field(lerobot_dict, "action", "action.")
    used_keys |= used

    # Tasks
    task, used = _collect_field(lerobot_dict, "task", "task.")
    used_keys |= used

    # Extra keys
    reserved = set(required_keys) | used_keys
    extra = {k: v for k, v in lerobot_dict.items() if k not in reserved}

    return Observation(
        images=cast("Any", images) if images is not None else {},
        state=cast("Any", state) if state is not None else None,
        action=cast("Any", action) if action is not None else None,
        task=cast("Any", task) if task is not None else None,
        episode_index=lerobot_dict["episode_index"],
        frame_index=lerobot_dict["frame_index"],
        index=lerobot_dict["index"],
        task_index=lerobot_dict["task_index"],
        timestamp=lerobot_dict["timestamp"],
        extra=extra,
    )


def _convert_observation_to_lerobot_dict(observation: Observation) -> dict[str, Any]:  # noqa: PLR0912
    """Convert our internal Observation format to LeRobot dict format.

    This function performs zero-copy conversion where possible by creating views
    of existing tensors rather than copying data.

    Args:
        observation: Observation in our internal format.

    Returns:
        dict: Dictionary in LeRobot format with flattened keys.

    Example:
        >>> obs = Observation(
        ...     images={"top": top_tensor, "wrist": wrist_tensor},
        ...     state=state_tensor,
        ...     action=action_tensor,
        ...     ...
        ... )

        >>> lerobot_dict = _convert_observation_to_lerobot_dict(obs)
        >>> # lerobot_dict = {
        >>> #     "observation.images.top": top_tensor,
        >>> #     "observation.images.wrist": wrist_tensor,
        >>> #     "observation.state": state_tensor,
        >>> #     "action": action_tensor,
        >>> #     ...
        >>> # }
    """
    batch: dict[str, Any] = {}

    # Handle images: dict[camera_name, tensor] -> "observation.images.{camera}"
    if observation.images is not None:
        if isinstance(observation.images, dict):
            for camera_name, img_tensor in observation.images.items():
                batch[f"observation.images.{camera_name}"] = img_tensor
        else:
            # Single image tensor without camera name
            batch["observation.image"] = observation.images

    # Handle state
    if observation.state is not None:
        batch["observation.state"] = observation.state

    # Handle action (can be single tensor or dict of action components)
    if observation.action is not None:
        if isinstance(observation.action, dict):
            for action_key, action_tensor in observation.action.items():
                batch[f"action.{action_key}"] = action_tensor
        else:
            batch["action"] = observation.action

    # Handle task
    if observation.task is not None:
        if isinstance(observation.task, dict):
            for task_key, task_tensor in observation.task.items():
                batch[f"task.{task_key}"] = task_tensor
        else:
            batch["task"] = observation.task

    # Add metadata (required fields)
    batch["episode_index"] = observation.episode_index
    batch["frame_index"] = observation.frame_index
    batch["index"] = observation.index
    batch["task_index"] = observation.task_index
    batch["timestamp"] = observation.timestamp

    # Add extra fields
    if observation.extra:
        batch.update(observation.extra)

    return batch


def _adjust_action_horizon(action: torch.Tensor, expected_horizon: int) -> torch.Tensor:
    """Adjust action tensor to match expected horizon.

    Args:
        action: Action tensor of shape (batch, action_horizon, action_dim)
        expected_horizon: Target horizon size

    Returns:
        Adjusted action tensor of shape (batch, expected_horizon, action_dim)
    """
    import torch  # noqa: PLC0415

    current_horizon = action.shape[1]
    if current_horizon < expected_horizon:
        # Pad with zeros
        pad_size = expected_horizon - current_horizon
        pad_shape = (action.shape[0], pad_size, action.shape[2])
        pad_values = torch.zeros(pad_shape, dtype=action.dtype, device=action.device)
        return torch.cat([action, pad_values], dim=1)
    # Trim to expected horizon
    return action[:, :expected_horizon, :]


def _adjust_action_is_pad(
    action_is_pad: torch.Tensor,
    expected_horizon: int,
    batch_size: int,
) -> torch.Tensor:
    """Adjust action_is_pad tensor to match expected horizon.

    Args:
        action_is_pad: Boolean tensor of shape (batch, action_horizon)
        expected_horizon: Target horizon size
        batch_size: Batch size for creating padding

    Returns:
        Adjusted action_is_pad tensor of shape (batch, expected_horizon)
    """
    import torch  # noqa: PLC0415

    current_horizon = action_is_pad.shape[1]
    if current_horizon < expected_horizon:
        # Pad with True (indicating these are padded actions)
        pad_size = expected_horizon - current_horizon
        pad_values = torch.ones(
            (batch_size, pad_size),
            dtype=torch.bool,
            device=action_is_pad.device,
        )
        return torch.cat([action_is_pad, pad_values], dim=1)
    # Trim to expected horizon
    return action_is_pad[:, :expected_horizon]


def _flatten_collated_dict_to_lerobot(collated_dict: dict[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912
    """Flatten a collated dict (from _collate_observations) to LeRobot format.

    This handles the output from the datamodule's collate function, which returns
    a dict with nested structure like {"images": {"top": tensor, "wrist": tensor}},
    and converts it to LeRobot's flattened format with keys like "observation.images.top".

    Args:
        collated_dict: Dictionary with nested structure from collate function.

    Returns:
        Dictionary in LeRobot format with flattened keys.

    Example:
        >>> collated = {
        ...     "images": {"top": top_tensor, "wrist": wrist_tensor},
        ...     "state": state_tensor,
        ...     "action": action_tensor,
        ...     ...
        ... }

        >>> lerobot = _flatten_collated_dict_to_lerobot(collated)
        >>> # lerobot = {
        >>> #     "observation.images.top": top_tensor,
        >>> #     "observation.images.wrist": wrist_tensor,
        >>> #     "observation.state": state_tensor,
        >>> #     "action": action_tensor,
        >>> #     ...
        >>> # }
    """
    result: dict[str, Any] = {}

    for key, value in collated_dict.items():
        if key == "images":
            # Handle images: dict[camera_name, tensor] -> "observation.images.{camera}"
            if isinstance(value, dict):
                for camera_name, img_tensor in value.items():
                    result[f"observation.images.{camera_name}"] = img_tensor
            elif value is not None:
                # Single image tensor
                result["observation.image"] = value

        elif key == "state":
            # Handle state
            if value is not None:
                result["observation.state"] = value

        elif key == "action":
            # Handle action (can be single tensor or dict of action components)
            if isinstance(value, dict):
                for action_key, action_tensor in value.items():
                    result[f"action.{action_key}"] = action_tensor
            elif value is not None:
                result["action"] = value

        elif key == "task":
            # Handle task
            if isinstance(value, dict):
                for task_key, task_tensor in value.items():
                    result[f"task.{task_key}"] = task_tensor
            elif value is not None and not isinstance(value, list):
                # Skip if it's a list (not tensors)
                result["task"] = value

        elif key == "extra":
            # Flatten extra fields
            if isinstance(value, dict):
                result.update(value)

        elif key in {"episode_index", "frame_index", "index", "task_index", "timestamp"}:
            # Copy metadata fields directly
            result[key] = value

        # Skip other fields like "info", "next_reward", "next_success" which are not part of LeRobot format

    return result


class FormatConverter:
    """Convert between GetiAction Observation and LeRobot dict formats.

    This class provides bidirectional format conversion with zero-copy operations:
    - GetiAction uses structured `Observation` dataclasses
    - LeRobot uses flattened dictionaries with dot-notation keys

    The converter automatically detects input format and handles:
    - Observation objects
    - Collated nested dicts
    - Already-formatted LeRobot dicts
    - Temporal dimension adjustments (n_obs_steps)
    - Action horizon adjustments (policy-specific)

    Example:
        >>> # Convert to LeRobot format
        >>> lerobot_batch = FormatConverter.to_lerobot_dict(observation)

        >>> # Convert with policy config for horizon/temporal handling
        >>> lerobot_batch = FormatConverter.to_lerobot_dict(
        ...     observation, 
        ...     policy_config=policy.config
        ... )

        >>> # Convert to Observation format
        >>> observation = FormatConverter.to_observation(lerobot_dict)
    """

    @staticmethod
    def to_lerobot_dict(
        batch: dict[str, Any] | Observation,
        policy_config: Any | None = None,
    ) -> dict[str, Any]:
        """Convert batch to LeRobot dictionary format.

        This method handles conversion from either:
        1. getiaction Observation objects -> LeRobot dict (with flattened keys like "observation.images.top")
        2. Collated dicts from datamodule -> LeRobot dict (flatten nested structure)
        3. Already-formatted LeRobot dicts -> pass through unchanged

        Additionally handles policy-specific requirements if policy_config is provided:
        - Temporal dimension adjustment (n_obs_steps)
        - Action horizon adjustment
        - Action padding mask adjustment

        Args:
            batch: Either an Observation object, a collated dict, or a dictionary in LeRobot format.
            policy_config: Optional policy configuration for dimension adjustments.
                If provided, ensures tensors match policy expectations:
                - observation.state temporal dimension
                - action horizon length
                - action_is_pad mask length

        Returns:
            Dictionary in LeRobot format with flattened keys like "observation.images.top".

        Example:
            >>> # Basic conversion
            >>> obs = Observation(images={"top": top_img}, state=state, action=action)
            >>> lerobot_dict = FormatConverter.to_lerobot_dict(obs)

            >>> # With policy config (handles temporal dims and horizons)
            >>> lerobot_dict = FormatConverter.to_lerobot_dict(obs, policy_config=policy.config)
        """
        import torch  # noqa: PLC0415

        # Step 1: Convert to base LeRobot dict format
        if isinstance(batch, Observation):
            result = _convert_observation_to_lerobot_dict(batch)
        elif isinstance(batch, dict):
            # Check if already in LeRobot format
            if any(key.startswith("observation.") for key in batch):
                result = batch
            # Check if it's a collated dict
            elif "images" in batch or "state" in batch:
                result = _flatten_collated_dict_to_lerobot(batch)
            else:
                result = batch
        else:
            result = batch

        # Step 2: Apply policy-specific adjustments if config provided
        if policy_config is not None and isinstance(result, dict):
            # Handle temporal dimension for observation.state
            # LeRobot expects (batch, n_obs_steps, state_dim)
            if "observation.state" in result:
                state = result["observation.state"]
                if isinstance(state, torch.Tensor) and state.ndim == 2:  # noqa: PLR2004
                    # Add temporal dimension: (batch, state_dim) -> (batch, n_obs_steps, state_dim)
                    # For n_obs_steps=1, shape becomes (batch, 1, state_dim)
                    state = state.unsqueeze(1)
                    result["observation.state"] = state

            # Handle action horizon adjustment
            if "action" in result and hasattr(policy_config, "horizon"):
                action = result["action"]
                if isinstance(action, torch.Tensor) and action.ndim == 3:  # noqa: PLR2004
                    batch_size, action_horizon, action_dim = action.shape
                    expected_horizon = policy_config.horizon

                    if action_horizon != expected_horizon:
                        # Adjust action horizon
                        result["action"] = _adjust_action_horizon(action, expected_horizon)

                        # Adjust action_is_pad if it exists
                        if "action_is_pad" in result:
                            result["action_is_pad"] = _adjust_action_is_pad(
                                result["action_is_pad"],
                                expected_horizon,
                                batch_size,
                            )

        return result

    @staticmethod
    def to_observation(batch: dict[str, Any] | Observation) -> Observation:
        """Convert batch to Observation format.

        If the batch is already an Observation, returns it unchanged (no-op).
        If the batch is a dict, converts it from LeRobot format to Observation.

        Args:
            batch: Either an Observation or a dict in LeRobot format.

        Returns:
            Observation dataclass with structured fields.

        Example:
            >>> # Already in correct format - returns unchanged
            >>> observation = FormatConverter.to_observation(observation)

            >>> # Converts from LeRobot dict
            >>> observation = FormatConverter.to_observation(lerobot_dict)
        """
        if isinstance(batch, Observation):
            return batch  # Already in Observation format
        return _convert_lerobot_dict_to_observation(batch)


__all__ = ["DataFormat", "FormatConverter"]
