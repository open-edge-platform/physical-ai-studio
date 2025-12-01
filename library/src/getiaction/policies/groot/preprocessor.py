# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""First-party preprocessor for Groot policy.

This module provides preprocessing functionality for the Groot policy without
requiring LeRobot dependencies. It handles:

- State normalization (min-max to [-1, 1])
- Action normalization (min-max to [-1, 1])
- State/action padding to max dimensions
- Video encoding via EagleProcessor
- Embodiment ID mapping

The preprocessing pipeline transforms LeRobot-style batch dicts into the format
expected by GrootModel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image

from getiaction.data.observation import Observation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from getiaction.policies.groot.components import EagleProcessor as EagleProcessorType

logger = logging.getLogger(__name__)

# Default embodiment mapping (matches NVIDIA's EMBODIMENT_TAG_MAPPING)
DEFAULT_EMBODIMENT_MAPPING: dict[str, int] = {
    "new_embodiment": 31,
    "oxe_droid": 17,
    "agibot_genie1": 26,
    "gr1": 24,
    "so100": 2,
    "unitree_g1": 3,
}


@dataclass
class GrootPreprocessor:
    """Preprocessor for Groot policy inputs.

    Transforms LeRobot-style batch dicts into the format expected by GrootModel:
    1. Normalizes state/action to [-1, 1] using min-max normalization
    2. Pads state/action to max dimensions
    3. Encodes images + text with EagleProcessor
    4. Adds embodiment ID

    Args:
        max_state_dim: Maximum state dimension (shorter states are zero-padded).
        max_action_dim: Maximum action dimension (shorter actions are zero-padded).
        action_horizon: Number of action steps (default 16 for GR00T).
        embodiment_tag: Embodiment identifier for this robot.
        normalize_min_max: Whether to apply min-max normalization.
        stats: Dataset statistics for normalization {key: {min, max}}.
        eagle_processor_repo: HuggingFace repo for Eagle processor.
        device: Target device for tensors.

    Examples:
        >>> preprocessor = GrootPreprocessor(
        ...     max_state_dim=64,
        ...     max_action_dim=32,
        ...     stats=dataset.meta.stats,
        ... )
        >>> batch = preprocessor(raw_batch)
    """

    max_state_dim: int = 64
    max_action_dim: int = 32
    action_horizon: int = 16
    embodiment_tag: str = "new_embodiment"
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None
    eagle_processor_repo: str = "lerobot/eagle2hg-processor-groot-n1p5"
    device: str = "cuda"

    # Internal cache for EagleProcessor
    _eagle_processor: Any = field(default=None, init=False, repr=False)

    @property
    def eagle_processor(self) -> EagleProcessorType:
        """Lazy-load Eagle processor."""
        if self._eagle_processor is None:
            from getiaction.policies.groot.components import EagleProcessor  # noqa: PLC0415

            self._eagle_processor = EagleProcessor(processor_repo=self.eagle_processor_repo)
        return self._eagle_processor  # type: ignore[return-value]

    def __call__(self, batch: Mapping[str, Any] | Observation) -> dict[str, torch.Tensor]:
        """Preprocess a batch for GrootModel.

        Args:
            batch: Input batch, either as:
                - An Observation dataclass (from getiaction DataModule)
                - A dict in LeRobot format with keys like:
                    - observation.state: (B, D) or (B, T, D)
                    - observation.images.*: (B, C, H, W) image tensors
                    - action: (B, D) or (B, T, D) action tensors
                    - task: str or list[str] task description

        Returns:
            Preprocessed batch with keys:
                - state: (B, 1, max_state_dim)
                - state_mask: (B, 1, max_state_dim)
                - action: (B, action_horizon, max_action_dim)
                - action_mask: (B, action_horizon, max_action_dim)
                - embodiment_id: (B,)
                - eagle_*: Encoded vision-language tensors
        """
        result: dict[str, Any] = {}

        # Convert Observation to dict if needed
        if isinstance(batch, Observation):
            batch_dict = batch.to_dict(flatten=True)
        else:
            batch_dict = dict(batch)

        # Infer batch size and device
        batch_size, device = self._infer_batch_info(batch_dict)

        # 1. Process state (support both "observation.state" and "state" keys)
        state_tensor = batch_dict.get("observation.state") or batch_dict.get("state")
        if state_tensor is not None:
            state, state_mask = self._process_state(state_tensor)
            result["state"] = state
            result["state_mask"] = state_mask

        # 2. Process action (for training)
        if "action" in batch_dict and batch_dict["action"] is not None:
            action, action_mask = self._process_action(batch_dict["action"])
            result["action"] = action
            result["action_mask"] = action_mask

        # 3. Add embodiment ID
        emb_id = DEFAULT_EMBODIMENT_MAPPING.get(self.embodiment_tag, 31)
        result["embodiment_id"] = torch.full((batch_size,), emb_id, dtype=torch.long, device=device)

        # 4. Process images with Eagle
        eagle_inputs = self._process_images(batch_dict)
        result.update(eagle_inputs)

        # 5. Move to target device
        return self._to_device(result)

    @staticmethod
    def _infer_batch_info(batch: dict[str, Any]) -> tuple[int, torch.device]:
        """Infer batch size and device from batch tensors.

        Args:
            batch: Input batch.

        Returns:
            Tuple of (batch_size, device).
        """
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0], value.device
        return 1, torch.device("cpu")

    def _process_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process state: normalize and pad.

        Args:
            state: State tensor (B, D) or (B, T, D).

        Returns:
            Tuple of (padded_state, state_mask).
        """
        # Tensor dimension constants
        tensor_2d = 2

        # Ensure (B, T, D) format
        if state.dim() == tensor_2d:
            state = state.unsqueeze(1)  # (B, D) -> (B, 1, D)

        batch_size, _t, orig_dim = state.shape

        # Normalize
        if self.normalize_min_max:
            state = self._min_max_normalize(state, "observation.state")

        # Pad to max_state_dim
        if orig_dim < self.max_state_dim:
            padding = torch.zeros(
                batch_size,
                1,
                self.max_state_dim - orig_dim,
                dtype=state.dtype,
                device=state.device,
            )
            state = torch.cat([state, padding], dim=-1)
        elif orig_dim > self.max_state_dim:
            state = state[..., : self.max_state_dim]
            orig_dim = self.max_state_dim

        # Create mask
        state_mask = torch.zeros(
            batch_size,
            1,
            self.max_state_dim,
            dtype=torch.bool,
            device=state.device,
        )
        state_mask[..., :orig_dim] = True

        return state, state_mask

    def _process_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process action: normalize, expand horizon, and pad.

        Args:
            action: Action tensor (B, D) or (B, T, D).

        Returns:
            Tuple of (padded_action, action_mask).
        """
        # Tensor dimension constants
        tensor_2d = 2
        tensor_3d = 3

        # Ensure (B, T, D) format
        if action.dim() == tensor_2d:
            # Single timestep - replicate to action_horizon
            action = action.unsqueeze(1).repeat(1, self.action_horizon, 1)
        elif action.dim() == tensor_3d:
            batch_size, t, _d = action.shape
            if t < self.action_horizon:
                # Pad by repeating last action
                last = action[:, -1:, :]
                padding = last.repeat(1, self.action_horizon - t, 1)
                action = torch.cat([action, padding], dim=1)
            elif t > self.action_horizon:
                action = action[:, : self.action_horizon, :]

        batch_size, horizon, orig_dim = action.shape

        # Normalize
        if self.normalize_min_max:
            # Flatten for normalization, then reshape back
            flat = action.reshape(batch_size * horizon, orig_dim)
            flat = self._min_max_normalize(flat, "action")
            action = flat.view(batch_size, horizon, orig_dim)

        # Pad to max_action_dim
        if orig_dim < self.max_action_dim:
            padding = torch.zeros(
                batch_size,
                horizon,
                self.max_action_dim - orig_dim,
                dtype=action.dtype,
                device=action.device,
            )
            action = torch.cat([action, padding], dim=-1)
        elif orig_dim > self.max_action_dim:
            action = action[..., : self.max_action_dim]
            orig_dim = self.max_action_dim

        # Create mask
        action_mask = torch.zeros(
            batch_size,
            horizon,
            self.max_action_dim,
            dtype=torch.bool,
            device=action.device,
        )
        action_mask[..., :orig_dim] = True

        return action, action_mask

    def _min_max_normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """Apply min-max normalization to [-1, 1].

        Args:
            x: Input tensor.
            key: Feature key for stats lookup.

        Returns:
            Normalized tensor.
        """
        if self.stats is None or key not in self.stats:
            return x

        stats = self.stats[key]
        last_dim = x.shape[-1]

        min_val = self._align_stats(stats.get("min", torch.zeros(last_dim)), last_dim, x.device)
        max_val = self._align_stats(stats.get("max", torch.ones(last_dim)), last_dim, x.device)

        denom = max_val - min_val
        # Avoid division by zero
        mask = denom != 0
        safe_denom = torch.where(mask, denom, torch.ones_like(denom))

        # Map to [-1, 1]
        normalized = 2 * (x - min_val) / safe_denom - 1
        return torch.where(mask, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _align_stats(
        stat: torch.Tensor | np.ndarray | list,
        target_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Align stats tensor to target dimension.

        Args:
            stat: Statistic value (tensor, array, or list).
            target_dim: Target dimension.
            device: Target device.

        Returns:
            Aligned tensor on correct device.
        """
        t = torch.as_tensor(stat, dtype=torch.float32, device=device).flatten()
        current_dim = t.shape[0]

        if current_dim == target_dim:
            return t
        if current_dim < target_dim:
            # Pad with zeros
            padding = torch.zeros(target_dim - current_dim, dtype=t.dtype, device=device)
            return torch.cat([t, padding])
        return t[:target_dim]

    def _process_images(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process images with Eagle processor.

        Args:
            batch: Input batch with image keys.

        Returns:
            Dict with eagle_* prefixed tensors.
        """
        # Find image keys - support multiple formats:
        # 1. "observation.images.*" (LeRobot format)
        # 2. "images.*" (Observation format with multiple cameras)
        # 3. "observation.image" (single image)
        # 4. "images" (single direct tensor)
        img_keys = sorted([k for k in batch if k.startswith("observation.images.")])
        if not img_keys:
            img_keys = sorted([k for k in batch if k.startswith("images.") and k != "images"])
        if not img_keys and "observation.image" in batch:
            img_keys = ["observation.image"]
        if not img_keys and "images" in batch and isinstance(batch["images"], torch.Tensor):
            img_keys = ["images"]

        if not img_keys:
            return {}

        # Get task description
        task = batch.get("task", "Perform the task.")
        if isinstance(task, list):
            task = task[0] if task else "Perform the task."

        # Convert tensors to PIL images
        batch_images: list[list[Image.Image]] = []
        batch_size = batch[img_keys[0]].shape[0]

        for b in range(batch_size):
            images = []
            for key in img_keys:
                img_tensor = batch[key][b]  # (C, H, W)
                # Convert to uint8 numpy
                if img_tensor.dtype.is_floating_point:
                    img_np = (img_tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                else:
                    img_np = img_tensor.cpu().numpy()
                # (C, H, W) -> (H, W, C)
                img_np = np.transpose(img_np, (1, 2, 0))
                images.append(Image.fromarray(img_np))
            batch_images.append(images)

        # Encode with Eagle
        batch_text = [task] * batch_size
        return self.eagle_processor.batch_encode(batch_images, batch_text)

    def _to_device(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Move batch to target device.

        Args:
            batch: Input batch.

        Returns:
            Batch with tensors on target device.
        """
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result


@dataclass
class GrootPostprocessor:
    """Postprocessor for Groot policy outputs.

    Transforms GrootModel outputs back to original action space:
    1. Slices action to environment dimension
    2. Denormalizes from [-1, 1] to original range

    Args:
        env_action_dim: Original action dimension of the environment.
        normalize_min_max: Whether min-max normalization was applied.
        stats: Dataset statistics for denormalization {key: {min, max}}.
        device: Target device for output.

    Examples:
        >>> postprocessor = GrootPostprocessor(
        ...     env_action_dim=7,
        ...     stats=dataset.meta.stats,
        ... )
        >>> action = postprocessor(model_output["action_pred"])
    """

    env_action_dim: int = 0
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None
    device: str = "cpu"

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        """Postprocess action output.

        Args:
            action: Model output (B, T, D) or (B, D).

        Returns:
            Denormalized action in original space (B, env_action_dim).
        """
        # Tensor dimension constant
        tensor_3d = 3

        # Select last timestep if multiple
        if action.dim() == tensor_3d:
            action = action[:, -1, :]  # (B, T, D) -> (B, D)

        # Slice to env dimension
        if self.env_action_dim and action.shape[-1] >= self.env_action_dim:
            action = action[..., : self.env_action_dim]

        # Denormalize
        if self.normalize_min_max:
            action = self._min_max_denormalize(action)

        return action.to(self.device)

    def _min_max_denormalize(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from [-1, 1] to original range.

        Args:
            action: Normalized action tensor.

        Returns:
            Denormalized action.
        """
        if self.stats is None or "action" not in self.stats:
            return action

        stats = self.stats["action"]
        d = action.shape[-1]

        min_val = torch.as_tensor(
            stats.get("min", torch.zeros(d)),
            dtype=action.dtype,
            device=action.device,
        ).flatten()[:d]
        max_val = torch.as_tensor(
            stats.get("max", torch.ones(d)),
            dtype=action.dtype,
            device=action.device,
        ).flatten()[:d]

        # Pad if needed
        if min_val.shape[0] < d:
            padding = torch.zeros(d - min_val.shape[0], dtype=min_val.dtype, device=min_val.device)
            min_val = torch.cat([min_val, padding])
            max_val = torch.cat([max_val, padding])

        denom = max_val - min_val
        mask = denom != 0
        safe_denom = torch.where(mask, denom, torch.ones_like(denom))

        # Inverse of min-max normalization: x = (normalized + 1) / 2 * denom + min
        denormalized = (action + 1.0) * 0.5 * safe_denom + min_val
        return torch.where(mask, denormalized, min_val)


def make_groot_preprocessors(
    *,
    max_state_dim: int = 64,
    max_action_dim: int = 32,
    action_horizon: int = 16,
    embodiment_tag: str = "new_embodiment",
    env_action_dim: int = 0,
    stats: dict[str, dict[str, Any]] | None = None,
    eagle_processor_repo: str = "nvidia/Eagle2-2B",
    device: str = "cuda",
) -> tuple[GrootPreprocessor, GrootPostprocessor]:
    """Create preprocessor and postprocessor for Groot policy.

    Convenience factory function that creates both processors with consistent settings.

    Args:
        max_state_dim: Maximum state dimension.
        max_action_dim: Maximum action dimension.
        action_horizon: Number of action steps.
        embodiment_tag: Embodiment identifier.
        env_action_dim: Original environment action dimension.
        stats: Dataset statistics for normalization.
        eagle_processor_repo: HuggingFace repo for Eagle processor.
        device: Target device.

    Returns:
        Tuple of (preprocessor, postprocessor).

    Examples:
        >>> preprocessor, postprocessor = make_groot_preprocessors(
        ...     max_state_dim=64,
        ...     max_action_dim=32,
        ...     env_action_dim=7,
        ...     stats=dataset.meta.stats,
        ... )
        >>> batch = preprocessor(raw_batch)
        >>> output = model(batch)
        >>> action = postprocessor(output["action_pred"])
    """
    preprocessor = GrootPreprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        action_horizon=action_horizon,
        embodiment_tag=embodiment_tag,
        normalize_min_max=True,
        stats=stats,
        eagle_processor_repo=eagle_processor_repo,
        device=device,
    )

    postprocessor = GrootPostprocessor(
        env_action_dim=env_action_dim,
        normalize_min_max=True,
        stats=stats,
        device="cpu",
    )

    return preprocessor, postprocessor


__all__ = [
    "DEFAULT_EMBODIMENT_MAPPING",
    "GrootPostprocessor",
    "GrootPreprocessor",
    "make_groot_preprocessors",
]
