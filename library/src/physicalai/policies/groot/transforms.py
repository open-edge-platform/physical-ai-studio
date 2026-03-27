# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Transforms for Groot policy inputs and outputs.

This module provides nn.Module-based transforms for the Groot policy:
- `GrootPreprocessor`: Transforms Observation → GrootModel input format
- `GrootPostprocessor`: Transforms GrootModel output → original action space

These modules can be:
- Moved to device with .to(device)
- Included in model export (ONNX, TorchScript)
- Composed with the model in forward()

The preprocessing pipeline handles:
- State normalization (min-max to [-1, 1])
- Action normalization (min-max to [-1, 1])
- State/action padding to max dimensions
- Video encoding via EagleProcessor
- Embodiment ID mapping
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Observation

if TYPE_CHECKING:
    from physicalai.policies.groot.components import EagleProcessor as EagleProcessorType

logger = logging.getLogger(__name__)

# ============================================================================ #
# Groot-specific Constants                                                     #
# ============================================================================ #

# LeRobot format keys (for backward compatibility)
OBSERVATION_STATE = "observation.state"
OBSERVATION_IMAGES_PREFIX = "observation.images."
OBSERVATION_IMAGE = "observation.image"

# Transform output keys
STATE_MASK = "state_mask"
ACTION_MASK = "action_mask"
EMBODIMENT_ID = "embodiment_id"

# Stats keys
STATS_MIN = "min"
STATS_MAX = "max"

# Default dimensions
MAX_STATE_DIM = 64
MAX_ACTION_DIM = 32
ACTION_HORIZON = 16  # GR00T maximum action horizon

# Embodiment mapping (matches NVIDIA's EMBODIMENT_TAG_MAPPING)
EMBODIMENT_MAPPING: dict[str, int] = {
    "new_embodiment": 31,
    "oxe_droid": 17,
    "agibot_genie1": 26,
    "gr1": 24,
    "so100": 2,
    "unitree_g1": 3,
}
DEFAULT_EMBODIMENT_TAG = "new_embodiment"
DEFAULT_EMBODIMENT_ID = EMBODIMENT_MAPPING[DEFAULT_EMBODIMENT_TAG]

# Model defaults
DEFAULT_EAGLE_PROCESSOR_REPO = "nvidia/Eagle2-2B"


# ============================================================================ #
# Preprocessor                                                                 #
# ============================================================================ #


class GrootPreprocessor(nn.Module):
    """Preprocessor for Groot policy inputs.

    Transforms Observation inputs into the format expected by GrootModel:
    1. Normalizes state/action to [-1, 1] using min-max normalization
    2. Pads state/action to max dimensions
    3. Encodes images + text with EagleProcessor
    4. Adds embodiment ID

    Stats are registered as buffers for automatic device handling.

    Args:
        max_state_dim: Maximum state dimension (shorter states are zero-padded).
        max_action_dim: Maximum action dimension (shorter actions are zero-padded).
        action_horizon: Number of action steps (default 16 for GR00T).
        embodiment_tag: Embodiment identifier for this robot.
        normalize_min_max: Whether to apply min-max normalization.
        stats: Dataset statistics for normalization {key: {min, max}}.
        eagle_processor_repo: HuggingFace repo for Eagle processor.

    Examples:
        >>> preprocessor = GrootPreprocessor(
        ...     max_state_dim=64,
        ...     max_action_dim=32,
        ...     stats=dataset.stats,
        ... )
        >>> batch = preprocessor(observation)
    """

    def __init__(
        self,
        *,
        max_state_dim: int = MAX_STATE_DIM,
        max_action_dim: int = MAX_ACTION_DIM,
        action_horizon: int = ACTION_HORIZON,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        normalize_min_max: bool = True,
        stats: dict[str, dict[str, list[float]]] | None = None,
        eagle_processor_repo: str = DEFAULT_EAGLE_PROCESSOR_REPO,
    ) -> None:
        """Initialize preprocessor with normalization statistics."""
        super().__init__()

        # Store hyperparameters
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.embodiment_tag = embodiment_tag
        self.normalize_min_max = normalize_min_max
        self.eagle_processor_repo = eagle_processor_repo

        # Get embodiment ID
        self._embodiment_id = EMBODIMENT_MAPPING.get(embodiment_tag, DEFAULT_EMBODIMENT_ID)

        # Register stats as buffers for device handling
        self._register_stats_buffers(stats)

        # Lazy-loaded Eagle processor (not an nn.Module, loaded on demand)
        self._eagle_processor: EagleProcessorType | None = None

    def _register_stats_buffers(self, stats: dict[str, dict[str, list[float]]] | None) -> None:
        """Register normalization statistics as buffers.

        Args:
            stats: Dataset statistics for normalization.
        """
        # State stats
        if stats is not None and OBSERVATION_STATE in stats:
            state_stats = stats[OBSERVATION_STATE]
            state_min = self._to_tensor(state_stats.get(STATS_MIN), self.max_state_dim)
            state_max = self._to_tensor(state_stats.get(STATS_MAX), self.max_state_dim)
        else:
            state_min = torch.zeros(self.max_state_dim)
            state_max = torch.ones(self.max_state_dim)

        self.register_buffer("state_min", state_min)
        self.register_buffer("state_max", state_max)

        # Action stats
        if stats is not None and ACTION in stats:
            action_stats = stats[ACTION]
            action_min = self._to_tensor(action_stats.get(STATS_MIN), self.max_action_dim)
            action_max = self._to_tensor(action_stats.get(STATS_MAX), self.max_action_dim)
        else:
            action_min = torch.zeros(self.max_action_dim)
            action_max = torch.ones(self.max_action_dim)

        self.register_buffer("action_min", action_min)
        self.register_buffer("action_max", action_max)

    @staticmethod
    def _to_tensor(value: list[float] | torch.Tensor | None, target_dim: int) -> torch.Tensor:
        """Convert value to tensor and pad/truncate to target dimension.

        Args:
            value: Value to convert (tensor, list, or None).
            target_dim: Target dimension.

        Returns:
            Tensor of shape (target_dim,).
        """
        if value is None:
            return torch.zeros(target_dim)

        t = torch.as_tensor(value, dtype=torch.float32).flatten()
        current_dim = t.shape[0]

        if current_dim == target_dim:
            return t
        if current_dim < target_dim:
            padding = torch.zeros(target_dim - current_dim, dtype=t.dtype)
            return torch.cat([t, padding])
        return t[:target_dim]

    @property
    def eagle_processor(self) -> EagleProcessorType:
        """Lazy-load Eagle processor."""
        if self._eagle_processor is None:
            from physicalai.policies.groot.components import EagleProcessor  # noqa: PLC0415

            self._eagle_processor = EagleProcessor(processor_repo=self.eagle_processor_repo)
        return self._eagle_processor

    def forward(self, batch: Observation | dict[str, Any]) -> dict[str, torch.Tensor]:
        """Preprocess a batch for GrootModel.

        Args:
            batch: Input batch as Observation or dict with keys:
                - state: (B, D) or (B, T, D) state tensor
                - images: dict of (B, C, H, W) image tensors
                - action: (B, D) or (B, T, D) action tensors (optional, for training)
                - task: str or list[str] task description

        Returns:
            Preprocessed batch with keys:
                - state: (B, 1, max_state_dim)
                - state_mask: (B, 1, max_state_dim)
                - action: (B, action_horizon, max_action_dim) (if input has action)
                - action_mask: (B, action_horizon, max_action_dim) (if input has action)
                - embodiment_id: (B,)
                - eagle_*: Encoded vision-language tensors
        """
        result: dict[str, Any] = {}

        # Convert Observation to flattened dict if needed
        batch_dict = batch.to_dict(flatten=True) if isinstance(batch, Observation) else dict(batch)

        # Infer batch size and device
        batch_size, device = self._infer_batch_info(batch_dict)

        # 1. Process state (support both "observation.state" and "state" keys)
        state_tensor = batch_dict.get(OBSERVATION_STATE)
        if state_tensor is None:
            state_tensor = batch_dict.get(STATE)
        if state_tensor is not None:
            state, state_mask = self._process_state(state_tensor)
            result[STATE] = state
            result[STATE_MASK] = state_mask

        # 2. Process action (for training)
        if ACTION in batch_dict and batch_dict[ACTION] is not None:
            action, action_mask = self._process_action(batch_dict[ACTION])
            result[ACTION] = action
            result[ACTION_MASK] = action_mask

        # 3. Add embodiment ID
        result[EMBODIMENT_ID] = torch.full((batch_size,), self._embodiment_id, dtype=torch.long, device=device)

        # 4. Process images with Eagle
        eagle_inputs = self._process_images(batch_dict)
        result.update(eagle_inputs)

        return result

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
        # Ensure (B, T, D) format
        if state.dim() == 2:  # noqa: PLR2004
            state = state.unsqueeze(1)  # (B, D) -> (B, 1, D)

        batch_size, _t, orig_dim = state.shape

        # Normalize using registered buffers
        if self.normalize_min_max:
            state = self._min_max_normalize(
                state,
                self.state_min[:orig_dim].to(state.device),
                self.state_max[:orig_dim].to(state.device),
            )

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
        # Ensure (B, T, D) format
        if action.dim() == 2:  # noqa: PLR2004
            # Single timestep - replicate to action_horizon
            action = action.unsqueeze(1).repeat(1, self.action_horizon, 1)
        elif action.dim() == 3:  # noqa: PLR2004
            batch_size, t, _d = action.shape
            if t < self.action_horizon:
                # Pad by repeating last action
                last = action[:, -1:, :]
                padding = last.repeat(1, self.action_horizon - t, 1)
                action = torch.cat([action, padding], dim=1)
            elif t > self.action_horizon:
                action = action[:, : self.action_horizon, :]

        batch_size, horizon, orig_dim = action.shape

        # Normalize using registered buffers
        if self.normalize_min_max:
            # Flatten for normalization, then reshape back
            flat = action.reshape(batch_size * horizon, orig_dim)
            flat = self._min_max_normalize(
                flat,
                self.action_min[:orig_dim].to(action.device),
                self.action_max[:orig_dim].to(action.device),
            )
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

    @staticmethod
    def _min_max_normalize(
        x: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ) -> torch.Tensor:
        """Apply min-max normalization to [-1, 1].

        Args:
            x: Input tensor.
            min_val: Minimum values for each dimension.
            max_val: Maximum values for each dimension.

        Returns:
            Normalized tensor in range [-1, 1].
        """
        denom = max_val - min_val
        # Avoid division by zero
        mask = denom != 0
        safe_denom = torch.where(mask, denom, torch.ones_like(denom))

        # Map to [-1, 1]: normalized = 2 * (x - min) / (max - min) - 1
        normalized = 2 * (x - min_val) / safe_denom - 1
        return torch.where(mask, normalized, torch.zeros_like(normalized))

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
        img_keys = sorted([k for k in batch if k.startswith(OBSERVATION_IMAGES_PREFIX)])
        if not img_keys:
            img_keys = sorted([k for k in batch if k.startswith(f"{IMAGES}.") and k != IMAGES])
        if not img_keys and OBSERVATION_IMAGE in batch:
            img_keys = [OBSERVATION_IMAGE]
        if not img_keys and IMAGES in batch and isinstance(batch[IMAGES], torch.Tensor):
            img_keys = [IMAGES]

        if not img_keys:
            return {}

        # Get task description
        task_value = batch.get(TASK, "Perform the task.")
        if isinstance(task_value, list):
            task_value = task_value[0] if task_value else "Perform the task."

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
        batch_text = [task_value] * batch_size
        return self.eagle_processor.batch_encode(batch_images, batch_text)


# ============================================================================ #
# Postprocessor                                                                #
# ============================================================================ #


class GrootPostprocessor(nn.Module):
    """Postprocessor for Groot policy outputs.

    Transforms GrootModel outputs back to original action space:
    1. Slices action to environment dimension
    2. Denormalizes from [-1, 1] to original range

    Stats are registered as buffers for automatic device handling.

    Args:
        env_action_dim: Original action dimension of the environment.
        normalize_min_max: Whether min-max normalization was applied.
        stats: Dataset statistics for denormalization {key: {min, max}}.

    Examples:
        >>> postprocessor = GrootPostprocessor(
        ...     env_action_dim=7,
        ...     stats=dataset.stats,
        ... )
        >>> action = postprocessor(model_output["action_pred"])
    """

    def __init__(
        self,
        *,
        env_action_dim: int = 0,
        normalize_min_max: bool = True,
        stats: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Initialize postprocessor with denormalization statistics."""
        super().__init__()

        self.env_action_dim = env_action_dim
        self.normalize_min_max = normalize_min_max

        # Register action stats as buffers
        if stats is not None and ACTION in stats:
            action_stats = stats[ACTION]
            # Use env_action_dim or a reasonable max
            dim = env_action_dim if env_action_dim > 0 else MAX_ACTION_DIM
            action_min = self._to_tensor(action_stats.get(STATS_MIN), dim)
            action_max = self._to_tensor(action_stats.get(STATS_MAX), dim)
        else:
            dim = env_action_dim if env_action_dim > 0 else MAX_ACTION_DIM
            action_min = torch.zeros(dim)
            action_max = torch.ones(dim)

        self.register_buffer("action_min", action_min)
        self.register_buffer("action_max", action_max)

    @staticmethod
    def _to_tensor(value: list[float] | torch.Tensor | None, target_dim: int) -> torch.Tensor:
        """Convert value to tensor and pad/truncate to target dimension.

        Args:
            value: Value to convert (tensor, list, or None).
            target_dim: Target dimension.

        Returns:
            Tensor of shape (target_dim,).
        """
        if value is None:
            return torch.zeros(target_dim)

        t = torch.as_tensor(value, dtype=torch.float32).flatten()
        current_dim = t.shape[0]

        if current_dim == target_dim:
            return t
        if current_dim < target_dim:
            padding = torch.zeros(target_dim - current_dim, dtype=t.dtype)
            return torch.cat([t, padding])
        return t[:target_dim]

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """Postprocess action output.

        Args:
            action: Model output (B, T, D) or (B, D).

        Returns:
            Denormalized action in original space (B, env_action_dim).
        """
        # Select last timestep if multiple
        if action.dim() == 3:  # noqa: PLR2004
            action = action[:, -1, :]  # (B, T, D) -> (B, D)

        # Slice to env dimension
        if self.env_action_dim > 0 and action.shape[-1] >= self.env_action_dim:
            action = action[..., : self.env_action_dim]

        # Denormalize
        if self.normalize_min_max:
            action = self._min_max_denormalize(action)

        return action

    def _min_max_denormalize(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from [-1, 1] to original range.

        Args:
            action: Normalized action tensor.

        Returns:
            Denormalized action.
        """
        d = action.shape[-1]

        min_val = self.action_min[:d].to(action.device)
        max_val = self.action_max[:d].to(action.device)

        denom = max_val - min_val
        mask = denom != 0
        safe_denom = torch.where(mask, denom, torch.ones_like(denom))

        # Inverse of min-max normalization: x = (normalized + 1) / 2 * denom + min
        denormalized = (action + 1.0) * 0.5 * safe_denom + min_val
        return torch.where(mask, denormalized, min_val)


# ============================================================================ #
# Factory Function                                                             #
# ============================================================================ #


def make_groot_transforms(
    *,
    max_state_dim: int = MAX_STATE_DIM,
    max_action_dim: int = MAX_ACTION_DIM,
    action_horizon: int = ACTION_HORIZON,
    embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
    env_action_dim: int = 0,
    stats: dict[str, dict[str, Any]] | None = None,
    eagle_processor_repo: str = DEFAULT_EAGLE_PROCESSOR_REPO,
) -> tuple[GrootPreprocessor, GrootPostprocessor]:
    """Create preprocessor and postprocessor for Groot policy.

    Convenience factory function that creates both transforms with consistent settings.

    Args:
        max_state_dim: Maximum state dimension.
        max_action_dim: Maximum action dimension.
        action_horizon: Number of action steps.
        embodiment_tag: Embodiment identifier.
        env_action_dim: Original environment action dimension.
        stats: Dataset statistics for normalization.
        eagle_processor_repo: HuggingFace repo for Eagle processor.

    Returns:
        Tuple of (preprocessor, postprocessor).

    Examples:
        >>> preprocessor, postprocessor = make_groot_transforms(
        ...     max_state_dim=64,
        ...     max_action_dim=32,
        ...     env_action_dim=7,
        ...     stats=dataset.stats,
        ... )
        >>> batch = preprocessor(observation)
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
    )

    postprocessor = GrootPostprocessor(
        env_action_dim=env_action_dim,
        normalize_min_max=True,
        stats=stats,
    )

    return preprocessor, postprocessor


__all__ = [
    "GrootPostprocessor",
    "GrootPreprocessor",
    "make_groot_transforms",
]
