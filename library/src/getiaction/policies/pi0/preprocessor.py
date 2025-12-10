# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor for Pi0/Pi0.5 models.

This module provides preprocessing functionality for transforming observations
and actions into the format expected by Pi0/Pi0.5 models.

Handles:
- Image resizing and normalization
- State/action normalization (quantile or z-score)
- State/action padding to max dimensions
- Language tokenization
- Output denormalization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from getiaction.data.observation import Observation

logger = logging.getLogger(__name__)


@dataclass
class NormStats:
    """Normalization statistics for a feature.

    Supports both z-score (mean/std) and quantile (q01/q99) normalization.

    Attributes:
        mean: Mean values for z-score normalization.
        std: Standard deviation for z-score normalization.
        q01: 1st percentile for quantile normalization.
        q99: 99th percentile for quantile normalization.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


@dataclass
class Pi0Preprocessor:
    """Preprocessor for Pi0/Pi0.5 model inputs.

    Transforms observations and actions into the format expected by Pi0Model:
    1. Resizes images to target resolution with padding
    2. Normalizes images to [-1, 1]
    3. Normalizes state/action using quantile or z-score normalization
    4. Pads state/action to max dimensions
    5. Tokenizes language prompts

    Args:
        max_state_dim: Maximum state dimension for padding.
        max_action_dim: Maximum action dimension for padding.
        action_horizon: Number of action steps to predict.
        image_resolution: Target image resolution (height, width).
        use_quantile_norm: Whether to use quantile normalization (Pi0.5 default).
        stats: Normalization statistics dict.
        tokenizer_name: HuggingFace tokenizer name.
        max_token_len: Maximum tokenized prompt length.

    Example:
        >>> preprocessor = Pi0Preprocessor(
        ...     max_state_dim=32,
        ...     max_action_dim=32,
        ...     stats=dataset_stats,
        ... )
        >>> batch = preprocessor(raw_batch)
    """

    max_state_dim: int = 32
    max_action_dim: int = 32
    action_horizon: int = 50
    image_resolution: tuple[int, int] = (224, 224)
    use_quantile_norm: bool = True  # Pi0.5 uses quantiles
    stats: dict[str, NormStats] | None = None
    tokenizer_name: str = "google/paligemma-3b-pt-224"
    max_token_len: int = 200

    # Internal state
    _tokenizer: Any = field(default=None, init=False, repr=False)

    @property
    def tokenizer(self) -> Any:  # noqa: ANN401
        """Lazy-load tokenizer.

        Raises:
            ImportError: If transformers library is not installed.
        """
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415

                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            except ImportError as e:
                msg = "Tokenizer requires transformers. Install with: pip install transformers"
                raise ImportError(msg) from e
        return self._tokenizer

    def __call__(self, batch: Mapping[str, Any] | Observation) -> dict[str, Any]:
        """Preprocess a batch for Pi0Model.

        Args:
            batch: Input batch, either as:
                - An Observation dataclass
                - A dict with keys like observation.state, observation.images.*, action, task

        Returns:
            Preprocessed batch with keys:
                - images: dict of processed image tensors
                - image_masks: dict of validity masks
                - state: padded and normalized state
                - tokenized_prompt: token IDs
                - tokenized_prompt_mask: token validity mask
                - actions: padded and normalized actions (if present)
        """
        # Convert Observation to dict if needed
        batch_dict = batch.to_dict(flatten=True) if hasattr(batch, "to_dict") else dict(batch)

        result: dict[str, Any] = {}

        # Process images
        images, image_masks = self._process_images(batch_dict)
        result["images"] = images
        result["image_masks"] = image_masks

        # Process state
        state = batch_dict.get("observation.state")
        if state is None:
            state = batch_dict.get("state")
        if state is not None:
            result["state"] = self._process_state(state)

        # Process language
        task = batch_dict.get("task") or batch_dict.get("prompt", "")
        tokens, masks = self._tokenize(task)
        result["tokenized_prompt"] = tokens
        result["tokenized_prompt_mask"] = masks

        # Process actions (for training)
        actions = batch_dict.get("action")
        if actions is not None:
            result["actions"] = self._process_actions(actions)

        return result

    def _process_images(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Process images: resize, normalize, and create masks.

        Args:
            batch: Input batch dict.

        Returns:
            Tuple of (images dict, image_masks dict).
        """
        images = {}
        image_masks = {}

        # Find image keys
        image_keys = [k for k in batch if "image" in k.lower() and "mask" not in k.lower()]

        for key in image_keys:
            img = batch[key]
            if img is None:
                continue

            # Convert to tensor if needed
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            # Ensure float and correct range
            if img.dtype == torch.uint8:
                img = img.float() / 255.0

            # Ensure (B, C, H, W) format (3D means single image: C, H, W)
            single_image_ndim = 3
            if img.ndim == single_image_ndim:
                img = img.unsqueeze(0)

            # Resize with padding
            target_h, target_w = self.image_resolution
            img = self._resize_with_pad(img, target_h, target_w)

            # Normalize to [-1, 1]
            img = img * 2.0 - 1.0

            # Extract clean name for output
            clean_name = key.replace("observation.images.", "").replace("observation.", "")
            images[clean_name] = img
            image_masks[clean_name] = torch.ones(img.shape[0], dtype=torch.bool, device=img.device)

        return images, image_masks

    @staticmethod
    def _resize_with_pad(
        images: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Resize images with padding to preserve aspect ratio.

        Args:
            images: Image tensor (batch, channels, h, w).
            height: Target height.
            width: Target width.

        Returns:
            Resized images.
        """
        import torch.nn.functional as F  # noqa: N812, PLC0415

        _, _, h, w = images.shape

        # Calculate scale to fit within target
        scale = min(height / h, width / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize
        resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # Pad to target size
        pad_h = height - new_h
        pad_w = width - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

    def _process_state(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Process state: normalize and pad.

        Args:
            state: State tensor (B, D) or (B, T, D).

        Returns:
            Processed state tensor.
        """
        # Convert to tensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        # Ensure float
        state = state.float()

        # Get original dimension
        orig_dim = state.shape[-1]

        # Normalize
        if self.stats is not None and "state" in self.stats:
            state = self._normalize(state, self.stats["state"])

        # Pad to max_state_dim
        batched_state_ndim = 2
        if orig_dim < self.max_state_dim:
            pad_size = self.max_state_dim - orig_dim
            if state.ndim == batched_state_ndim:
                padding = torch.zeros(state.shape[0], pad_size, dtype=state.dtype, device=state.device)
            else:
                padding = torch.zeros(*state.shape[:-1], pad_size, dtype=state.dtype, device=state.device)
            state = torch.cat([state, padding], dim=-1)
        elif orig_dim > self.max_state_dim:
            state = state[..., : self.max_state_dim]

        return state

    def _process_actions(self, actions: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Process actions: normalize and pad.

        Args:
            actions: Action tensor (B, T, D) or (B, D).

        Returns:
            Processed action tensor.
        """
        # Convert to tensor
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        # Ensure float
        actions = actions.float()

        # Ensure (B, T, D) format (2D means single timestep: B, D)
        single_timestep_ndim = 2
        if actions.ndim == single_timestep_ndim:
            actions = actions.unsqueeze(1)

        orig_dim = actions.shape[-1]

        # Normalize
        if self.stats is not None and "actions" in self.stats:
            actions = self._normalize(actions, self.stats["actions"])

        # Pad action dimension
        if orig_dim < self.max_action_dim:
            pad_size = self.max_action_dim - orig_dim
            padding = torch.zeros(*actions.shape[:-1], pad_size, dtype=actions.dtype, device=actions.device)
            actions = torch.cat([actions, padding], dim=-1)
        elif orig_dim > self.max_action_dim:
            actions = actions[..., : self.max_action_dim]

        # Pad/truncate time dimension
        if actions.shape[1] < self.action_horizon:
            pad_size = self.action_horizon - actions.shape[1]
            padding = torch.zeros(
                actions.shape[0],
                pad_size,
                actions.shape[2],
                dtype=actions.dtype,
                device=actions.device,
            )
            actions = torch.cat([actions, padding], dim=1)
        elif actions.shape[1] > self.action_horizon:
            actions = actions[:, : self.action_horizon]

        return actions

    def _normalize(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        """Normalize tensor using stats.

        Args:
            x: Input tensor.
            stats: Normalization statistics.

        Returns:
            Normalized tensor.
        """
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            # Quantile normalization to [-1, 1]
            q01 = torch.tensor(stats.q01, dtype=x.dtype, device=x.device)
            q99 = torch.tensor(stats.q99, dtype=x.dtype, device=x.device)
            # Truncate to actual dimension
            dim = x.shape[-1]
            q01 = q01[..., :dim]
            q99 = q99[..., :dim]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        if stats.mean is not None and stats.std is not None:
            # Z-score normalization
            mean = torch.tensor(stats.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(stats.std, dtype=x.dtype, device=x.device)
            dim = x.shape[-1]
            mean = mean[..., :dim]
            std = std[..., :dim]
            return (x - mean) / (std + 1e-6)
        return x

    def _tokenize(self, text: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text prompts.

        Args:
            text: Text string or list of strings.

        Returns:
            Tuple of (token_ids, attention_mask).
        """
        # Handle single string
        if isinstance(text, str):
            text = [text]

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt",
        )

        return encoded["input_ids"], encoded["attention_mask"].bool()


@dataclass
class Pi0Postprocessor:
    """Postprocessor for Pi0/Pi0.5 model outputs.

    Transforms model outputs back to the original action space:
    1. Truncates to actual action dimension
    2. Denormalizes using dataset statistics

    Args:
        action_dim: Actual action dimension (before padding).
        max_action_dim: Padded action dimension.
        use_quantile_norm: Whether quantile normalization was used.
        stats: Normalization statistics dict.
    """

    action_dim: int
    max_action_dim: int = 32
    use_quantile_norm: bool = True
    stats: dict[str, NormStats] | None = None

    def __call__(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Postprocess model outputs.

        Args:
            outputs: Model output dict with "actions" key.

        Returns:
            Postprocessed outputs with denormalized actions.
        """
        result = dict(outputs)

        if "actions" in result:
            actions = result["actions"]

            # Truncate to actual dimension
            actions = actions[..., : self.action_dim]

            # Denormalize
            if self.stats is not None and "actions" in self.stats:
                actions = self._denormalize(actions, self.stats["actions"])

            result["actions"] = actions

        return result

    def _denormalize(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        """Denormalize tensor using stats.

        Args:
            x: Normalized tensor.
            stats: Normalization statistics.

        Returns:
            Denormalized tensor.
        """
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = torch.tensor(stats.q01, dtype=x.dtype, device=x.device)
            q99 = torch.tensor(stats.q99, dtype=x.dtype, device=x.device)
            dim = x.shape[-1]
            q01 = q01[..., :dim]
            q99 = q99[..., :dim]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        if stats.mean is not None and stats.std is not None:
            mean = torch.tensor(stats.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(stats.std, dtype=x.dtype, device=x.device)
            dim = x.shape[-1]
            mean = mean[..., :dim]
            std = std[..., :dim]
            return x * (std + 1e-6) + mean
        return x


def make_pi0_preprocessors(
    max_state_dim: int = 32,
    max_action_dim: int = 32,
    action_horizon: int = 50,
    env_action_dim: int | None = None,
    stats: dict[str, dict[str, list[float]]] | None = None,
    *,
    use_quantile_norm: bool = True,
    image_resolution: tuple[int, int] = (224, 224),
    tokenizer_name: str = "google/paligemma-3b-pt-224",
    max_token_len: int = 200,
) -> tuple[Pi0Preprocessor, Pi0Postprocessor]:
    """Create preprocessor and postprocessor pair.

    Args:
        max_state_dim: Maximum state dimension.
        max_action_dim: Maximum action dimension.
        action_horizon: Number of action steps.
        env_action_dim: Actual environment action dimension.
        stats: Dataset statistics as nested dicts.
        use_quantile_norm: Whether to use quantile normalization.
        image_resolution: Target image resolution.
        tokenizer_name: HuggingFace tokenizer name.
        max_token_len: Maximum token length.

    Returns:
        Tuple of (preprocessor, postprocessor).
    """
    # Convert stats format if needed
    norm_stats: dict[str, NormStats] | None = None
    if stats is not None:
        norm_stats = {}
        for key, stat_dict in stats.items():
            norm_stats[key] = NormStats(
                mean=np.array(stat_dict.get("mean")) if "mean" in stat_dict else None,
                std=np.array(stat_dict.get("std")) if "std" in stat_dict else None,
                q01=np.array(stat_dict.get("q01")) if "q01" in stat_dict else None,
                q99=np.array(stat_dict.get("q99")) if "q99" in stat_dict else None,
            )

    preprocessor = Pi0Preprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        action_horizon=action_horizon,
        image_resolution=image_resolution,
        use_quantile_norm=use_quantile_norm,
        stats=norm_stats,
        tokenizer_name=tokenizer_name,
        max_token_len=max_token_len,
    )

    postprocessor = Pi0Postprocessor(
        action_dim=env_action_dim or max_action_dim,
        max_action_dim=max_action_dim,
        use_quantile_norm=use_quantile_norm,
        stats=norm_stats,
    )

    return preprocessor, postprocessor
