# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Statistics-based normalizer with optional multi-embodiment support.

This module provides the `Normalizer` class for normalizing and denormalizing
observation and action data using dataset statistics. The normalizer is designed
as an `nn.Module` so that normalization statistics are automatically included in
the model's state_dict and saved/loaded with checkpoints.

For single-embodiment (default):
    normalizer = Normalizer(features=dataset.observation_features)

For multi-embodiment:
    normalizer = Normalizer(
        features={"franka": franka_features, "ur5": ur5_features},
        max_state_dim=64,
        max_action_dim=32,
    )
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from getiaction.data import Feature


class NormalizationMode(StrEnum):
    """Normalization mode for the normalizer."""

    MEAN_STD = "mean_std"
    """Normalize using mean and standard deviation: (x - mean) / std"""

    MIN_MAX = "min_max"
    """Normalize to [-1, 1] using min and max: 2 * (x - min) / (max - min) - 1"""

    IDENTITY = "identity"
    """No normalization applied."""


class Normalizer(nn.Module):
    """Statistics-based normalizer with optional multi-embodiment support.

    The normalizer stores normalization statistics (mean, std, min, max) as registered
    buffers, ensuring they are:
    - Automatically moved to the correct device with the model
    - Saved and loaded with the model's state_dict
    - Exported correctly with ONNX

    For single-embodiment usage (default), simply pass features from the dataset.
    For multi-embodiment usage, pass a dict of features keyed by embodiment ID.

    The normalization formulas are compatible with LeRobot's implementation:
    - MEAN_STD: `(x - mean) / (std + eps)` for normalize, `x * std + mean` for denormalize
    - MIN_MAX: `2 * (x - min) / (max - min + eps) - 1` for normalize to [-1, 1]

    Args:
        features: Dataset features containing normalization statistics.
            - Single-embodiment: `{feature_name: Feature}`
            - Multi-embodiment: `{embodiment_id: {feature_name: Feature}}`
        norm_mode: Normalization mode to use. Defaults to MEAN_STD.
        max_state_dim: Maximum state dimension for padding (multi-embodiment only).
            If None, no padding is applied.
        max_action_dim: Maximum action dimension for padding (multi-embodiment only).
            If None, no padding is applied.
        eps: Small epsilon value for numerical stability (division safety).
            Defaults to 1e-6 for compatibility with LeRobot.

    Examples:
        Single-embodiment (simple, default case):

            normalizer = Normalizer(features=dataset.observation_features)
            normalized = normalizer.normalize(batch)
            actions = normalizer.denormalize({"action": pred_actions})

        Multi-embodiment (explicit opt-in):

            normalizer = Normalizer(
                features={
                    "franka": franka_dataset.observation_features,
                    "ur5": ur5_datasets.observation_features,
                },
                max_state_dim=64,
                max_action_dim=32,
            )
            normalized, masks = normalizer.normalize(batch, embodiment_id="franka")
            actions = normalizer.denormalize(
                {"action": pred_actions}, embodiment_id="franka", masks=masks
            )
    """

    def __init__(
        self,
        features: dict[str, Feature] | dict[str, dict[str, Feature]],
        norm_mode: NormalizationMode = NormalizationMode.MEAN_STD,
        max_state_dim: int | None = None,
        max_action_dim: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the Normalizer.

        Args:
            features: Dataset features containing normalization statistics.
                - Single-embodiment: `{feature_name: Feature}`
                - Multi-embodiment: `{embodiment_id: {feature_name: Feature}}`
            norm_mode: Normalization mode to use. Defaults to MEAN_STD.
            max_state_dim: Maximum state dimension for padding (multi-embodiment only).
                If None, no padding is applied.
            max_action_dim: Maximum action dimension for padding (multi-embodiment only).
                If None, no padding is applied.
            eps: Small epsilon value for numerical stability (division safety).
                Defaults to 1e-6 for compatibility with LeRobot.
        """
        super().__init__()
        self.norm_mode = norm_mode
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.eps = eps

        # Detect mode: if values are Features, single-embodiment; if dicts, multi-embodiment
        self._multi_embodiment = self._is_multi_embodiment(features)

        # Store feature names for each embodiment (needed for mask creation)
        self._feature_names: dict[str, list[str]] = {}

        if self._multi_embodiment:
            # Multi-embodiment: features = {embodiment_id: {feature_name: Feature}}
            self._embodiments = list(features.keys())
            for emb_id in self._embodiments:
                emb_features: dict[str, Feature] = features[emb_id]  # type: ignore[assignment]
                self._feature_names[emb_id] = list(emb_features.keys())
                self._register_embodiment_stats(emb_id, emb_features)
        else:
            # Single-embodiment: features = {feature_name: Feature}
            features_single: dict[str, Feature] = features  # type: ignore[assignment]
            self._embodiments = ["default"]
            self._feature_names["default"] = list(features_single.keys())
            self._register_embodiment_stats("default", features_single)

    @staticmethod
    def _is_multi_embodiment(features: dict[str, Any]) -> bool:
        """Check if features dict contains embodiments or direct features.

        Args:
            features: The features dictionary to check.

        Returns:
            True if multi-embodiment (values are dicts), False if single-embodiment.
        """
        if not features:
            return False
        first_value = next(iter(features.values()))
        # If the first value is a dict, we're in multi-embodiment mode
        # Feature objects are not dicts
        return isinstance(first_value, dict)

    def _register_embodiment_stats(
        self,
        emb_id: str,
        features: dict[str, Feature],
    ) -> None:
        """Register normalization stats as buffers for one embodiment.

        Args:
            emb_id: Embodiment identifier (or "default" for single-embodiment).
            features: Features dictionary with normalization data.
        """
        for name, feature in features.items():
            if feature.normalization_data is not None:
                # Create safe buffer name (replace dots with underscores)
                safe_name = f"{emb_id}_{name}".replace(".", "_")

                norm_data = feature.normalization_data

                # Register mean
                if norm_data.mean is not None:
                    mean = self._to_tensor(norm_data.mean)
                    self.register_buffer(f"{safe_name}_mean", mean)

                # Register std (without clamping - eps is added at runtime)
                if norm_data.std is not None:
                    std = self._to_tensor(norm_data.std)
                    self.register_buffer(f"{safe_name}_std", std)

                # Register min
                if norm_data.min is not None:
                    min_val = self._to_tensor(norm_data.min)
                    self.register_buffer(f"{safe_name}_min", min_val)

                # Register max
                if norm_data.max is not None:
                    max_val = self._to_tensor(norm_data.max)
                    self.register_buffer(f"{safe_name}_max", max_val)

    @staticmethod
    def _to_tensor(value: torch.Tensor | np.ndarray | float | list) -> torch.Tensor:
        """Convert value to tensor if needed.

        Args:
            value: Value to convert (tensor, numpy array, float, or list).

        Returns:
            Tensor representation of the value.
        """
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy()).float()
        return torch.tensor(value, dtype=torch.float32)

    def normalize(
        self,
        batch: dict[str, torch.Tensor],
        embodiment_id: str | None = None,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Normalize batch using stored statistics.

        Args:
            batch: Input tensors to normalize. Keys should match feature names.
            embodiment_id: Required for multi-embodiment, ignored for single.

        Returns:
            Single-embodiment: Normalized batch dict.
            Multi-embodiment: Tuple of (normalized + padded batch, masks).
        """
        emb_id = embodiment_id or "default"
        normalized = self._apply_norm(batch, emb_id, inverse=False)

        if self._multi_embodiment and self.max_state_dim is not None:
            padded, masks = self._pad_to_max(normalized)
            return padded, masks

        return normalized

    def denormalize(
        self,
        batch: dict[str, torch.Tensor],
        embodiment_id: str | None = None,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Denormalize batch using stored statistics.

        Args:
            batch: Normalized tensors to denormalize.
            embodiment_id: Required for multi-embodiment, ignored for single.
            masks: Required for multi-embodiment to unpad.

        Returns:
            Denormalized batch dict in original dimensions.
        """
        emb_id = embodiment_id or "default"

        if self._multi_embodiment and masks is not None:
            batch = self._unpad(batch, masks)

        return self._apply_norm(batch, emb_id, inverse=True)

    def _apply_norm(
        self,
        batch: dict[str, torch.Tensor],
        emb_id: str,
        *,
        inverse: bool,
    ) -> dict[str, torch.Tensor]:
        """Apply normalization or denormalization.

        Args:
            batch: Input tensors.
            emb_id: Embodiment identifier.
            inverse: If True, denormalize; if False, normalize.

        Returns:
            Transformed batch.
        """
        result = {}
        for key, tensor in batch.items():
            safe_name = f"{emb_id}_{key}".replace(".", "_")

            # Check if we have stats for this key
            if hasattr(self, f"{safe_name}_mean"):
                mean = getattr(self, f"{safe_name}_mean")
                std = getattr(self, f"{safe_name}_std")

                # Ensure stats are on same device as tensor
                mean = mean.to(tensor.device)
                std = std.to(tensor.device)

                if self.norm_mode == NormalizationMode.MEAN_STD:
                    # LeRobot-compatible: add eps to denominator at runtime
                    denom = std + self.eps
                    if inverse:
                        result[key] = tensor * std + mean
                    else:
                        result[key] = (tensor - mean) / denom

                elif self.norm_mode == NormalizationMode.MIN_MAX:
                    min_val = getattr(self, f"{safe_name}_min").to(tensor.device)
                    max_val = getattr(self, f"{safe_name}_max").to(tensor.device)
                    denom = max_val - min_val
                    # LeRobot-compatible: only substitute eps when denom is exactly zero
                    denom = torch.where(
                        denom == 0,
                        torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype),
                        denom,
                    )

                    if inverse:
                        # From [-1, 1] to original range
                        result[key] = (tensor + 1) / 2 * denom + min_val
                    else:
                        # From original range to [-1, 1]
                        result[key] = 2 * (tensor - min_val) / denom - 1

                else:  # IDENTITY
                    result[key] = tensor
            else:
                # No stats for this key, pass through unchanged
                result[key] = tensor

        return result

    def _pad_to_max(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Pad tensors to max dimensions and create masks.

        Args:
            batch: Input tensors to pad.

        Returns:
            Tuple of (padded batch, masks dict).
        """
        padded = {}
        masks = {}

        for key, tensor in batch.items():
            # Determine target dimension based on key type
            if "state" in key and self.max_state_dim is not None:
                target_dim = self.max_state_dim
            elif "action" in key and self.max_action_dim is not None:
                target_dim = self.max_action_dim
            else:
                # No padding needed for this key
                padded[key] = tensor
                continue

            current_dim = tensor.shape[-1]
            if current_dim < target_dim:
                # Create padding
                pad_size = target_dim - current_dim
                pad_shape = list(tensor.shape)
                pad_shape[-1] = pad_size
                padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
                padded[key] = torch.cat([tensor, padding], dim=-1)

                # Create mask (1 for valid, 0 for padding)
                mask = torch.zeros(target_dim, dtype=torch.bool, device=tensor.device)
                mask[:current_dim] = True
                masks[key] = mask
            else:
                padded[key] = tensor
                masks[key] = torch.ones(target_dim, dtype=torch.bool, device=tensor.device)

        return padded, masks

    @staticmethod
    def _unpad(
        batch: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Remove padding using masks.

        Args:
            batch: Padded tensors.
            masks: Boolean masks indicating valid dimensions.

        Returns:
            Unpadded batch.
        """
        result = {}
        for key, tensor in batch.items():
            if key in masks:
                mask = masks[key]
                # Get original dimension from mask
                original_dim = mask.sum().item()
                result[key] = tensor[..., :original_dim]
            else:
                result[key] = tensor

        return result

    @property
    def embodiments(self) -> list[str]:
        """List of embodiment IDs (or ['default'] for single-embodiment)."""
        return self._embodiments

    @property
    def is_multi_embodiment(self) -> bool:
        """Whether this normalizer is in multi-embodiment mode."""
        return self._multi_embodiment

    def get_feature_names(self, embodiment_id: str | None = None) -> list[str]:
        """Get feature names for an embodiment.

        Args:
            embodiment_id: Embodiment ID (or None for single-embodiment).

        Returns:
            List of feature names.
        """
        emb_id = embodiment_id or "default"
        return self._feature_names.get(emb_id, [])

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        embodiment_id: str | None = None,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass (alias for normalize).

        This allows using the normalizer in nn.Sequential or similar constructs.

        Args:
            batch: Input tensors to normalize.
            embodiment_id: Required for multi-embodiment, ignored for single.

        Returns:
            Same as normalize().
        """
        return self.normalize(batch, embodiment_id)


__all__ = ["NormalizationMode", "Normalizer"]
