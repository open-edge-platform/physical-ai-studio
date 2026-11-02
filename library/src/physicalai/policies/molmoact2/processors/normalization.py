# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared MolmoAct2 normalization transforms."""

from __future__ import annotations

import torch

from physicalai.data.observation import Feature, FeatureType
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType


class MolmoAct2NormalizeTransform(FeatureNormalizeTransform):
    """Normalize or denormalize MolmoAct2 state and action features.

    Steps:
        1. Resolve state and action feature metadata.
        2. Select configured or identity normalization per feature type.
        3. Validate unnormalized pass-through dimensions on forward input.
        4. Move statistics to the batch device and apply the base transform.
    """

    def __init__(
        self,
        *,
        input_features: list[Feature],
        output_features: list[Feature],
        normalization_mode: str = "QUANTILES",
        inverse: bool = False,
    ) -> None:
        """Initialize normalization from resolved feature metadata.

        Args:
            input_features: Observation feature definitions.
            output_features: Action feature definitions.
            normalization_mode: Statistical normalization mode for state and action.
            inverse: Whether to apply inverse normalization.
        """
        features = {feature.name: feature for feature in input_features + output_features if feature.name is not None}
        mode = NormalizationType(normalization_mode)
        state_feature = next(
            (feature for feature in input_features if feature.ftype == FeatureType.STATE),
            None,
        )
        action_feature = next(
            (feature for feature in output_features if feature.ftype == FeatureType.ACTION),
            None,
        )
        norm_map = {
            FeatureType.STATE: (
                mode
                if state_feature is not None and state_feature.normalization_data is not None
                else NormalizationType.IDENTITY
            ),
            FeatureType.ACTION: (
                mode
                if action_feature is not None and action_feature.normalization_data is not None
                else NormalizationType.IDENTITY
            ),
            FeatureType.VISUAL: NormalizationType.IDENTITY,
        }
        super().__init__(features, norm_map, inverse=inverse)
        self._inverse = inverse

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply normalization and validate forward pass-through dimensions.

        Returns:
            The batch with configured features transformed.
        """
        if not self._inverse:
            self._validate_passthrough_values(batch)

        device = next((value.device for value in batch.values() if torch.is_tensor(value)), None)
        if device is not None:
            self.to(device)
        return super().forward(batch)

    def _validate_passthrough_values(self, batch: dict[str, torch.Tensor]) -> None:
        """Require dimensions excluded from normalization to already use unit range.

        Raises:
            ValueError: If a pass-through value is outside [-1, 1].
        """
        for feature_name, feature in self._features.items():
            normalization_data = feature.normalization_data
            mask = normalization_data.mask if normalization_data is not None else None
            if mask is None or all(mask):
                continue

            for batch_key, value in batch.items():
                if batch_key != feature_name and not batch_key.endswith(f".{feature_name}"):
                    continue
                if not torch.is_tensor(value):
                    continue
                tensor_mask = torch.tensor(mask, device=value.device, dtype=torch.bool)
                if tensor_mask.ndim != 1 or value.shape[-1] != tensor_mask.shape[0]:
                    continue
                passthrough_values = value[..., ~tensor_mask]
                if ((passthrough_values < -1.0) | (passthrough_values > 1.0)).any():
                    msg = (
                        f"MolmoAct2 {batch_key} pass-through values are not under [-1, 1]. "
                        "Please set normalize_gripper=True."
                    )
                    raise ValueError(msg)
