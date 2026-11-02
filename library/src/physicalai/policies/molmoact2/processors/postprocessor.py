# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 postprocessor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from physicalai.data.observation import ACTION, Feature, FeatureType
from physicalai.policies.utils.features import get_feature_by_type

from .normalization import MolmoAct2NormalizeTransform

if TYPE_CHECKING:
    from physicalai.policies.utils import JointFrameTransform


class MolmoAct2Postprocessor(torch.nn.Module):
    """Convert normalized MolmoAct2 outputs back into action space.

    Steps:
        1. Resolve the action tensor from model outputs.
        2. Clamp normalized actions to the unit range.
        3. Denormalize actions using resolved feature statistics.
        4. Optionally map checkpoint joints back to the SO-101 frame.
    """

    def __init__(
        self,
        *,
        output_features: list[Feature],
        normalization_mode: str = "QUANTILES",
        joint_transform: JointFrameTransform | None = None,
    ) -> None:
        """Initialize MolmoAct2 postprocessor.

        Args:
            output_features: Output feature definitions.
            normalization_mode: Normalization mode for action denormalization.
            joint_transform: Optional calibration transform applied after denormalization.
        """
        super().__init__()
        action_feature = get_feature_by_type(output_features, FeatureType.ACTION)
        self.action_name = action_feature.name if action_feature else ACTION
        self._denormalizer = MolmoAct2NormalizeTransform(
            input_features=[],
            output_features=output_features,
            normalization_mode=normalization_mode,
            inverse=True,
        )
        self._joint_transform = joint_transform

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Denormalize, clamp and (optionally) map actions back to the robot frame.

        Args:
            batch: Batch containing ACTION or "actions" tensor.

        Returns:
            Batch with ACTION denormalized and clamped.

        Raises:
            ValueError: If no action tensor is present in the batch.
        """
        batch = dict(batch)
        action = batch.get(ACTION, batch.get("actions"))
        if action is None:
            msg = "MolmoAct2 postprocessor expected an action tensor in outputs."
            raise ValueError(msg)

        action = action.clamp(-1.0, 1.0)
        action = self._denormalizer.to(action.device)({self.action_name: action})[self.action_name]
        if self._joint_transform is not None:
            action = self._joint_transform.inverse(action)
        batch[ACTION] = action
        return batch
