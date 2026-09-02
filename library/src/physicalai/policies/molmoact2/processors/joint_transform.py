# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO-100/101 joint frame transform for MolmoAct2.

The released MolmoAct2-SO100_101 checkpoint was trained with the pre-#777
LeRobot joint calibration. Newer LeRobot data uses a different convention, so
joint observations/actions must be mapped into the checkpoint convention on the
way in and back to the robot convention on the way out.

- Robot -> checkpoint:  ``x_ckpt = sign * x_robot + offset``
- Checkpoint -> robot:  ``x_robot = sign * (x_ckpt - offset)``

``sign`` is +/-1 (so ``1 / sign == sign``). The transform touches only the
leading joint dimensions; any trailing dimensions pass through unchanged. For
SO-101 the defaults flip ``shoulder_lift`` and shift ``shoulder_lift`` /
``elbow_flex`` by 90 degrees, matching the LeRobot backward-compatibility guide.
"""

from __future__ import annotations

from itertools import starmap
from typing import cast

import torch

from physicalai.data import NormalizationParameters

SO101_JOINT_SIGNS = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
SO101_JOINT_OFFSETS = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)


class JointFrameTransform:
    """Map SO-101 joint values between robot and checkpoint calibration frames.

    Steps:
        1. Select the leading SO-101 joint dimensions.
        2. Move fixed signs and offsets to the input tensor device and dtype.
        3. Apply the forward or inverse affine calibration.
        4. Preserve any trailing non-joint dimensions unchanged.
    """

    def __init__(self) -> None:
        """Store the fixed SO-101 joint signs and offsets."""
        self.num_joints = len(SO101_JOINT_SIGNS)
        self._signs = torch.tensor(SO101_JOINT_SIGNS, dtype=torch.float32)
        self._offsets = torch.tensor(SO101_JOINT_OFFSETS, dtype=torch.float32)

    def to_checkpoint(self, values: torch.Tensor) -> torch.Tensor:
        """Map robot-frame joints to the checkpoint frame.

        Returns:
            ``values`` with the leading joint dims mapped ``sign * x + offset``.
        """
        return self._apply(values, inverse=False)

    def to_robot(self, values: torch.Tensor) -> torch.Tensor:
        """Map checkpoint-frame joints back to the robot frame.

        Returns:
            ``values`` with the leading joint dims mapped ``sign * (x - offset)``.
        """
        return self._apply(values, inverse=True)

    def normalization_to_checkpoint(
        self,
        normalization: NormalizationParameters,
        dimension: int,
    ) -> NormalizationParameters:
        """Map robot-frame normalization metadata to the checkpoint frame.

        Returns:
            New normalization metadata aligned with ``to_checkpoint`` values.

        Raises:
            ValueError: If a statistic does not match the feature dimension.
        """
        if normalization.mask is not None and len(normalization.mask) != dimension:
            msg = f"Normalization mask length {len(normalization.mask)} does not match feature dimension {dimension}."
            raise ValueError(msg)
        mean = self._transform_stat(normalization.mean, dimension, include_offset=True)
        std = self._transform_stat(normalization.std, dimension, absolute_scale=True)
        minimum, maximum = self._transform_bounds(normalization.min, normalization.max, dimension)
        q01, q99 = self._transform_bounds(normalization.q01, normalization.q99, dimension)
        return NormalizationParameters(
            mean=mean,
            std=std,
            min=minimum,
            max=maximum,
            q01=q01,
            q99=q99,
            mask=None if normalization.mask is None else list(normalization.mask),
        )

    def _transform_bounds(
        self,
        lower: list[float] | list[list[float]] | list[list[list[float]]] | float | None,
        upper: list[float] | list[list[float]] | list[list[list[float]]] | float | None,
        dimension: int,
    ) -> tuple[list[float] | None, list[float] | None]:
        if lower is None or upper is None:
            return (
                self._transform_stat(lower, dimension, include_offset=True),
                self._transform_stat(upper, dimension, include_offset=True),
            )
        transformed_lower = self._transform_stat(lower, dimension, include_offset=True)
        transformed_upper = self._transform_stat(upper, dimension, include_offset=True)
        if transformed_lower is None or transformed_upper is None:
            msg = "Transformed bounds resulted in None values."
            raise ValueError(msg)
        return (
            list(starmap(min, zip(transformed_lower, transformed_upper, strict=True))),
            list(starmap(max, zip(transformed_lower, transformed_upper, strict=True))),
        )

    def _transform_stat(
        self,
        statistic: list[float] | list[list[float]] | list[list[list[float]]] | float | None,
        dimension: int,
        *,
        include_offset: bool = False,
        absolute_scale: bool = False,
    ) -> list[float] | None:
        if statistic is None:
            return None
        if isinstance(statistic, int | float):
            values = [float(statistic)] * dimension
        else:
            if any(isinstance(value, list) for value in statistic):
                msg = "Joint normalization statistics must be scalar or one-dimensional."
                raise ValueError(msg)
            values = list(cast("list[float]", statistic))
        if len(values) != dimension:
            msg = f"Normalization statistic length {len(values)} does not match feature dimension {dimension}."
            raise ValueError(msg)

        count = min(self.num_joints, dimension)
        output = list(values)
        for index in range(count):
            sign = float(self._signs[index])
            scale = abs(sign) if absolute_scale else sign
            output[index] = scale * values[index] + (float(self._offsets[index]) if include_offset else 0.0)
        return output

    def _apply(self, values: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        """Apply the (inverse) affine joint transform to the leading joint dims.

        Returns:
            A new tensor with the leading joint dimensions transformed.
        """
        num_joints = min(self.num_joints, values.shape[-1])
        signs = self._signs[:num_joints].to(device=values.device, dtype=values.dtype)
        offsets = self._offsets[:num_joints].to(device=values.device, dtype=values.dtype)

        out = values.clone()
        joints = values[..., :num_joints]
        out[..., :num_joints] = signs * (joints - offsets) if inverse else signs * joints + offsets
        return out
