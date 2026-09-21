# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint calibration frame transforms.

Example:
    >>> import torch
    >>> from physicalai.policies.utils import JointFrameTransform
    >>> transform = JointFrameTransform(signs=[1.0, -1.0], offsets=[10.0, 20.0])
    >>> source_joints = torch.tensor([[2.0, 3.0, 4.0]])
    >>> transformed_joints = transform.forward(source_joints)
    >>> transformed_joints
    tensor([[12., 17.,  4.]])
    >>> transform.inverse(transformed_joints)
    tensor([[2., 3., 4.]])
"""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, cast

import torch

from physicalai.data import NormalizationParameters, NormalizationValue

if TYPE_CHECKING:
    from collections.abc import Sequence


class JointFrameTransform:
    """Apply an invertible affine transform to leading joint values."""

    def __init__(self, *, signs: Sequence[float], offsets: Sequence[float]) -> None:
        """Store the joint signs and offsets.

        Raises:
            ValueError: If signs and offsets differ in length or a sign is not +/-1.
        """
        if len(signs) != len(offsets):
            msg = f"signs ({len(signs)}) and offsets ({len(offsets)}) must match."
            raise ValueError(msg)
        if any(sign not in {-1.0, 1.0} for sign in signs):
            msg = "Joint frame transform signs must be either -1 or 1."
            raise ValueError(msg)
        self.num_joints = len(signs)
        self._signs = torch.tensor(signs, dtype=torch.float32)
        self._offsets = torch.tensor(offsets, dtype=torch.float32)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply ``sign * value + offset`` to leading joint values.

        Returns:
            A transformed copy of ``values``.
        """
        return self._apply(values, inverse=False)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        """Apply ``sign * (value - offset)`` to leading joint values.

        Returns:
            An inverse-transformed copy of ``values``.
        """
        return self._apply(values, inverse=True)

    def forward_normalization(
        self,
        normalization: NormalizationParameters,
        dimension: int,
    ) -> NormalizationParameters:
        """Apply the forward affine transform to normalization metadata.

        Returns:
            New normalization metadata aligned with ``forward`` values.
        """
        self._validate_mask(normalization, dimension)
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

    def forward_normalization_from_scaled_input(
        self,
        normalization: NormalizationParameters,
        dimension: int,
        *,
        scales: Sequence[float],
    ) -> NormalizationParameters:
        """Align normalization metadata with an input that uses different scales.

        ``scales`` applies to leading configured dimensions. Remaining dimensions
        pass through unchanged.

        Returns:
            New normalization metadata aligned with ``forward`` applied to scaled inputs.

        Raises:
            ValueError: If a statistic or mask has the wrong dimension or a scale is zero.
        """
        self._validate_mask(normalization, dimension)
        active_scales = scales[: min(len(scales), self.num_joints, dimension)]
        if any(scale == 0 for scale in active_scales):
            msg = "Joint input scales must be non-zero."
            raise ValueError(msg)
        mean = self._scaled_stat(normalization.mean, dimension, active_scales, include_offset=True)
        std = self._scaled_stat(normalization.std, dimension, active_scales, absolute_scale=True)
        minimum, maximum = self._scaled_bounds(normalization.min, normalization.max, dimension, active_scales)
        q01, q99 = self._scaled_bounds(normalization.q01, normalization.q99, dimension, active_scales)
        return NormalizationParameters(
            mean=mean,
            std=std,
            min=minimum,
            max=maximum,
            q01=q01,
            q99=q99,
            mask=None if normalization.mask is None else list(normalization.mask),
        )

    @staticmethod
    def _validate_mask(normalization: NormalizationParameters, dimension: int) -> None:
        if normalization.mask is not None and len(normalization.mask) != dimension:
            msg = f"Normalization mask length {len(normalization.mask)} does not match feature dimension {dimension}."
            raise ValueError(msg)

    def _scaled_bounds(
        self,
        lower: NormalizationValue,
        upper: NormalizationValue,
        dimension: int,
        scales: Sequence[float],
    ) -> tuple[list[float] | None, list[float] | None]:
        transformed_lower = self._scaled_stat(lower, dimension, scales, include_offset=True)
        transformed_upper = self._scaled_stat(upper, dimension, scales, include_offset=True)
        if transformed_lower is None or transformed_upper is None:
            return transformed_lower, transformed_upper
        return (
            list(starmap(min, zip(transformed_lower, transformed_upper, strict=True))),
            list(starmap(max, zip(transformed_lower, transformed_upper, strict=True))),
        )

    def _scaled_stat(
        self,
        statistic: NormalizationValue,
        dimension: int,
        scales: Sequence[float],
        *,
        include_offset: bool = False,
        absolute_scale: bool = False,
    ) -> list[float] | None:
        values = self._stat_values(statistic, dimension)
        if values is None:
            return None

        output = list(values)
        for index, input_scale in enumerate(scales[:dimension]):
            if absolute_scale:
                output[index] = values[index] / abs(input_scale)
            elif include_offset:
                offset = float(self._offsets[index])
                output[index] = offset + (values[index] - offset) / input_scale
            else:
                output[index] = values[index] / input_scale
        return output

    def _transform_bounds(
        self,
        lower: NormalizationValue,
        upper: NormalizationValue,
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
        statistic: NormalizationValue,
        dimension: int,
        *,
        include_offset: bool = False,
        absolute_scale: bool = False,
    ) -> list[float] | None:
        values = self._stat_values(statistic, dimension)
        if values is None:
            return None

        count = min(self.num_joints, dimension)
        output = list(values)
        for index in range(count):
            sign = float(self._signs[index])
            scale = abs(sign) if absolute_scale else sign
            output[index] = scale * values[index] + (float(self._offsets[index]) if include_offset else 0.0)
        return output

    @staticmethod
    def _stat_values(statistic: NormalizationValue, dimension: int) -> list[float] | None:
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
        return values

    def _apply(self, values: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        num_joints = min(self.num_joints, values.shape[-1])
        signs = self._signs[:num_joints].to(device=values.device, dtype=values.dtype)
        offsets = self._offsets[:num_joints].to(device=values.device, dtype=values.dtype)

        output = values.clone()
        joints = values[..., :num_joints]
        output[..., :num_joints] = signs * (joints - offsets) if inverse else signs * joints + offsets
        return output
