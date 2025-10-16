# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utils for dataset features normalization."""

from enum import StrEnum
from numbers import Integral
from typing import cast

import numpy as np
import torch
from torch import nn

from getiaction.data import Feature, FeatureType, NormalizationParameters


class NormalizationType(StrEnum):
    """Enum for feature normalization methods."""

    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"


class FeatureNormalizeTransform(nn.Module):
    """A PyTorch module for normalizing features.

    This transform applies different normalization strategies to features
    in observation batch dictionary. It can operate in both forward (normalize) and inverse (denormalize) modes.
    Module stores normalization statistics as buffers, which are not updated during training.

    Attributes:
        _features: Dictionary of features to normalize.
        _norm_map: Mapping of feature types to normalization methods.
        _inverse: Whether to apply inverse transformation.
        buffer_*: Automatically created buffers containing normalization statistics (mean, std, min, max)
            for each feature, with dots in feature names replaced by underscores.

    Note:
        - For MEAN_STD normalization: normalizes to zero mean and unit variance
        - For MIN_MAX normalization: first normalizes to [0,1], then to [-1,1] range
        - For IDENTITY normalization: no transformation is applied
        - Visual features (images) are treated specially with shape adjusted to (channels, 1, 1)
        - Normalization statistics must be properly initialized to avoid infinity values
    """

    def __init__(
        self,
        features: dict[str, Feature],
        norm_map: dict[FeatureType, NormalizationType],
        *,
        inverse: bool = False,
    ) -> None:
        """Initialize the FeatureNormalizeTransform.

        Args:
            features (dict[str, Feature]): Dictionary mapping feature names to Feature objects
                containing shape and normalization data information.
            norm_map (dict[FeatureType, NormalizationType]): Mapping from feature types to their
                corresponding normalization methods (IDENTITY, MEAN_STD, or MIN_MAX).
            inverse (bool, optional): If True, applies inverse normalization (denormalization).
                If False, applies forward normalization. Defaults to False.
        """
        super().__init__()
        self._features = features
        self._norm_map = norm_map
        self._inverse = inverse

        buffers = self._create_stats_buffers(features, norm_map)
        for key, buffer in buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply normalization to features in the input batch.

        This method processes each known features and applies
        the appropriate normalization based on the feature type mapping.
        Features can be located either directly in the batch dictionary or nested
        within sub-dictionaries of the batch.

        Args:
            batch (dict[str, torch.Tensor]): Input batch containing tensors to normalize.
                                           May contain nested dictionaries with features.

        Returns:
            dict[str, torch.Tensor]: The input batch with normalization applied to
                                   the relevant features in-place.
        """
        for raw_name, ft in self._features.items():
            root_dict = {}
            if raw_name in batch:
                root_dict = batch
            else:
                for item in batch.values():
                    if isinstance(item, dict) and ft.name in item:
                        root_dict = item
                        break
            if root_dict:
                key = raw_name
                norm_mode = self._norm_map.get(cast("FeatureType", ft.ftype), NormalizationType.IDENTITY)
                if norm_mode is NormalizationType.IDENTITY:
                    continue
                buffer = getattr(self, "buffer_" + key.replace(".", "_"))
                self._apply_normalization(root_dict, key, norm_mode, buffer, inverse=self._inverse)

        return batch

    @staticmethod
    def _apply_normalization(
        batch: dict,
        key: str,
        norm_mode: NormalizationType,
        buffer: dict,
        *,
        inverse: bool,
    ) -> None:
        def check_inf(t: torch.Tensor, name: str = "") -> None:
            if torch.isinf(t).any():
                msg = (
                    f"Normalization buffer '{name}' is infinity. You should either initialize "
                    "model with correct features stats, or use a pretrained model."
                )
                raise ValueError(
                    msg,
                )

        # Skip normalization if the value is None (e.g., during gym rollouts where action is not available)
        if batch[key] is None:
            return

        if norm_mode is NormalizationType.MEAN_STD:
            mean = buffer["mean"]
            std = buffer["std"]
            check_inf(mean, "mean")
            check_inf(std, "std")
            if inverse:
                batch[key] = batch[key] * std + mean
            else:
                batch[key] = (batch[key] - mean) / (std + 1e-8)

        elif norm_mode is NormalizationType.MIN_MAX:
            min_ = buffer["min"]
            max_ = buffer["max"]
            check_inf(min_, "min")
            check_inf(max_, "max")
            if inverse:
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max_ - min_) + min_
            else:
                # normalize to [0,1]
                batch[key] = (batch[key] - min_) / (max_ - min_ + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
        else:
            raise ValueError(norm_mode)

    @staticmethod
    def _create_stats_buffers(
        features: dict[str, Feature],
        norm_map: dict[FeatureType, NormalizationType],
    ) -> dict[str, dict[str, nn.ParameterDict]]:
        """Create buffers per modality (e.g. "observation.image", "action") containing their normalization statistics.

        Args:
            features: Dictionary mapping feature names to Feature objects containing
                shape and normalization data information.
            norm_map: Dictionary mapping FeatureType to NormalizationType, specifying
                the normalization method for each feature type.

        Returns:
            A dictionary where keys are modalities and values are `nn.ParameterDict` containing
                `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.

        Raises:
            ValueError: If visual features don't have exactly 3 dimensions or
                are not in channel-first format.
            TypeError: If normalization data is not a numpy array or torch tensor.
            TypeError: If normalization mode is not a valid NormalizationType.
        """
        stats_buffers = {}

        for key, ft in features.items():
            norm_mode = norm_map.get(cast("FeatureType", ft.ftype), NormalizationType.IDENTITY)
            if norm_mode is NormalizationType.IDENTITY:
                continue

            if not isinstance(norm_mode, NormalizationType):
                msg = f"Invalid type of normalization mode object: {norm_mode}"
                raise TypeError(msg)

            shape = ft.shape if ft.shape is not None else ()

            if ft.ftype is FeatureType.VISUAL:
                # sanity checks
                visual_feature_len = 3
                if len(shape) != visual_feature_len:
                    msg = f"number of dimensions of {key} != {visual_feature_len} ({shape=})"
                    raise ValueError(msg)

                c, h, w = shape
                if not (c < h and c < w):
                    msg = f"{key} is not channel first ({shape=})"
                    raise ValueError(msg)

                # override image shape to be invariant to height and width
                shape = (c, 1, 1)

            # Note: we initialize mean, std, min, max to infinity. They should be overwritten
            # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
            # we assert they are not infinity anymore.

            def get_torch_tensor(arr: np.ndarray | torch.Tensor | Integral, shape: tuple[int, ...]) -> torch.Tensor:
                if isinstance(arr, np.ndarray):
                    return torch.from_numpy(arr).to(dtype=torch.float32).view(shape)
                if isinstance(arr, torch.Tensor):
                    return arr.clone().to(dtype=torch.float32).view(shape)
                if isinstance(arr, Integral):
                    return torch.tensor(arr, dtype=torch.float32).view(shape)
                type_ = type(arr)
                msg = f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead."
                raise TypeError(msg)

            buffer = {}
            if norm_mode is NormalizationType.MEAN_STD:
                mean = torch.ones(shape, dtype=torch.float32) * torch.inf
                std = torch.ones(shape, dtype=torch.float32) * torch.inf
                buffer = nn.ParameterDict(
                    {
                        "mean": nn.Parameter(mean, requires_grad=False),
                        "std": nn.Parameter(std, requires_grad=False),
                    },
                )
                buffer["mean"].data = get_torch_tensor(
                    cast("NormalizationParameters", ft.normalization_data).mean,
                    shape,
                )
                buffer["std"].data = get_torch_tensor(cast("NormalizationParameters", ft.normalization_data).std, shape)
            elif norm_mode is NormalizationType.MIN_MAX:
                min_ = torch.ones(shape, dtype=torch.float32) * torch.inf
                max_ = torch.ones(shape, dtype=torch.float32) * torch.inf
                buffer = nn.ParameterDict(
                    {
                        "min": nn.Parameter(min_, requires_grad=False),
                        "max": nn.Parameter(max_, requires_grad=False),
                    },
                )
                buffer["min"].data = get_torch_tensor(cast("NormalizationParameters", ft.normalization_data).min, shape)
                buffer["max"].data = get_torch_tensor(cast("NormalizationParameters", ft.normalization_data).max, shape)

            stats_buffers[key] = buffer
        return stats_buffers
