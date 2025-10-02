# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utils for dataset features normalization."""


import numpy as np
import torch
from torch import nn

from getiaction.data import (
    Feature,
    FeatureType,
    NormalizationType,
)


class FeatureNormalizeTransform(nn.Module):
    def __init__(
        self,
        features: dict[str, Feature],
        norm_map: dict[FeatureType, NormalizationType],
        inverse: bool = False,
    ) -> None:
        super().__init__()
        self._features = features
        self._norm_map = norm_map
        self._inverse = inverse

        buffers = self._create_stats_buffers(features, norm_map)
        for key, buffer in buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
                norm_mode = self._norm_map.get(ft.ftype, NormalizationType.IDENTITY)
                if norm_mode is NormalizationType.IDENTITY:
                    continue
                buffer = getattr(self, "buffer_" + key.replace(".", "_"))
                self._apply_normalization(root_dict, key, norm_mode, buffer, self._inverse)

        return batch

    @staticmethod
    def _apply_normalization(batch, key, norm_mode, buffer, inverse):
        def check_inf(t: torch.Tensor, name: str = "") -> None:
            if torch.isinf(t).any():
                raise ValueError(
                    f"Normalization buffer '{name}' is infinity. You should either initialize "
                    "model with correct features stats, or use a pretrained model."
                )

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
            min = buffer["min"]
            max = buffer["max"]
            check_inf(min, "min")
            check_inf(max, "max")
            if inverse:
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min) + min
            else:
                # normalize to [0,1]
                batch[key] = (batch[key] - min) / (max - min + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
        else:
            raise ValueError(norm_mode)

    @staticmethod
    def _create_stats_buffers(
        features: dict[str, Feature],
        norm_map: dict[FeatureType, NormalizationType],
    ) -> dict[str, dict[str, nn.ParameterDict]]:
        """Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
        statistics.

        Args: (see Normalize and Unnormalize)

        Returns:
            dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
                `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
        """
        stats_buffers = {}

        for key, ft in features.items():
            norm_mode = norm_map.get(ft.ftype, NormalizationType.IDENTITY)
            if norm_mode is NormalizationType.IDENTITY:
                continue

            assert isinstance(norm_mode, NormalizationType)

            shape = ft.shape

            if ft.ftype is FeatureType.VISUAL:
                # sanity checks
                assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
                c, h, w = shape
                assert c < h and c < w, f"{key} is not channel first ({shape=})"
                # override image shape to be invariant to height and width
                shape = (c, 1, 1)

            # Note: we initialize mean, std, min, max to infinity. They should be overwritten
            # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
            # we assert they are not infinity anymore.

            def get_torch_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
                if isinstance(arr, np.ndarray):
                    return torch.from_numpy(arr).to(dtype=torch.float32)
                if isinstance(arr, torch.Tensor):
                    return arr.clone().to(dtype=torch.float32)
                type_ = type(arr)
                raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead.")

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
                buffer["mean"].data = get_torch_tensor(ft.normalization_data.mean)
                buffer["std"].data = get_torch_tensor(ft.normalization_data.std)
            elif norm_mode is NormalizationType.MIN_MAX:
                min = torch.ones(shape, dtype=torch.float32) * torch.inf
                max = torch.ones(shape, dtype=torch.float32) * torch.inf
                buffer = nn.ParameterDict(
                    {
                        "min": nn.Parameter(min, requires_grad=False),
                        "max": nn.Parameter(max, requires_grad=False),
                    },
                )
                buffer["min"].data = get_torch_tensor(ft.normalization_data.min)
                buffer["max"].data = get_torch_tensor(ft.normalization_data.max)

            stats_buffers[key] = buffer
        return stats_buffers
