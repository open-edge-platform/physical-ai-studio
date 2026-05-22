# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for defining exportable PyTorch models."""

from abc import ABC, abstractmethod

import torch


class ExportableModelMixin(torch.nn.Module, ABC):
    """Mixin class for exportable PyTorch models.

    This mixin provides a common interface and utilities for PyTorch models that can be exported
    to various formats (e.g., ONNX, OpenVINO). It is designed to be used in conjunction with the
    base Model class to enable seamless integration of export functionality.
    """

    @property
    @abstractmethod
    def sample_input(self) -> dict[str, torch.Tensor]:
        """Return a sample input dictionary for the model.

        This sample input is used during the export process to trace the model's computation graph.
        It should contain example tensors that match the expected input format of the model.

        Returns:
            A dictionary mapping input names to example torch.Tensor objects.
        """
