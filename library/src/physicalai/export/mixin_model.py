# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for defining exportable PyTorch models."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class ExportableModel(torch.nn.Module, ABC):
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

    @property
    @abstractmethod
    def extra_export_args(self) -> dict[str, Any]:
        """Return extra arguments for the export process.

        This method can be overridden to provide additional arguments that may be required by specific
        export formats or tools. The returned dictionary can include any relevant information that
        should be considered during export.

        Returns:
            A dictionary of extra arguments for the export process.
        """
