# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base adapter interface for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class RuntimeAdapter(ABC):
    """Abstract base class for backend-specific inference adapters.

    Adapters provide a unified interface for running inference across
    different backends (OpenVINO, ONNX, TensorRT, etc.).

    Each adapter is responsible for:
    - Loading model files from disk
    - Running inference with backend-specific APIs
    - Converting inputs/outputs to/from numpy arrays
    """

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the adapter.

        Args:
            device: Device to run inference on ('cpu', 'cuda', 'gpu', etc.)
            **kwargs: Backend-specific configuration options
        """
        self.device = device
        self.config = kwargs
        self.model = None

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load model from disk.

        Args:
            model_path: Path to the model file
        """

    @abstractmethod
    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on the model.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays
        """

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """Get model input names.

        Returns:
            List of input tensor names
        """

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Get model output names.

        Returns:
            List of output tensor names
        """

    def __repr__(self) -> str:
        """Return string representation of the adapter."""
        return f"{self.__class__.__name__}(device={self.device})"
