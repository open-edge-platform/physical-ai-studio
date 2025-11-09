# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ExecuTorch (Torch Export IR) runtime adapter for inference."""

from pathlib import Path

import numpy as np
import torch

from .base import RuntimeAdapter


class ExecuTorchAdapter(RuntimeAdapter):
    """Runtime adapter for ExecuTorch (Torch Export IR) models.

    This adapter loads and runs models exported via `to_torch_export_ir()`.
    ExecuTorch is designed for edge and mobile deployment with optimized
    execution for resource-constrained devices.

    Note:
        This adapter is implemented but will be fully tested with real policy
        exports in subsequent PRs. The implementation follows the same pattern
        as other adapters (OpenVINO, ONNX, TorchScript).

    Example:
        >>> adapter = ExecuTorchAdapter()
        >>> adapter.load("model.ptir")
        >>> outputs = adapter.predict({"image": image_array, "state": state_array})
    """

    def __init__(self) -> None:
        """Initialize the ExecuTorch adapter."""
        self._program = None
        self._module = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: Path) -> None:
        """Load ExecuTorch program from file.

        Args:
            model_path: Path to the .ptir file created by torch.export.save()

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        try:
            program = torch.export.load(str(model_path))  # nosec B614
            self._program = program
            self._module = program.module()

            # Extract input/output names from the graph signature
            graph_signature = program.graph_signature
            self._input_names = [str(name) for name in graph_signature.user_inputs]
            self._output_names = [str(name) for name in graph_signature.user_outputs]

        except Exception as e:
            msg = f"Failed to load ExecuTorch model from {model_path}: {e}"
            raise RuntimeError(msg) from e

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference using ExecuTorch runtime.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            RuntimeError: If model is not loaded or inference fails
            ValueError: If input names don't match model expectations
        """
        if self._module is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        # Validate input names
        missing_inputs = set(self._input_names) - set(inputs.keys())
        if missing_inputs:
            msg = f"Missing required inputs: {missing_inputs}. Expected: {self._input_names}"
            raise ValueError(msg)

        try:
            # Convert numpy arrays to torch tensors
            torch_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}

            # Run inference - module expects dict input
            with torch.no_grad():
                torch_outputs = self._module(torch_inputs)

            # Handle different output formats
            return self._convert_outputs_to_numpy(torch_outputs)

        except Exception as e:
            msg = f"Inference failed: {e}"
            raise RuntimeError(msg) from e

    def _convert_outputs_to_numpy(self, torch_outputs: torch.Tensor | dict | list | tuple) -> dict[str, np.ndarray]:
        """Convert model outputs to numpy format.

        Args:
            torch_outputs: Model outputs (tensor, dict, list, or tuple)

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            TypeError: If output type is unexpected
        """
        if isinstance(torch_outputs, torch.Tensor):
            # Single output
            return {self._output_names[0]: torch_outputs.numpy()}
        if isinstance(torch_outputs, dict):
            # Dict output
            return {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in torch_outputs.items()}
        if isinstance(torch_outputs, (list, tuple)):
            # Multiple outputs as list/tuple
            return {name: output.numpy() for name, output in zip(self._output_names, torch_outputs, strict=True)}

        # Unexpected output type
        msg = f"Unexpected output type: {type(torch_outputs)}"
        raise TypeError(msg)

    @property
    def input_names(self) -> list[str]:
        """Get model input names.

        Returns:
            List of input tensor names

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._module is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Get model output names.

        Returns:
            List of output tensor names

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._module is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)
        return self._output_names
