# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch runtime adapter for inference."""

from pathlib import Path

import numpy as np
import torch
import yaml

from getiaction.data.observation import Observation
from getiaction.policies import get_getiaction_policy_class as get_policy_class

from .base import RuntimeAdapter


class TorchAdapter(RuntimeAdapter):
    """Runtime adapter for Torch models.

    This adapter loads and runs models exported via `to_torch()`
    using PyTorch's API.

    Example:
        >>> adapter = TorchAdapter()
        >>> adapter.load("model.pt")
        >>> outputs = adapter.predict({"image": image_array, "state": state_array})
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        """Initialize the Torch adapter.

        Args:
            device: Device for inference ('cpu', 'cuda', 'xpu', etc.)
        """
        self.device = torch.device(device)
        self._policy: torch.nn.Module | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: Path | str) -> None:
        """Load Torch model from file.

        Args:
            model_path: Path to the .pt file created by torch.save()

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
            KeyError: If metadata is missing required entries
        """
        model_path = Path(model_path)
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        metadata_path = model_path.parent / "metadata.yaml"
        if not metadata_path.exists():
            msg = f"Metadata file not found: {metadata_path}"
            raise FileNotFoundError(msg)

        with metadata_path.open() as f:
            metadata = yaml.safe_load(f)

            policy_class_path = metadata.get("policy_class", None)
            if policy_class_path is None:
                msg = "Metadata missing 'policy_class' entry."
                raise KeyError(msg)

        try:
            _, class_name = policy_class_path.rsplit(".", 1)
            policy_class = get_policy_class(class_name)

            self._policy = policy_class.load_from_checkpoint(model_path, map_location="cpu").to(self.device).eval()

            self._input_names = list(self._policy.model.extra_export_args["torch"]["input_names"])
            self._output_names = self._policy.model.extra_export_args["torch"]["output_names"]

        except Exception as e:
            msg = f"Failed to load Torch model from {model_path}: {e}"
            raise RuntimeError(msg) from e

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference using Torch.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            RuntimeError: If model is not loaded
            TypeError: If observation data is not a dict
        """
        if self._policy is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        # Extract observation data from inputs dict
        # Handle both "observation" (lowercase) and "Observation" (uppercase) keys
        if "observation" in inputs:
            obs_data = inputs["observation"]
        elif "Observation" in inputs:
            obs_data = inputs["Observation"]
        else:
            # If no observation key, assume the entire dict is the observation
            obs_data = inputs

        # Validate obs_data is a dict
        if not isinstance(obs_data, dict):
            msg = f"Expected dict for observation data, got {type(obs_data)}"
            raise TypeError(msg)

        # Reconstruct Observation from numpy dict
        observation = Observation.from_dict(obs_data)

        # Convert numpy arrays to torch tensors and move to device
        observation = observation.to_torch(self.device)

        # Run policy inference
        torch_outputs = self._policy(observation)
        return self._convert_outputs_to_numpy(torch_outputs)

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
        if self._policy is None:
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
        if self._policy is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)
        return self._output_names
