# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch runtime adapter for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from physicalai.data.observation import Observation
from physicalai.policies import get_physicalai_policy_class as get_policy_class

from .base import RuntimeAdapter


class TorchAdapter(RuntimeAdapter):
    """Runtime adapter for Torch models.

    This adapter loads and runs models exported via ``to_torch()``
    using PyTorch's API.  It accepts the same ``dict[str, np.ndarray]``
    input contract as every other adapter.  Internally it converts
    numpy arrays to torch tensors, wraps them in an ``Observation``
    dataclass and forwards them to the policy.

    Example:
        >>> adapter = TorchAdapter()
        >>> adapter.load("model.pt")
        >>> outputs = adapter.predict({"state": state_array, "images": images_dict})
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

            # Derive granular input names from the policy's sample_input.
            # sample_input returns e.g. {"state": tensor, "images": tensor}
            # which are the real observation keys the model consumes.
            self._input_names = list(self._policy.model.sample_input.keys())
            self._output_names = self._policy.model.extra_export_args["torch"]["output_names"]

        except Exception as e:
            msg = f"Failed to load Torch model from {model_path}: {e}"
            raise RuntimeError(msg) from e

    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run inference using Torch.

        Accepts ``dict[str, np.ndarray]`` (same contract as all other
        adapters).  Values may be plain numpy arrays *or* dicts of numpy
        arrays (e.g. multi-camera images).  The method converts them to
        torch tensors, wraps them in an :class:`Observation` and passes
        the result to the loaded policy.

        Args:
            inputs: Dictionary mapping input names to numpy arrays
                (or dicts of numpy arrays for nested fields like images).

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            RuntimeError: If model is not loaded or inference fails
        """
        if self._policy is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        try:
            # Build Observation from numpy dict and convert to torch tensors on device
            observation = Observation.from_dict(inputs).to_torch(self.device)

            # Run policy forward pass
            torch_outputs = self._policy(observation)
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
            return {self._output_names[0]: torch_outputs.detach().cpu().numpy()}
        if isinstance(torch_outputs, dict):
            # Dict output
            return {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in torch_outputs.items()}
        if isinstance(torch_outputs, (list, tuple)):
            # Multiple outputs as list/tuple
            outputs_iter = zip(self._output_names, torch_outputs, strict=True)
            return {name: out.detach().cpu().numpy() for name, out in outputs_iter}

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
