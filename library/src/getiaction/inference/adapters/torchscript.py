# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TorchScript adapter for inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from getiaction.inference.adapters.base import RuntimeAdapter

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import torch


class TorchScriptAdapter(RuntimeAdapter):
    """TorchScript inference adapter.

    Provides inference through PyTorch's TorchScript format.
    Supports CPU and CUDA acceleration.

    Examples:
        >>> adapter = TorchScriptAdapter(device="cuda")
        >>> adapter.load(Path("model.pt"))
        >>> outputs = adapter.predict({"input": input_array})
    """

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize TorchScript adapter.

        Args:
            device: Device for inference ('cpu', 'cuda')
            **kwargs: Additional configuration options
        """
        super().__init__(device, **kwargs)
        self.model: torch.jit.ScriptModule | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = ["output"]  # TorchScript doesn't preserve names

    def load(self, model_path: Path) -> None:
        """Load TorchScript model from file.

        Args:
            model_path: Path to .pt model file

        Raises:
            ImportError: If torch is not installed
            FileNotFoundError: If model file doesn't exist
        """
        try:
            import torch  # noqa: PLC0415
        except ImportError as e:
            msg = "PyTorch is not installed. Install with: uv pip install torch"
            raise ImportError(msg) from e

        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        # Load model
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

        # Try to extract input names from model if available
        try:
            # Get input names from the first forward method signature
            self._input_names = list(self.model.forward.schema.arguments[1:])  # Skip 'self'
        except (AttributeError, IndexError):
            # Fallback to generic names if schema is not available
            self._input_names = ["input"]

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with TorchScript.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            RuntimeError: If model is not loaded
        """
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if self.model is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        # Convert numpy inputs to torch tensors
        torch_inputs = {name: torch.from_numpy(array).to(self.device) for name, array in inputs.items()}

        # Run inference
        with torch.no_grad():
            # TorchScript models typically take positional arguments
            input_values = [torch_inputs[name] for name in self._input_names if name in torch_inputs]
            outputs = self.model(input_values[0]) if len(input_values) == 1 else self.model(*input_values)

        # Handle single or multiple outputs
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        elif isinstance(outputs, dict):
            return {name: tensor.cpu().numpy() for name, tensor in outputs.items()}

        # Convert to dictionary with output names
        return {name: tensor.cpu().numpy() for name, tensor in zip(self._output_names, outputs, strict=False)}

    @property
    def input_names(self) -> list[str]:
        """Get input tensor names.

        Returns:
            List of input names
        """
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Get output tensor names.

        Returns:
            List of output names
        """
        return self._output_names
