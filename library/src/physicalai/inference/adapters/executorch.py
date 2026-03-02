# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ExecuTorch adapter for inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physicalai.inference.adapters.base import RuntimeAdapter

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class ExecuTorchAdapter(RuntimeAdapter):
    """ExecuTorch inference adapter.

    Provides inference through ExecuTorch runtime for edge devices.
    Supports .pte model files.

    Examples:
        >>> adapter = ExecuTorchAdapter(device="cpu")
        >>> adapter.load(Path("model.pte"))
        >>> outputs = adapter.predict({"input": input_array})
    """

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize ExecuTorch adapter.

        Args:
            device: Device for inference (currently only 'cpu' supported)
            **kwargs: Additional configuration options
        """
        super().__init__(device, **kwargs)
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: Path) -> None:
        """Load ExecuTorch model from file.

        Args:
            model_path: Path to .pte model file

        Raises:
            ImportError: If executorch is not installed
            FileNotFoundError: If model file doesn't exist
        """
        try:
            import executorch.runtime  # noqa: PLC0415
        except ImportError as e:
            msg = "ExecuTorch is not installed. Install with: pip install executorch"
            raise ImportError(msg) from e

        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        rt = executorch.runtime.Runtime.get()
        self.model = rt.load_program(model_path)

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with ExecuTorch.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        method = self.model.load_method("forward")
        raw_outputs = method.execute(list(inputs.values()))
        return dict(zip(self._output_names, raw_outputs, strict=False))

    def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None

    def default_device(self) -> str:
        """Get default device for ExecuTorch runtime.

        Returns:
            'cpu' (ExecuTorch primarily targets edge/CPU inference)
        """
        return "cpu"

    @property
    def input_names(self) -> list[str]:
        """Get model input names.

        Returns:
            List of input names
        """
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Get model output names.

        Returns:
            List of output names
        """
        return self._output_names
