# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime adapter for inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from getiaction.inference.adapters.base import RuntimeAdapter

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import onnxruntime


class ONNXAdapter(RuntimeAdapter):
    """ONNX Runtime inference adapter.

    Provides cross-platform inference through ONNX Runtime.
    Supports CPU and GPU acceleration.

    Examples:
        >>> adapter = ONNXAdapter(device="cpu")
        >>> adapter.load(Path("model.onnx"))
        >>> outputs = adapter.predict({"input": input_array})
    """

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize ONNX adapter.

        Args:
            device: Device for inference ('cpu', 'cuda', 'tensorrt')
            **kwargs: Additional ONNX Runtime session options
        """
        super().__init__(device, **kwargs)
        self.session: onnxruntime.InferenceSession | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: Path) -> None:
        """Load ONNX model from file.

        Args:
            model_path: Path to .onnx model file

        Raises:
            ImportError: If onnxruntime is not installed
            FileNotFoundError: If model file doesn't exist
        """
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as e:
            msg = "ONNX Runtime is not installed. Install with: uv pip install onnxruntime"
            raise ImportError(msg) from e

        # Configure providers based on device
        providers = self._get_providers()

        # Create inference session
        self.session = ort.InferenceSession(str(model_path), providers=providers, **self.config)

        # Cache input/output names
        self._input_names = [input_meta.name for input_meta in self.session.get_inputs()]
        self._output_names = [output_meta.name for output_meta in self.session.get_outputs()]

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with ONNX Runtime.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.session is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        # Run inference - outputs is a list that can contain various types
        # For typical models, these are numpy arrays
        raw_outputs = self.session.run(self._output_names, inputs)
        outputs = cast("list[np.ndarray]", raw_outputs)

        # Convert to dictionary
        return dict(zip(self._output_names, outputs, strict=False))

    def _get_providers(self) -> list[str]:
        """Get ONNX Runtime providers based on device.

        Returns:
            List of provider names in priority order
        """
        device_lower = self.device.lower()

        if device_lower in {"cuda", "gpu"}:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device_lower == "tensorrt":
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]

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
