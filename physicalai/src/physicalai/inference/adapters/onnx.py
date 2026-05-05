# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime adapter for inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.registry import adapter_registry

if TYPE_CHECKING:
    from pathlib import Path

    import onnxruntime


@adapter_registry.register("onnx", extensions=(".onnx",))
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
        self._input_dtypes: dict[str, np.dtype] = {}

    def load(self, model_path: Path) -> None:
        """Load ONNX model from file.

        Args:
            model_path: Path to .onnx model file

        Raises:
            ImportError: If onnxruntime is not installed
            FileNotFoundError: If model file doesn't exist
        """
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as e:
            msg = "ONNX Runtime is not installed. Install with: uv pip install onnxruntime"
            raise ImportError(msg) from e

        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        # Configure providers based on device
        providers = self._get_providers()

        # Create inference session
        self.session = ort.InferenceSession(str(model_path), providers=providers, **self.config)

        # Cache input/output names
        self._input_names = [input_meta.name for input_meta in self.session.get_inputs()]
        self._output_names = [output_meta.name for output_meta in self.session.get_outputs()]
        self._input_dtypes = {
            input_meta.name: dtype
            for input_meta in self.session.get_inputs()
            if (dtype := _onnx_type_to_numpy_dtype(input_meta.type)) is not None
        }

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with ONNX Runtime.

        Matches input names to session metadata, coerces dtypes as needed.

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

        coerced: dict[str, np.ndarray] = {}
        for name, array in inputs.items():
            expected = self._input_dtypes.get(name)
            if expected is not None and array.dtype != expected:
                coerced[name] = array.astype(expected, copy=False)
            else:
                coerced[name] = array

        raw_outputs = self.session.run(self._output_names, coerced)
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

    def default_device(self) -> str:  # noqa: PLR6301
        """Get default ONNX Runtime device.

        Returns:
            'cpu' (consistent with other adapters' default behavior)
        """
        return "cpu"

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


_ONNX_TYPE_TO_NUMPY: dict[str, str] = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(float16)": "float16",
    "tensor(bfloat16)": "bfloat16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(int16)": "int16",
    "tensor(int8)": "int8",
    "tensor(uint64)": "uint64",
    "tensor(uint32)": "uint32",
    "tensor(uint16)": "uint16",
    "tensor(uint8)": "uint8",
    "tensor(bool)": "bool",
}


def _onnx_type_to_numpy_dtype(onnx_type: str) -> np.dtype | None:
    """Map an ONNX Runtime input type string to a numpy dtype.

    Args:
        onnx_type: Type string reported by ``InferenceSession`` metadata
            (e.g. ``"tensor(float)"``).

    Returns:
        Matching numpy dtype, or ``None`` if the type is not recognized
        (e.g. string/sequence inputs), in which case callers should pass
        the array through unchanged.
    """
    np_name = _ONNX_TYPE_TO_NUMPY.get(onnx_type)
    if np_name is None:
        return None
    return np.dtype(np_name)
