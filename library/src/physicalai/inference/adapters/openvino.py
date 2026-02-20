# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO adapter for inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physicalai.inference.adapters.base import RuntimeAdapter

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import openvino


class OpenVINOAdapter(RuntimeAdapter):
    """OpenVINO inference adapter.

    Provides inference through Intel OpenVINO Runtime, optimized
    for Intel hardware (CPU, GPU, NPU).

    Examples:
        >>> adapter = OpenVINOAdapter(device="CPU")
        >>> adapter.load(Path("model.xml"))
        >>> outputs = adapter.predict({"input": input_array})
    """

    def __init__(self, device: str = "CPU", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize OpenVINO adapter.

        Args:
            device: OpenVINO device ('CPU', 'GPU', 'NPU', 'AUTO')
            **kwargs: Additional OpenVINO compile options
        """
        super().__init__(device, **kwargs)
        self.compiled_model: openvino.CompiledModel | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: Path) -> None:
        """Load OpenVINO model from XML file.

        Args:
            model_path: Path to .xml model file (.bin file should be in same directory)

        Raises:
            ImportError: If OpenVINO is not installed
            FileNotFoundError: If model files don't exist
        """
        try:
            import openvino as ov  # noqa: PLC0415
        except ImportError as e:
            msg = "OpenVINO is not installed. Install with: uv pip install openvino"
            raise ImportError(msg) from e

        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        # Load and compile model
        core = ov.Core()
        model = core.read_model(model=str(model_path))
        self.compiled_model = core.compile_model(model=model, device_name=self.device, config=self.config)

        # Cache input/output names
        self._input_names = [input_node.any_name for input_node in self.compiled_model.inputs]
        self._output_names = [output_node.any_name for output_node in self.compiled_model.outputs]

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with OpenVINO.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            RuntimeError: If model is not loaded
        """
        import numpy as np  # noqa: PLC0415

        if self.compiled_model is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        # Run inference
        results = self.compiled_model(inputs)

        # Convert to dictionary with output names
        return {name: np.array(results[i]) for i, name in enumerate(self._output_names)}

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
