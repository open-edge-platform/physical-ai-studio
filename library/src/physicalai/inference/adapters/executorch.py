# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ExecuTorch runtime adapter for inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import yaml

from .base import RuntimeAdapter

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

logger = logging.getLogger(__name__)


class ExecuTorchAdapter(RuntimeAdapter):
    """Runtime adapter for ExecuTorch .pte model inference.

    This adapter loads and runs models exported for ExecuTorch runtime using
    the `.pte` format. Input and output names are read from `metadata.yaml`
    colocated with the model when available. Inference expects dictionary
    inputs and converts them to the ordered tensor list required by
    ExecuTorch's `method.execute(...)` API.

    Examples:
        >>> adapter = ExecuTorchAdapter()
        >>> adapter.load(Path("model.pte"))
        >>> outputs = adapter.predict({"state": state_array})
    """

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize ExecuTorchAdapter.

        Args:
            device: Device hint for inference runtime configuration.
            **kwargs: Additional backend-specific adapter options.
        """
        super().__init__(device, **kwargs)
        self._program: Any = None
        self._method: Any = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: Path) -> None:
        """Load .pte model and optional metadata.

        Args:
            model_path: Path to the ExecuTorch `.pte` model file.

        Raises:
            FileNotFoundError: If the model path does not exist.
            ImportError: If the `executorch` package is not installed.
            RuntimeError: If program or method loading fails.
        """
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        try:
            from executorch.runtime import Runtime  # noqa: PLC0415
        except ImportError as exc:
            msg = "executorch package required for ExecuTorchAdapter. Install with: uv pip install executorch"
            raise ImportError(msg) from exc

        try:
            runtime = Runtime.get()
            self._program = runtime.load_program(model_path)
            self._method = self._program.load_method("forward")
        except (RuntimeError, OSError) as exc:
            msg = f"Failed to load ExecuTorch program from {model_path}: {exc}"
            raise RuntimeError(msg) from exc

        metadata_path = model_path.parent / "metadata.yaml"
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = yaml.safe_load(handle) or {}

                input_names = metadata.get("input_names", [])
                output_names = metadata.get("output_names", [])
                self._input_names = [str(name) for name in input_names]
                self._output_names = [str(name) for name in output_names]
            except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
                logger.warning("Failed to read metadata from %s: %s", metadata_path, exc)
                self._input_names = []
                self._output_names = []

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference.

        Args:
            inputs: Mapping of input names to numpy arrays.

        Returns:
            Mapping of output names to numpy arrays.

        Raises:
            RuntimeError: If model has not been loaded.
            ValueError: If required named inputs are missing.
        """
        if self._method is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        if self._input_names:
            missing_inputs = [name for name in self._input_names if name not in inputs]
            if missing_inputs:
                msg = f"Missing required inputs: {missing_inputs}. Expected: {self._input_names}"
                raise ValueError(msg)
            ordered_inputs = [
                value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
                for value in (inputs[name] for name in self._input_names)
            ]
        else:
            ordered_inputs = [
                value if isinstance(value, torch.Tensor) else torch.from_numpy(value) for value in inputs.values()
            ]

        outputs = self._method.execute(ordered_inputs)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        if self._output_names and len(self._output_names) == len(outputs):
            names = self._output_names
        else:
            names = [f"output_{idx}" for idx in range(len(outputs))]

        result: dict[str, np.ndarray] = {}
        for name, output in zip(names, outputs, strict=True):
            result[name] = output.numpy() if isinstance(output, torch.Tensor) else output

        return result

    @property
    def input_names(self) -> list[str]:
        """Get model input names."""
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Get model output names."""
        return self._output_names

    def default_device(self) -> str:  # noqa: PLR6301
        """Get default ExecuTorch device.

        Returns:
            str: The default device string for ExecuTorch runtime.
        """
        return "cpu"
