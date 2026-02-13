# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Production-ready inference model with unified API.

This module is torch-free - it works with numpy arrays only.
For torch tensor support, install getiaction[torch].
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import yaml

from getiaction.export.types import ExportBackend

if TYPE_CHECKING:
    from getiaction.inference.adapters.base import RuntimeAdapter


@runtime_checkable
class ObservationLike(Protocol):
    """Protocol for observation objects that can be converted to dict.

    This allows InferenceModel.select_action() to accept either:
    - dict[str, Any]: Direct observation dictionary
    - ObservationLike: Any object with a to_dict() method (e.g., Observation from getiaction.data)
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert observation to dictionary format."""
        ...


STATE = "state"
IMAGES = "images"
ACTION = "action"
TASK = "task"


class InferenceModel:
    """Unified inference interface for exported policies.

    Automatically detects backend and provides consistent API across
    all export formats (OpenVINO, ONNX, Torch Export IR).

    This class is torch-free and works with numpy arrays only.
    For torch tensor input/output, convert manually or use the Observation
    class from getiaction.data (requires getiaction[torch]).

    The interface provides:
    - `select_action(obs)` - Get action from observation dict
    - `reset()` - Reset policy state for new episode

    Examples:
        >>> policy = InferenceModel.load("./exports/act_policy")
        >>> policy.reset()
        >>> obs = {"state": np.array([...]), "images": np.array([...])}
        >>> action = policy.select_action(obs)  # Returns np.ndarray
    """

    def __init__(
        self,
        export_dir: str | Path,
        policy_name: str | None = None,
        backend: str | ExportBackend = "auto",
        device: str = "auto",
        **adapter_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize InferenceModel with optional auto-detection.

        Args:
            export_dir: Directory containing exported policy files
            policy_name: Policy name (auto-detected if None)
            backend: Backend to use, or 'auto' to detect from metadata/files
            device: Device for inference ('auto', 'cpu', 'cuda', 'CPU', 'GPU', etc.)
            **adapter_kwargs: Backend-specific configuration options

        Raises:
            FileNotFoundError: If export directory or required files don't exist
        """
        self.export_dir = Path(export_dir)
        if not self.export_dir.exists():
            msg = f"Export directory not found: {export_dir}"
            raise FileNotFoundError(msg)

        self.metadata = self._load_metadata()

        if policy_name is None:
            policy_name = self._detect_policy_name()
        self.policy_name = policy_name

        if backend == "auto":
            backend = self.metadata.get("backend") or self._detect_backend()
        self.backend = ExportBackend(backend) if isinstance(backend, str) else backend

        if device == "auto":
            device = self._detect_device()
        self.device = device

        from getiaction.inference.adapters import get_adapter  # noqa: PLC0415

        self.adapter: RuntimeAdapter = get_adapter(self.backend, device=device, **adapter_kwargs)
        model_path = self._get_model_path()
        self.adapter.load(model_path)

        self._action_queue: deque[np.ndarray] = deque()
        self.use_action_queue = self.metadata.get("use_action_queue", False)
        self.chunk_size = self.metadata.get("chunk_size", 1)

    @classmethod
    def load(
        cls,
        export_dir: str | Path,
        **kwargs: Any,  # noqa: ANN401
    ) -> InferenceModel:
        """Load inference model with auto-detection.

        Convenience constructor that automatically detects all parameters
        from the export directory.

        Args:
            export_dir: Directory containing exported policy files
            **kwargs: Additional arguments passed to __init__

        Returns:
            Initialized InferenceModel instance
        """
        return cls(export_dir=export_dir, **kwargs)

    def select_action(self, observation: dict[str, Any] | ObservationLike) -> np.ndarray:
        """Select action for given observation.

        For chunked policies (chunk_size > 1), manages action queue
        automatically and returns one action at a time.

        Args:
            observation: Dict mapping input names to numpy arrays, or an object
                with to_dict() method (e.g., Observation from getiaction.data).
                Example: {"state": np.array(...), "images": np.array(...)}

        Returns:
            Action as numpy array. Shape: (action_dim,) or (batch, action_dim)
        """
        if self.use_action_queue and len(self._action_queue) > 0:
            return self._action_queue.popleft()

        inputs = self._prepare_inputs(observation)
        outputs = self.adapter.predict(inputs)

        action_key = self._get_action_output_key(outputs)
        actions = outputs[action_key]

        if self.use_action_queue and self.chunk_size > 1:
            batch_actions = np.moveaxis(actions, 1, 0)
            self._action_queue.extend(batch_actions)
            return self._action_queue.popleft()

        temporal_dim = 3
        if actions.ndim == temporal_dim and actions.shape[1] == 1:
            actions = np.squeeze(actions, axis=1)

        return actions

    def __call__(self, observation: dict[str, Any] | ObservationLike) -> dict[str, np.ndarray]:
        """Run inference and return all outputs."""
        inputs = self._prepare_inputs(observation)
        return self.adapter.predict(inputs)

    def reset(self) -> None:
        """Reset policy state for new episode. Call at episode start."""
        self._action_queue.clear()

    def _prepare_inputs(
        self,
        observation: dict[str, Any] | ObservationLike,
    ) -> dict[str, Any]:
        """Convert observation to model input format, raising ValueError if inputs missing."""
        obs_dict: dict[str, Any] = observation.to_dict() if isinstance(observation, ObservationLike) else observation

        if "input_names" in self.metadata:
            expected_input_names = self.metadata["input_names"]
            adapter_input_names = list(self.adapter.input_names)
        else:
            expected_input_names = list(self.adapter.input_names)
            adapter_input_names = expected_input_names

        expected_inputs = set(expected_input_names)

        if expected_inputs == {"observation"}:
            # Torch adapter expects raw unpacked observation dict
            # and reconstructs the Observation object internally
            return obs_dict

        field_mapping = self._build_field_mapping(obs_dict, expected_inputs)

        inputs = {}
        for obs_key, model_key in field_mapping.items():
            value = obs_dict.get(obs_key)

            if value is None:
                continue

            if obs_key == "images" and isinstance(value, list):
                if len(value) > 0:
                    value = value[0]
                else:
                    continue

            if isinstance(value, np.ndarray):
                inputs[model_key] = value
            else:
                inputs[model_key] = np.array(value)

        missing_inputs = expected_inputs - set(inputs.keys())
        if missing_inputs:
            available_fields = list(obs_dict.keys())
            msg = f"Missing required model inputs: {missing_inputs}. Available observation fields: {available_fields}"
            raise ValueError(msg)

        if expected_input_names != adapter_input_names:
            return {adapter_input_names[i]: inputs[expected_input_names[i]] for i in range(len(expected_input_names))}

        return inputs

    @staticmethod
    def _build_field_mapping(obs_dict: dict[str, Any], expected_inputs: set[str]) -> dict[str, str]:
        """Build mapping from observation fields to model input names."""
        mapping = {key: key for key in obs_dict if key in expected_inputs}

        if len(mapping) == len(expected_inputs):
            return mapping

        obs_fields = {
            STATE: [STATE, f"observation.{STATE}", f"observation_{STATE}"],
            IMAGES: [
                IMAGES,
                "image",
                "observation.image",
                f"observation.{IMAGES}",
                "observation_image",
                f"observation_{IMAGES}",
            ],
            ACTION: [ACTION],
        }

        for obs_key, possible_model_keys in obs_fields.items():
            if obs_key not in obs_dict:
                continue

            for model_key in possible_model_keys:
                if model_key in expected_inputs:
                    mapping[obs_key] = model_key
                    break

        return mapping

    @staticmethod
    def _get_action_output_key(outputs: dict[str, np.ndarray]) -> str:
        """Determine which output contains actions."""
        action_keys = ["actions", "action", "output", "pred_actions"]

        for key in action_keys:
            if key in outputs:
                return key

        return next(iter(outputs))

    def _load_metadata(self) -> dict[str, Any]:
        """Load export metadata from yaml or json file."""
        yaml_path = self.export_dir / "metadata.yaml"
        if yaml_path.exists():
            with yaml_path.open() as f:
                return yaml.safe_load(f) or {}

        json_path = self.export_dir / "metadata.json"
        if json_path.exists():
            with json_path.open() as f:
                return json.load(f)

        return {}

    def _detect_policy_name(self) -> str:
        """Auto-detect policy name from files or metadata, raising ValueError if undetermined."""
        if "policy_class" in self.metadata:
            class_path = self.metadata["policy_class"]
            parts = class_path.lower().split(".")
            min_parts_for_module_extraction = 3
            if len(parts) >= min_parts_for_module_extraction:
                return parts[-2]

        model_files = list(self.export_dir.glob("*.*"))
        if model_files:
            name = model_files[0].stem
            for suffix in ["_policy", "_model"]:
                name = name.removesuffix(suffix)
            return name

        msg = f"Cannot determine policy name from {self.export_dir}"
        raise ValueError(msg)

    def _detect_backend(self) -> str:
        """Auto-detect backend from model files, raising ValueError if unsupported."""
        extension_map = {
            ".xml": "openvino",
            ".onnx": "onnx",
            ".pt2": "torch_export_ir",
            ".ptir": "torch_export_ir",
            ".ckpt": "torch",
            ".pt": "torch",
        }

        for ext, backend in extension_map.items():
            if list(self.export_dir.glob(f"*{ext}")):
                return backend

        msg = f"Cannot detect backend from files in {self.export_dir}"
        raise ValueError(msg)

    def _detect_device(self) -> str:
        """Auto-detect best available device for the backend."""
        if self.backend == ExportBackend.OPENVINO:
            return "CPU"

        if self.backend == ExportBackend.ONNX:
            try:
                import onnxruntime as ort  # noqa: PLC0415

                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    return "cuda"
            except ImportError:
                pass

        return "cpu"

    def _get_model_path(self) -> Path:
        """Get path to model file, raising FileNotFoundError if not found."""
        extension_map = {
            ExportBackend.OPENVINO: [".xml"],
            ExportBackend.ONNX: [".onnx"],
            ExportBackend.TORCH_EXPORT_IR: [".pt2", ".ptir"],
            ExportBackend.TORCH: [".ckpt", ".pt"],
        }

        extensions = extension_map[self.backend]

        if self.policy_name:
            for ext in extensions:
                model_path = self.export_dir / f"{self.policy_name}{ext}"
                if model_path.exists():
                    return model_path

        for ext in extensions:
            files = list(self.export_dir.glob(f"*{ext}"))
            if files:
                return files[0]

        ext_str = " or ".join(extensions)
        msg = f"No {ext_str} model file found in {self.export_dir}"
        raise FileNotFoundError(msg)

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"{self.__class__.__name__}(policy={self.policy_name}, backend={self.backend.value}, device={self.device})"
        )
