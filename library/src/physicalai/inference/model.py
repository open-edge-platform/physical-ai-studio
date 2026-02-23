# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Production-ready inference model with unified API."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

from physicalai.data.observation import ACTION, IMAGES, STATE
from physicalai.export import ExportBackend
from physicalai.inference.adapters import get_adapter

if TYPE_CHECKING:
    import numpy as np

    from physicalai.data import Observation
    from physicalai.inference.adapters.base import RuntimeAdapter


class InferenceModel:
    """Unified inference interface for exported policies.

    Automatically detects backend and provides consistent API across
    all export formats (OpenVINO, ONNX, Torch Export IR).

    The interface matches PyTorch policy API:
    - `select_action(obs)` - Get action from observation
    - `reset()` - Reset policy state for new episode

    Examples:
        >>> # Auto-detect everything
        >>> policy = InferenceModel.load("./exports/act_policy")
        >>> policy.reset()
        >>> action = policy.select_action(obs)

        >>> # Explicit backend and device
        >>> policy = InferenceModel(
        ...     export_dir="./exports",
        ...     policy_name="act",
        ...     backend="openvino",
        ...     device="CPU"
        ... )
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

        # Load metadata
        self.metadata = self._load_metadata()

        # Auto-detect policy name if not specified
        if policy_name is None:
            policy_name = self._detect_policy_name()
        self.policy_name = policy_name

        # Auto-detect backend if not specified
        if backend == "auto":
            backend = self.metadata.get("backend") or self._detect_backend()
        self.backend = ExportBackend(backend) if isinstance(backend, str) else backend

        # Auto-detect device if not specified
        if device == "auto":
            device = self._detect_device()
        self.device = device

        # Create and load adapter
        self.adapter: RuntimeAdapter = get_adapter(self.backend, device=device, **adapter_kwargs)
        model_path = self._get_model_path()
        self.adapter.load(model_path)

        # State management for stateful policies
        self._action_queue: deque[torch.Tensor] = deque()
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

        Examples:
            >>> policy = InferenceModel.load("./exports/act_policy")
            >>> policy = InferenceModel.load("./exports", backend="onnx")
        """
        return cls(export_dir=export_dir, **kwargs)

    def select_action(self, observation: Observation) -> torch.Tensor:
        """Select action for given observation.

        Matches PyTorch policy API for seamless transition from
        training to production.

        For chunked policies (chunk_size > 1), manages action queue
        automatically and returns one action at a time.

        Args:
            observation: Robot observation (images, states, etc.)

        Returns:
            Action tensor to execute. Shape: (batch_size, action_dim)
                or (action_dim,) for single observation

        Examples:
            >>> obs = env.reset()
            >>> action = policy.select_action(obs)
            >>> next_obs, reward, done = env.step(action)
        """
        # For chunked policies, use action queue
        if self.use_action_queue and len(self._action_queue) > 0:
            return self._action_queue.popleft()

        # Convert observation to model inputs
        inputs = self._prepare_inputs(observation)

        # Run inference
        outputs = self.adapter.predict(inputs)

        # Extract actions from outputs
        action_key = self._get_action_output_key(outputs)
        actions = torch.from_numpy(outputs[action_key])

        # Manage action queue for chunked policies
        if self.use_action_queue and self.chunk_size > 1:
            # actions shape: (batch, chunk_size, action_dim)
            # Queue shape: (chunk_size, batch, action_dim)
            batch_actions = actions.transpose(0, 1)
            self._action_queue.extend(batch_actions)
            return self._action_queue.popleft()

        # For non-chunked policies, return directly
        # Remove temporal dimension if present: (batch, 1, action_dim) -> (batch, action_dim)
        temporal_dim = 3
        if actions.dim() == temporal_dim and actions.shape[1] == 1:
            actions = actions.squeeze(1)

        return actions

    def reset(self) -> None:
        """Reset policy state for new episode.

        Clears action queue and any other internal state.
        Call this at the start of each episode.

        Examples:
            >>> for episode in range(num_episodes):
            ...     policy.reset()
            ...     obs = env.reset()
            ...     done = False
            ...     while not done:
            ...         action = policy.select_action(obs)
            ...         obs, reward, done = env.step(action)
        """
        self._action_queue.clear()

    def _prepare_inputs(self, observation: Observation) -> dict[str, np.ndarray | Observation]:
        """Convert observation to model input format.

        Handles both first-party (state, images) and LeRobot (observation.*, action)
        naming conventions by mapping Observation fields to expected model inputs.
        Returns the original Observation object when the model expects
        a single "Observation" input.

        Args:
            observation: Robot observation

        Returns:
            Dictionary mapping input names to numpy arrays or the input Observation

        Raises:
            ValueError: If required model inputs are missing from observation
        """
        import numpy as np  # noqa: PLC0415

        # Convert observation to numpy arrays using unified .to() API
        obs_numpy = observation.to_numpy()

        # Convert observation to dict format
        obs_dict = obs_numpy.to_dict()

        # Get model input names - prefer metadata over adapter when available
        # Metadata contains semantic names (e.g., "observation_state")
        # Adapter may return internal node names for some backends (e.g., OpenVINO Cast nodes)
        if "input_names" in self.metadata:
            expected_input_names = self.metadata["input_names"]
            adapter_input_names = list(self.adapter.input_names)
        else:
            expected_input_names = list(self.adapter.input_names)
            adapter_input_names = expected_input_names

        expected_inputs = set(expected_input_names)

        if expected_inputs == {"observation"}:
            return {"observation": observation}

        # Build mapping from observation fields to expected input names
        # This handles different naming conventions:
        # - First-party: "state", "images" -> "state", "images"
        # - LeRobot: "state", "images" -> "observation.state", "observation.image"
        field_mapping = self._build_field_mapping(obs_dict, expected_inputs)

        # Build inputs using the mapping
        inputs = {}
        for obs_key, model_key in field_mapping.items():
            value = obs_dict[obs_key]

            # Skip None values
            if value is None:
                continue

            # Handle images (can be list or single tensor)
            if obs_key == "images" and isinstance(value, list):
                # For LeRobot, take first image from list
                # NOTE: Multiple camera support will be added in future
                if len(value) > 0:
                    value = value[0]
                else:
                    continue

            # Add to inputs (already numpy arrays after to_numpy())
            if isinstance(value, np.ndarray):
                inputs[model_key] = value
            else:
                # Handle nested structures if needed
                inputs[model_key] = np.array(value)

        # Validate all expected inputs are present
        missing_inputs = expected_inputs - set(inputs.keys())
        if missing_inputs:
            available_fields = list(obs_dict.keys())
            msg = f"Missing required model inputs: {missing_inputs}. Available observation fields: {available_fields}"
            raise ValueError(msg)

        # If adapter uses different names (e.g., OpenVINO Cast nodes), map by position
        if expected_input_names != adapter_input_names:
            # Reorder inputs to match expected order, then use adapter input names (indices)
            return {adapter_input_names[i]: inputs[expected_input_names[i]] for i in range(len(expected_input_names))}

        return inputs

    @staticmethod
    def _build_field_mapping(obs_dict: dict[str, Any], expected_inputs: set[str]) -> dict[str, str]:
        """Build mapping from observation fields to model input names.

        Supports both first-party and LeRobot naming conventions.

        Args:
            obs_dict: Observation dictionary with keys like "state", "images"
            expected_inputs: Set of expected model input names

        Returns:
            Dictionary mapping observation keys to model input names
        """
        mapping = {}

        # Dummy matching for exact matches
        mapping = {key: key for key in obs_dict if key in expected_inputs}

        if len(mapping) == len(expected_inputs):
            return mapping

        # Common observation fields with their possible model input names
        # Supports both first-party (e.g., "state") and LeRobot (e.g., "observation.state") conventions
        # NOTE: ONNX converts dots to underscores in input names
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

            # Find which model key matches expected inputs
            for model_key in possible_model_keys:
                if model_key in expected_inputs:
                    mapping[obs_key] = model_key
                    break

        return mapping

    @staticmethod
    def _get_action_output_key(outputs: dict[str, np.ndarray]) -> str:
        """Determine which output contains actions.

        Args:
            outputs: Model outputs

        Returns:
            Key for action tensor
        """
        # Try common action key names
        action_keys = ["actions", "action", "output", "pred_actions"]

        for key in action_keys:
            if key in outputs:
                return key

        # Fallback to first output
        return next(iter(outputs))

    def _load_metadata(self) -> dict[str, Any]:
        """Load export metadata from yaml or json file.

        Returns:
            Metadata dictionary
        """
        # Try YAML first (preferred)
        yaml_path = self.export_dir / "metadata.yaml"
        if yaml_path.exists():
            with yaml_path.open() as f:
                return yaml.safe_load(f) or {}

        # Try JSON
        json_path = self.export_dir / "metadata.json"
        if json_path.exists():
            with json_path.open() as f:
                return json.load(f)

        # No metadata file found, return empty dict
        return {}

    def _detect_policy_name(self) -> str:
        """Auto-detect policy name from files or metadata.

        Returns:
            Policy name (e.g., 'act', 'diffusion')

        Raises:
            ValueError: If policy name cannot be determined
        """
        # Try metadata first
        if "policy_class" in self.metadata:
            # Extract policy name from class path
            # e.g., "physicalai.policies.act.ACT" -> "act"
            class_path = self.metadata["policy_class"]
            parts = class_path.lower().split(".")
            min_parts_for_module_extraction = 3
            if len(parts) >= min_parts_for_module_extraction:
                return parts[-2]  # Get policy module name

        # Try to infer from model files
        model_files = list(self.export_dir.glob("*.*"))
        if model_files:
            # Extract name from first model file
            # e.g., "act.onnx" -> "act", "act_policy.xml" -> "act_policy"
            name = model_files[0].stem
            # Remove common suffixes
            for suffix in ["_policy", "_model"]:
                name = name.removesuffix(suffix)
            return name

        msg = f"Cannot determine policy name from {self.export_dir}"
        raise ValueError(msg)

    def _detect_backend(self) -> str:
        """Auto-detect backend from model files.

        Returns:
            Backend name

        Raises:
            ValueError: If backend cannot be determined
        """
        # Map file extensions to backends
        extension_map = {
            ".xml": "openvino",  # OpenVINO IR
            ".onnx": "onnx",  # ONNX
            ".pt2": "torch_export_ir",  # Torch Export IR (PyTorch 2.x)
            ".ptir": "torch_export_ir",  # Torch Export IR (alternative extension)
            ".ckpt": "torch",  # Torch
            ".pt": "torch",  # Torch (alternative extension)
        }

        for ext, backend in extension_map.items():
            if list(self.export_dir.glob(f"*{ext}")):
                return backend

        msg = f"Cannot detect backend from files in {self.export_dir}"
        raise ValueError(msg)

    def _detect_device(self) -> str:
        """Auto-detect best available device.

        Returns:
            Device name
        """
        # For OpenVINO, prefer CPU as default (most compatible)
        if self.backend == ExportBackend.OPENVINO:
            return "CPU"

        # For ONNX/Torch Export IR, check CUDA availability
        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def _get_model_path(self) -> Path:
        """Get path to model file based on backend.

        Returns:
            Path to model file

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        # Map backend to file extension(s)
        extension_map = {
            ExportBackend.OPENVINO: [".xml"],
            ExportBackend.ONNX: [".onnx"],
            ExportBackend.TORCH_EXPORT_IR: [".pt2", ".ptir"],
            ExportBackend.TORCH: [".ckpt", ".pt"],
        }

        extensions = extension_map[self.backend]

        # Try with policy name first
        if self.policy_name:
            for ext in extensions:
                model_path = self.export_dir / f"{self.policy_name}{ext}"
                if model_path.exists():
                    return model_path

        # Try finding any file with any of the extensions
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
