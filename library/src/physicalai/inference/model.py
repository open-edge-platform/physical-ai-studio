# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Production-ready inference model with unified API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from physicalai.export.backends import ExportBackend
from physicalai.inference.adapters import get_adapter
from physicalai.inference.runners import get_runner

if TYPE_CHECKING:
    import numpy as np

    from physicalai.inference.adapters.base import RuntimeAdapter
    from physicalai.inference.runners.base import InferenceRunner


class InferenceModel:
    """Unified inference interface for exported policies.

    Automatically detects backend and provides consistent API across
    all export formats (OpenVINO, ONNX, Torch Export IR).

    The interface matches PyTorch policy API:
    - ``select_action(obs)`` — Get action from observation
    - ``reset()`` — Reset policy state for new episode
    - ``__call__(inputs)`` — Primary inference API (delegates to runner)

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

        >>> # Override the runner to disable action chunking (e.g. for benchmarking):
        >>> from physicalai.inference.runners import SinglePass
        >>> policy = InferenceModel.load("./exports/act_policy", runner=SinglePass())

        >>> # Force action chunking with a custom chunk size:
        >>> from physicalai.inference.runners import ActionChunking, SinglePass
        >>> policy = InferenceModel.load(
        ...     "./exports/act_policy",
        ...     runner=ActionChunking(SinglePass(), chunk_size=20),
        ... )
    """

    def __init__(
        self,
        export_dir: str | Path,
        policy_name: str | None = None,
        backend: str | ExportBackend = "auto",
        device: str = "auto",
        runner: InferenceRunner | None = None,
        **adapter_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize InferenceModel with optional auto-detection.

        Args:
            export_dir: Directory containing exported policy files
            policy_name: Policy name (auto-detected if None)
            backend: Backend to use, or 'auto' to detect from metadata/files
            device: Device for inference ('auto', 'cpu', 'cuda', 'CPU', 'GPU', etc.)
            runner: Execution runner override. If None, auto-selected from metadata.
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

        self.adapter: RuntimeAdapter = get_adapter(self.backend, device=device, **adapter_kwargs)
        model_path = self._get_model_path()
        self.adapter.load(model_path)

        self.runner: InferenceRunner = runner if runner is not None else get_runner(self.metadata)

    @property
    def use_action_queue(self) -> bool:
        """Whether action queuing is enabled (backward compat)."""
        return self.metadata.get("use_action_queue", False)

    @property
    def chunk_size(self) -> int:
        """Action chunk size from metadata (backward compat)."""
        return self.metadata.get("chunk_size", 1)

    @classmethod
    def load(
        cls,
        export_dir: str | Path,
        **kwargs: Any,  # noqa: ANN401
    ) -> InferenceModel:
        """Load inference model with auto-detection.

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

    def __call__(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Primary inference API — prepare inputs and delegate to runner.

        Args:
            observation: Robot observation as a dict mapping input names to numpy arrays.

        Returns:
            Action array to execute.
        """
        inputs = self._prepare_inputs(observation)
        return self.runner.run(self.adapter, inputs)

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Select action for given observation.

        Matches PyTorch policy API for seamless transition from
        training to production. Delegates to ``__call__``.

        Args:
            observation: Robot observation as a dict mapping input names to numpy arrays.

        Returns:
            Action array to execute.

        Examples:
            >>> obs = env.reset()
            >>> action = policy.select_action(obs)
            >>> next_obs, reward, done = env.step(action)
        """
        return self(observation)

    def reset(self) -> None:
        """Reset policy state for new episode.

        Clears runner internal state (e.g. action queues).
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
        self.runner.reset()

    def _prepare_inputs(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Flatten and filter observation dict for the adapter.

        Flattens nested dicts using dot notation (e.g., ``{"obs": {"image": x}}``
        becomes ``{"obs.image": x}``), then filters to only the keys the adapter
        expects.

        Args:
            observation: Observation dict mapping input names to arrays. Values
                may be nested dicts, which are flattened with dot-separated keys.

        Returns:
            Flat dict containing only the adapter's expected inputs. If the
            adapter has no declared input names, returns ``observation`` unchanged.

        Raises:
            KeyError: If an expected adapter input is not found in the
                (flattened) observation.
        """
        expected = self.adapter.input_names

        if expected:
            flat_observation: dict[str, np.ndarray] = {}
            for key, value in observation.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_observation[f"{key}.{sub_key}"] = sub_value
                else:
                    flat_observation[key] = value

            filtered: dict[str, np.ndarray] = {}
            for k in expected:
                if k in flat_observation:
                    filtered[k] = flat_observation[k]
                else:
                    msg = (
                        f"Expected input '{k}' not found in observation.\n"
                        f"Available keys: {list(flat_observation.keys())}"
                    )
                    raise KeyError(msg)

            return filtered
        return observation

    @staticmethod
    def _get_action_output_key(outputs: dict[str, np.ndarray]) -> str:
        """Determine which output contains actions.

        Args:
            outputs: Model outputs

        Returns:
            Key for action tensor
        """
        action_keys = ["actions", "action", "output", "pred_actions"]

        for key in action_keys:
            if key in outputs:
                return key

        return next(iter(outputs))

    def _load_metadata(self) -> dict[str, Any]:
        """Load export metadata from yaml or json file.

        Returns:
            Metadata dict, or empty dict if no metadata file is found.
        """
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
        """Auto-detect policy name from files or metadata.

        Returns:
            Policy name (e.g., 'act', 'diffusion')

        Raises:
            ValueError: If policy name cannot be determined
        """
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
        """Auto-detect backend from model files.

        Returns:
            Backend name

        Raises:
            ValueError: If backend cannot be determined
        """
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
        """Auto-detect best available device using adapter-native detection.

        Returns:
            Device string for the best available device.
        """
        # Create a lightweight adapter instance to query its preferred device
        adapter = get_adapter(self.backend, device="cpu")
        return adapter.default_device()

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
            f"{self.__class__.__name__}("
            f"policy={self.policy_name}, "
            f"backend={self.backend.value}, "
            f"device={self.device}, "
            f"runner={self.runner!r})"
        )
