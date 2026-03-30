# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Production-ready inference model with unified API."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import yaml

from physicalai.export.backends import ExportBackend
from physicalai.inference.adapters import get_adapter
from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.constants import ACTION
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.runners import get_runner

if TYPE_CHECKING:
    import numpy as np

    from physicalai.inference.adapters.base import RuntimeAdapter
    from physicalai.inference.callbacks.base import Callback
    from physicalai.inference.postprocessors.base import Postprocessor
    from physicalai.inference.preprocessors.base import Preprocessor
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
        preprocessors: list[Preprocessor] | None = None,
        postprocessors: list[Postprocessor] | None = None,
        callbacks: list[Callback] | None = None,
        **adapter_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize InferenceModel with optional auto-detection.

        Args:
            export_dir: Directory containing exported policy files
            policy_name: Policy name (auto-detected if None)
            backend: Backend to use, or 'auto' to detect from metadata/files
            device: Device for inference ('auto', 'cpu', 'cuda', 'CPU', 'GPU', etc.)
            runner: Execution runner override. If None, auto-selected from metadata.
            preprocessors: Pipeline stages applied to observations before the
                runner.  If ``None``, loaded from manifest (empty if not
                declared).
            postprocessors: Pipeline stages applied to runner output.  If
                ``None``, loaded from manifest (empty if not declared).
            callbacks: Lifecycle callbacks for instrumentation (timing,
                logging, safety checks, etc.).  Defaults to no callbacks.
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
            backend = self._detect_backend_from_metadata() or self._detect_backend()
        self.backend = ExportBackend(backend) if isinstance(backend, str) else backend

        if device == "auto":
            device = self._detect_device()
        self.device = device

        self.adapter: RuntimeAdapter = get_adapter(self.backend, device=device, **adapter_kwargs)
        model_path = self._get_model_path()
        self.adapter.load(model_path)

        self.runner: InferenceRunner = runner if runner is not None else get_runner(self.metadata)

        self.preprocessors: list[Preprocessor] = (
            preprocessors if preprocessors is not None else self._load_processors("preprocessors")
        )
        self.postprocessors: list[Postprocessor] = (
            postprocessors if postprocessors is not None else self._load_processors("postprocessors")
        )

        self.callbacks: list[Callback] = callbacks if callbacks is not None else []

        for callback in self.callbacks:
            callback.on_load(self)

    @property
    def use_action_queue(self) -> bool:
        """Whether action queuing is enabled (backward compat)."""
        policy = self.metadata.get("policy", {})
        if isinstance(policy, dict) and policy.get("kind") == "action_chunking":
            return True
        return self.metadata.get("use_action_queue", False)

    @property
    def chunk_size(self) -> int:
        """Action chunk size from metadata (backward compat)."""
        runner_spec = self.metadata.get("runner", {})
        if isinstance(runner_spec, dict):
            chunk = runner_spec.get("init_args", {}).get("chunk_size")
            if chunk is not None:
                return int(chunk)
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

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the full inference pipeline and return model outputs.

        Pipeline: callbacks(start) → preprocessors → _prepare_inputs →
        runner → postprocessors → callbacks(end).

        This is the generic inference API — it returns the full output
        dict without assuming any domain-specific keys.

        Args:
            inputs: Input payload as a dict mapping names to numpy arrays.

        Returns:
            Model outputs after runner execution and postprocessing.
        """
        for callback in self.callbacks:
            modified = callback.on_predict_start(inputs)
            if modified is not None:
                inputs = modified

        for preprocessor in self.preprocessors:
            inputs = preprocessor(inputs)

        prepared = self._prepare_inputs(inputs)
        outputs = self.runner.run(self.adapter, prepared)

        for postprocessor in self.postprocessors:
            outputs = postprocessor(outputs)

        for callback in self.callbacks:
            modified = callback.on_predict_end(outputs)
            if modified is not None:
                outputs = modified

        return outputs

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Select action for given observation.

        Domain-specific convenience method for robotics policies.
        Delegates to ``__call__`` and extracts the ``"action"`` key.

        Args:
            observation: Observation dict mapping names to numpy arrays.

        Returns:
            Action array to execute.

        Examples:
            >>> obs = env.reset()
            >>> action = policy.select_action(obs)
            >>> next_obs, reward, done = env.step(action)
        """
        outputs = self(observation)
        return outputs[ACTION]

    def reset(self) -> None:
        """Reset policy state for new episode.

        Clears runner internal state (e.g. action queues) and
        notifies all callbacks.
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
        for callback in self.callbacks:
            callback.on_reset()

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns:
            The model instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager."""

    def _prepare_inputs(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Flatten and filter input dict for the adapter.

        Flattens nested dicts using dot notation (e.g., ``{"obs": {"image": x}}``
        becomes ``{"obs.image": x}``), then filters to only the keys the adapter
        expects.

        Args:
            inputs: Input dict mapping names to arrays. Values
                may be nested dicts, which are flattened with dot-separated keys.

        Returns:
            Flat dict containing only the adapter's expected inputs. If the
            adapter has no declared input names, returns ``inputs`` unchanged.

        Raises:
            KeyError: If an expected adapter input is not found in the
                (flattened) inputs.
        """
        expected = self.adapter.input_names

        if expected:
            flat_inputs: dict[str, np.ndarray] = {}
            for key, value in inputs.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_inputs[f"{key}.{sub_key}"] = sub_value
                else:
                    flat_inputs[key] = value

            filtered: dict[str, np.ndarray] = {}
            for k in expected:
                if k in flat_inputs:
                    filtered[k] = flat_inputs[k]
                else:
                    msg = f"Expected input '{k}' not found in inputs.\nAvailable keys: {list(flat_inputs.keys())}"
                    raise KeyError(msg)

            return filtered
        return inputs

    def _load_metadata(self) -> dict[str, Any]:
        """Load export metadata from manifest.json, metadata.yaml, or metadata.json.

        Tries ``manifest.json`` first (new format), then falls back to
        ``metadata.yaml`` and ``metadata.json`` for backward compatibility.

        .. deprecated::
            Loading from ``metadata.yaml`` or ``metadata.json`` is
            deprecated.  Re-export models to generate ``manifest.json``.

        Returns:
            Metadata dict, or empty dict if no metadata file is found.
        """
        manifest_path = self.export_dir / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as f:
                return json.load(f)

        yaml_path = self.export_dir / "metadata.yaml"
        json_path = self.export_dir / "metadata.json"
        legacy_path = yaml_path if yaml_path.exists() else json_path if json_path.exists() else None

        if legacy_path is not None:
            warnings.warn(
                f"Loading from '{legacy_path.name}' is deprecated. "
                "Re-export your model to generate 'manifest.json'. "
                "Legacy metadata support will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            if legacy_path.suffix == ".yaml":
                with legacy_path.open(encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            with legacy_path.open(encoding="utf-8") as f:
                return json.load(f)

        return {}

    def _load_processors(self, key: str) -> list[Any]:
        """Instantiate preprocessors or postprocessors from manifest metadata.

        Args:
            key: ``"preprocessors"`` or ``"postprocessors"``.

        Returns:
            List of instantiated processor objects, or empty list if the
            manifest does not declare any for *key*.
        """
        specs = self.metadata.get(key, [])
        if not specs:
            return []
        return [instantiate_component(ComponentSpec.model_validate(s)) for s in specs]

    def _detect_policy_name(self) -> str:
        """Auto-detect policy name from files or metadata.

        Checks manifest ``policy.name`` first, then falls back to
        legacy ``policy_class`` extraction, then file-name heuristics.

        Returns:
            Policy name (e.g., 'act', 'diffusion')

        Raises:
            ValueError: If policy name cannot be determined
        """
        policy = self.metadata.get("policy", {})
        if isinstance(policy, dict) and policy.get("name"):
            return policy["name"]

        class_path = ""
        if isinstance(policy, dict) and policy.get("class_path"):
            class_path = policy["class_path"]
        elif "policy_class" in self.metadata:
            class_path = self.metadata["policy_class"]

        if class_path:
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

    def _detect_backend_from_metadata(self) -> str | None:
        """Extract backend from manifest artifacts or legacy metadata.

        Returns:
            Backend string, or ``None`` if not found.
        """
        artifacts = self.metadata.get("artifacts", {})
        if isinstance(artifacts, dict) and artifacts:
            return next(iter(artifacts))

        backend = self.metadata.get("backend")
        if backend:
            return str(backend)

        return None

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
            ".pte": "executorch",
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
            ExportBackend.EXECUTORCH: [".pte"],
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
