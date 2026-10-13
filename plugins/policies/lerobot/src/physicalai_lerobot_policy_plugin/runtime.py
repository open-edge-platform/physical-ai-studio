# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime adapter for native LeRobot policy folders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.registry import adapter_registry
from physicalai.inference.callbacks.base import Callback

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel

_IMAGE_RANK_2 = 2
_IMAGE_RANK_3 = 3
_IMAGE_RANK_4 = 4
_CHANNEL_COUNTS = {1, 3, 4}


@adapter_registry.register("lerobot", extensions=(".lerobot",))
class LeRobotRuntimeAdapter(RuntimeAdapter):
    """Load and run policies from LeRobot's native ``pretrained_model`` layout."""

    def __init__(self, device: str = "cpu") -> None:
        """Initialize the adapter for LeRobot inference.

        Args:
            device: Torch device string selected by Runtime or the caller.
        """
        super().__init__(device=device)
        self._policy: object | None = None
        self._preprocessor: object | None = None
        self._postprocessor: object | None = None

    def default_device(self) -> str:  # ruff: ignore[no-self-use]
        """Select the native LeRobot default device.

        Returns:
            Device string chosen by LeRobot's device helper.
        """
        from lerobot.utils.device_utils import auto_select_torch_device  # ruff: ignore[import-outside-top-level]

        return str(auto_select_torch_device())

    def load(self, model_path: Path | str) -> None:
        """Load a native policy and its saved LeRobot processors.

        Args:
            model_path: Runtime's LeRobot artifact directory.

        Raises:
            FileNotFoundError: If the artifact or native config file is missing.
        """
        model_path = Path(model_path).expanduser()
        if not model_path.is_dir():
            msg = f"LeRobot model directory not found: {model_path}"
            raise FileNotFoundError(msg)
        native_dir = _native_model_directory(model_path)

        from lerobot.configs.policies import PreTrainedConfig  # ruff: ignore[import-outside-top-level]
        from lerobot.policies import make_pre_post_processors  # ruff: ignore[import-outside-top-level]
        from lerobot.policies.factory import get_policy_class  # ruff: ignore[import-outside-top-level]
        from lerobot.utils.import_utils import register_third_party_plugins  # ruff: ignore[import-outside-top-level]

        register_third_party_plugins()
        config = PreTrainedConfig.from_pretrained(
            native_dir,
            cli_overrides=[f"--device={self.device}"],
        )
        policy_class = get_policy_class(config.type)
        policy = policy_class.from_pretrained(native_dir, config=config)
        policy.to(self.device).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(native_dir),
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor

    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run LeRobot preprocessing, stateful action selection, and postprocessing.

        Args:
            inputs: Runtime observation mapping with state, image, and task values.

        Returns:
            Runtime action output as a NumPy array.

        Raises:
            RuntimeError: If the model has not been loaded.
            TypeError: If LeRobot's action selector does not return a tensor.
        """
        if self._policy is None or self._preprocessor is None or self._postprocessor is None:
            msg = "LeRobot model not loaded. Call load() first."
            raise RuntimeError(msg)

        batch = _to_lerobot_batch(inputs, self.device)
        preprocess = self._preprocessor
        select_action = getattr(self._policy, "select_action", None)
        postprocess = self._postprocessor
        if not callable(preprocess) or not callable(select_action) or not callable(postprocess):
            msg = "LeRobot policy and processor pipelines must be callable."
            raise TypeError(msg)
        with torch.inference_mode():
            processed = preprocess(batch)
            action = select_action(processed)
            action = postprocess(action)
        if not isinstance(action, torch.Tensor):
            msg = f"LeRobot select_action() must return a tensor, got {type(action).__name__}."
            raise TypeError(msg)
        action = action.detach().to("cpu")
        if action.dtype == torch.bfloat16:
            action = action.to(torch.float32)
        return {"action": action.numpy()}

    def reset(self) -> None:
        """Reset policy queues/history and LeRobot processor state."""
        for component in (self._policy, self._preprocessor, self._postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    @property
    def input_names(self) -> list[str]:
        """Input names are empty so Runtime passes the structured observation unchanged.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._policy is None:
            msg = "LeRobot model not loaded. Call load() first."
            raise RuntimeError(msg)
        return []

    @property
    def output_names(self) -> list[str]:
        """Action output name after loading.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._policy is None:
            msg = "LeRobot model not loaded. Call load() first."
            raise RuntimeError(msg)
        return ["action"]


def _native_model_directory(artifact_path: Path) -> Path:
    """Resolve the packaged native folder while rejecting escaping paths.

    Args:
        artifact_path: Runtime artifact returned by its suffix lookup.

    Returns:
        Resolved LeRobot native model directory.

    Raises:
        FileNotFoundError: If the native model config is missing.
        ValueError: If the model path escapes the artifact directory.
    """
    artifact_parent = artifact_path.parent.resolve(strict=True)
    artifact_root = artifact_path.resolve(strict=True)
    if not artifact_root.is_relative_to(artifact_parent):
        msg = "LeRobot artifact directory escapes its Runtime export directory."
        raise ValueError(msg)
    nested = artifact_path / "pretrained_model"
    native_dir = nested.resolve(strict=True) if nested.is_dir() else artifact_root
    if not native_dir.is_relative_to(artifact_root):
        msg = "LeRobot pretrained_model directory escapes its Runtime artifact folder."
        raise ValueError(msg)
    if not native_dir.is_dir() or not (native_dir / "config.json").is_file():
        msg = f"LeRobot config.json not found in {native_dir}."
        raise FileNotFoundError(msg)
    if any(path.is_symlink() for path in native_dir.rglob("*")):
        msg = "LeRobot model folders containing symbolic links are not supported."
        raise ValueError(msg)
    return native_dir


def _to_lerobot_batch(inputs: Mapping[str, object], device: str) -> dict[str, object]:
    """Convert Runtime observations to LeRobot's native observation keys.

    Args:
        inputs: Structured Runtime observation mapping.
        device: Target device for tensor inputs.

    Returns:
        LeRobot batch mapping with batched tensors and native key names.

    Raises:
        TypeError: If a camera name is not a string.
    """
    batch: dict[str, object] = {}
    for key, value in inputs.items():
        if key == "images" and isinstance(value, Mapping):
            for camera_name, image in value.items():
                if not isinstance(camera_name, str):
                    msg = "LeRobot camera names must be strings."
                    raise TypeError(msg)
                batch[f"observation.images.{camera_name}"] = _to_device_tensor(image, device, image=True)
        elif key.startswith("images."):
            camera_name = key.removeprefix("images.")
            batch[f"observation.images.{camera_name}"] = _to_device_tensor(value, device, image=True)
        elif key == "state":
            batch["observation.state"] = _to_device_tensor(value, device)
        elif key == "image":
            batch["observation.image"] = _to_device_tensor(value, device, image=True)
        elif key == "task":
            batch[key] = _task_batch(value)
        elif key.startswith("observation."):
            image = key.startswith("observation.image")
            batch[key] = _to_device_tensor(value, device, image=image)
    return batch


def _to_device_tensor(value: object, device: str, *, image: bool = False) -> torch.Tensor:
    """Convert an observation value to a batched tensor on its target device.

    Args:
        value: Native NumPy or PyTorch observation value.
        device: Device string passed to PyTorch.
        image: Whether to normalize image channel layout before LeRobot preprocessing.

    Returns:
        Tensor on *device*, before LeRobot's saved batch processor runs.
    """
    tensor = value.to(device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
    if image:
        tensor = _image_layout(tensor)
    return tensor


def _image_layout(image: torch.Tensor) -> torch.Tensor:
    """Normalize image channel order before LeRobot's native batch processor.

    Args:
        image: Image tensor in channel-first or channel-last layout.

    Returns:
        Image tensor in channel-first layout without an added batch dimension.
    """
    if image.ndim == _IMAGE_RANK_2:
        image = image.unsqueeze(0)
    elif image.ndim == _IMAGE_RANK_3:
        if image.shape[0] not in _CHANNEL_COUNTS and image.shape[-1] in _CHANNEL_COUNTS:
            image = image.permute(2, 0, 1)
    elif image.ndim == _IMAGE_RANK_4 and image.shape[1] not in _CHANNEL_COUNTS and image.shape[-1] in _CHANNEL_COUNTS:
        image = image.permute(0, 3, 1, 2)
    return image


def _task_batch(value: object) -> object:
    """Normalize a task string to LeRobot's batch-of-text representation.

    Args:
        value: Runtime task value.

    Returns:
        A task sequence or the unchanged non-text value.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return value


class LeRobotResetCallback(Callback):
    """Forward Runtime episode resets to LeRobot's stateful model and processors."""

    def __init__(self) -> None:
        """Create a callback that binds to the loaded LeRobot adapter."""
        self._adapter: LeRobotRuntimeAdapter | None = None

    def on_load(self, model: InferenceModel) -> None:
        """Bind the callback to its loaded LeRobot adapter.

        Args:
            model: Runtime model after its adapter has loaded.

        Raises:
            TypeError: If the manifest callback is attached to another backend.
        """
        if model.backend != "lerobot" or not isinstance(model.adapter, LeRobotRuntimeAdapter):
            msg = "LeRobot reset callback requires the lerobot Runtime adapter."
            raise TypeError(msg)
        self._adapter = model.adapter

    def on_reset(self) -> None:
        """Reset LeRobot state when Runtime starts a new episode."""
        if self._adapter is not None:
            self._adapter.reset()
