# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Composable MolmoAct2 preprocessing steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms.functional as tv_functional

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK

from .utils import build_discrete_state_string, build_robot_text, normalize_text

_BCHW_DIMENSIONS = 4
_RGB_CHANNELS = 3
_UNBATCHED_ACTION_DIMENSIONS = 2
_BATCHED_ACTION_DIMENSIONS = 3


@dataclass
class PreprocessBatchBundle:
    """Carry extracted preprocessing values between components.

    Steps:
        1. Store the normalized state batch.
        2. Preserve one normalized task string per example.
        3. Preserve ordered camera images per example.
    """

    state: torch.Tensor
    tasks: list[str]
    images_by_example: list[list[torch.Tensor]]


@dataclass
class PromptPack:
    """Carry encoded prompts and their flattened image sequence.

    Steps:
        1. Store one encoded prompt per batch example.
        2. Preserve images in prompt traversal order.
    """

    prompt_texts: list[str]
    flat_images: list[torch.Tensor]


class StateTaskImageExtractor:
    """Extract state, task, and images from a flattened input batch.

    Steps:
        1. Resolve and shape the state tensor.
        2. Normalize and broadcast task text across the batch.
        3. Resolve explicit, flattened, nested, or single-camera images.
        4. Group ordered BCHW images by batch example.
    """

    def __init__(self, *, image_keys: list[str]) -> None:
        """Store the preferred camera-key order."""
        self.image_keys = image_keys

    def extract(self, batch: dict[str, Any]) -> PreprocessBatchBundle:
        """Extract normalized state, tasks, and per-example images.

        Returns:
            Extracted state, tasks, and ordered images.

        Raises:
            ValueError: If state, task, or image inputs are invalid.
        """
        state = batch.get(STATE)
        if state is None:
            msg = "MolmoAct2 requires a state tensor in the input batch."
            raise ValueError(msg)
        state = torch.as_tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch_size = int(state.shape[0])
        return PreprocessBatchBundle(
            state=state.clamp(-1.0, 1.0),
            tasks=self._tasks(batch, batch_size),
            images_by_example=self._images(batch, batch_size),
        )

    @staticmethod
    def _tasks(batch: dict[str, Any], batch_size: int) -> list[str]:
        source = batch.get(TASK)
        if isinstance(source, str):
            tasks = [source] * batch_size
        elif torch.is_tensor(source):
            tasks = [str(value) for value in source.detach().cpu().reshape(-1).tolist()]  # type: ignore[union-attr]
        elif isinstance(source, (list, tuple)):
            tasks = [str(value) for value in source]
        else:
            tasks = [str(source or "")]
        if len(tasks) == 1 and batch_size > 1:
            tasks *= batch_size
        if len(tasks) != batch_size:
            msg = f"Expected {batch_size} task strings, got {len(tasks)}."
            raise ValueError(msg)
        return [normalize_text(task) for task in tasks]

    def _image_keys(self, batch: dict[str, Any]) -> list[str]:
        images = batch.get(IMAGES)
        if isinstance(images, dict):
            configured = [f"{IMAGES}.{name}" for name in self.image_keys if name in images]
            fallback = sorted(f"{IMAGES}.{name}" for name in images if "is_pad" not in str(name))
            return configured or fallback
        if images is not None:
            return [IMAGES]

        configured = [f"{IMAGES}.{name}" for name in self.image_keys]
        available = [key for key in configured if batch.get(key) is not None]
        if available:
            return available
        metadata_keys = batch.get(f"_{IMAGES}_keys")
        if isinstance(metadata_keys, list):
            flattened = [key for key in metadata_keys if isinstance(key, str) and batch.get(key) is not None]
            if flattened:
                return flattened
        flattened = sorted(
            str(key)
            for key, value in batch.items()
            if str(key).startswith(f"{IMAGES}.") and "is_pad" not in str(key) and value is not None
        )
        if flattened:
            return flattened
        msg = "MolmoAct2 requires image tensors in BCHW format."
        raise ValueError(msg)

    @staticmethod
    def _image(batch: dict[str, Any], key: str) -> torch.Tensor:
        value = batch.get(key)
        if value is None and key.startswith(f"{IMAGES}.") and isinstance(batch.get(IMAGES), dict):
            value = batch[IMAGES].get(key.removeprefix(f"{IMAGES}."))
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        if tensor.ndim != _BCHW_DIMENSIONS or int(tensor.shape[1]) != _RGB_CHANNELS:  # type: ignore[union-attr]
            msg = f"Expected BCHW image tensor at {key}, got shape {tuple(tensor.shape)}"  # type: ignore[union-attr]
            raise ValueError(msg)
        return tensor

    def _images(self, batch: dict[str, Any], batch_size: int) -> list[list[torch.Tensor]]:
        output: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        for key in self._image_keys(batch):
            images = self._image(batch, key)
            if int(images.shape[0]) != batch_size:
                msg = f"Image batch size mismatch at {key}: expected {batch_size}, got {int(images.shape[0])}"
                raise ValueError(msg)
            for index in range(batch_size):
                output[index].append(images[index])
        return output


class RobotPromptEncoder:
    """Build MolmoAct2 prompt text from extracted values.

    Steps:
        1. Discretize each normalized state vector into state tokens.
        2. Combine task, setup, control, state, and image placeholders.
        3. Preserve flattened images in prompt order.
    """

    def __init__(
        self,
        *,
        num_state_tokens: int,
        setup_type: str,
        control_mode: str,
        add_setup_tokens: bool,
        add_control_tokens: bool,
    ) -> None:
        """Store prompt formatting and state-token settings."""
        self.num_state_tokens = num_state_tokens
        self.setup_type = setup_type
        self.control_mode = control_mode
        self.add_setup_tokens = add_setup_tokens
        self.add_control_tokens = add_control_tokens

    def encode(self, bundle: PreprocessBatchBundle) -> PromptPack:
        """Encode one prompt per batch element.

        Returns:
            Encoded prompt strings and images in prompt order.
        """
        prompts = [
            build_robot_text(
                task=bundle.tasks[index],
                discrete_state_string=build_discrete_state_string(bundle.state[index], self.num_state_tokens),
                setup_type=self.setup_type,
                control_mode=self.control_mode,
                add_setup_tokens=self.add_setup_tokens,
                add_control_tokens=self.add_control_tokens,
                num_images=len(bundle.images_by_example[index]),
            )
            for index in range(int(bundle.state.shape[0]))
        ]
        return PromptPack(prompts, [image for images in bundle.images_by_example for image in images])


class ImageResizeNormalizer(torch.nn.Module):
    """Resize images and normalize them to [0, 1].

    Steps:
        1. Convert float images onto the uint8 pixel grid when needed.
        2. Resize each image to the configured height and width.
        3. Convert pixels to float32 in the [0, 1] range.
    """

    def __init__(self, *, image_size: tuple[int, int]) -> None:
        """Store the target image height and width."""
        super().__init__()
        self.image_size = image_size

    def forward(self, images: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]:
        """Resize every image while preserving nested layout.

        Returns:
            Images resized and converted to float32 in the [0, 1] range.
        """
        return [[self._resize(image) for image in example] for example in images]

    def _resize(self, image: torch.Tensor) -> torch.Tensor:
        pixels = (
            image
            if image.dtype == torch.uint8
            else (image.float() * (255.0 if float(image.max()) <= 1.0 else 1.0)).clamp(0, 255).byte()
        )
        return tv_functional.resize(pixels, list(self.image_size), antialias=False).float() / 255.0


class ImagePacker(torch.nn.Module):
    """Pack per-example images into [N, B, C, H, W].

    Steps:
        1. Validate a consistent camera count across examples.
        2. Stack each camera slot across the batch.
        3. Stack camera slots and create their validity mask.
    """

    def __init__(self, *, image_size: tuple[int, int]) -> None:
        """Store the image size used for empty packed batches."""
        super().__init__()
        self.image_size = image_size

    def forward(self, images: list[list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack images with a consistent count per batch element.

        Returns:
            Packed images and their validity mask.

        Raises:
            ValueError: If examples contain different image counts.
        """
        if not images:
            height, width = self.image_size
            return torch.empty((0, 0, 3, height, width)), torch.empty((0, 0), dtype=torch.bool)
        count = len(images[0])
        if any(len(example) != count for example in images):
            msg = "MolmoAct2 requires a consistent number of images per batch element."
            raise ValueError(msg)
        packed = torch.stack([torch.stack([example[index] for example in images]) for index in range(count)])
        return packed, torch.ones((count, len(images)), dtype=torch.bool, device=packed.device)


class ActionExtractor:
    """Extract an optional action tensor.

    Steps:
        1. Resolve canonical or flattened action input.
        2. Convert a present action to float32.
    """

    @staticmethod
    def extract(batch: dict[str, Any]) -> torch.Tensor | None:
        """Return action as float32 when present.

        Returns:
            The optional float32 action tensor.
        """
        action = batch.get(ACTION)
        return None if action is None else torch.as_tensor(action, dtype=torch.float32)


class ActionPadder(torch.nn.Module):
    """Pad actions to the model action dimension.

    Steps:
        1. Promote actions to the [B, T, D] layout.
        2. Validate and clamp action values.
        3. Right-pad actions to the fixed model dimension.
        4. Build horizon and action-dimension padding masks.
    """

    def __init__(self, *, max_action_dim: int) -> None:
        """Store the fixed model action dimension."""
        super().__init__()
        self.max_action_dim = max_action_dim

    def forward(
        self,
        action: torch.Tensor,
        horizon_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad action values and construct their padding masks.

        Returns:
            Padded action, horizon mask, and action-dimension mask.

        Raises:
            ValueError: If action dimensions or the horizon mask are invalid.
        """
        if action.ndim == _UNBATCHED_ACTION_DIMENSIONS:
            action = action.unsqueeze(1)
        if action.ndim != _BATCHED_ACTION_DIMENSIONS:
            msg = f"MolmoAct2 expected action shape [B, T, D], got {tuple(action.shape)}."
            raise ValueError(msg)
        if int(action.shape[-1]) > self.max_action_dim:
            msg = f"Action dim {int(action.shape[-1])} exceeds max_action_dim={self.max_action_dim}."
            raise ValueError(msg)
        padded = torch.zeros((*action.shape[:-1], self.max_action_dim), device=action.device, dtype=torch.float32)
        padded[..., : action.shape[-1]] = action.float().clamp(-1.0, 1.0)
        horizon_mask = (
            torch.zeros(action.shape[:2], device=action.device, dtype=torch.bool)
            if horizon_mask is None
            else horizon_mask.to(action.device, torch.bool)
        )
        if tuple(horizon_mask.shape) != tuple(action.shape[:2]):
            msg = "action_horizon_is_pad must match action horizon shape."
            raise ValueError(msg)
        dim_mask = torch.ones((action.shape[0], self.max_action_dim), device=action.device, dtype=torch.bool)
        dim_mask[:, : action.shape[-1]] = False
        return padded, horizon_mask, dim_mask
