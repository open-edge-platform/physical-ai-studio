# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessing and postprocessing for the XR1 policy.

The preprocessor turns an :class:`~physicalai.data.observation.Observation` batch
into the tensors the backbone and action expert consume:

* state and action are normalized with the dataset statistics, then zero-padded to
  ``max_state_dim`` / ``max_action_dim``, with a mask recording which entries are
  real;
* camera images are letterboxed to ``image_resolution`` and laid out in the order
  given by ``camera_views``, each announced by name in the prompt so the backbone
  can tell the views apart;
* the instruction is rendered through the Qwen3-VL chat template.

The postprocessor denormalizes predicted actions and trims the padding away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from physicalai.data import Feature, FeatureType
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType
from physicalai.policies.xr1.io import build_action_mask, pad_vector, resize_with_pad

if TYPE_CHECKING:
    from physicalai.policies.xr1.config import XR1Config

_NORMALIZATION_TYPES = {
    "MEAN_STD": NormalizationType.MEAN_STD,
    "QUANTILES": NormalizationType.MIN_MAX,
}


def normalization_map(mode: str) -> dict[FeatureType, NormalizationType]:
    """Map a configured normalization mode onto per-feature-type strategies.

    Args:
        mode: Either ``"MEAN_STD"`` or ``"QUANTILES"``.

    Returns:
        Mapping used to build a :class:`FeatureNormalizeTransform`.

    Raises:
        ValueError: If the mode is unknown.
    """
    if mode not in _NORMALIZATION_TYPES:
        msg = f"Unsupported normalization_mode: {mode}"
        raise ValueError(msg)
    norm_type = _NORMALIZATION_TYPES[mode]
    return {FeatureType.STATE: norm_type, FeatureType.ACTION: norm_type}


def split_features(features: dict[str, Feature] | None) -> tuple[dict[str, Feature], dict[str, Feature]]:
    """Split a feature dict into state-like and action features.

    Args:
        features: Feature schema, or ``None``.

    Returns:
        ``(state_features, action_features)``; both empty when ``features`` is
        ``None``.
    """
    if not features:
        return {}, {}
    state = {name: feature for name, feature in features.items() if feature.ftype == FeatureType.STATE}
    action = {name: feature for name, feature in features.items() if feature.ftype == FeatureType.ACTION}
    return state, action


class XR1Preprocessor(nn.Module):
    """Turn an observation batch into backbone and action-expert inputs."""

    def __init__(
        self,
        config: XR1Config,
        features: dict[str, Feature] | None = None,
        processor: Any = None,  # noqa: ANN401 - a transformers processor, typed loosely to keep it injectable
    ) -> None:
        """Initialize the preprocessor.

        Args:
            config: Model configuration.
            features: Feature schema with normalization statistics. When ``None``,
                normalization is skipped, which is only useful in tests.
            processor: Pre-built Qwen3-VL processor. When ``None`` it is loaded
                lazily from ``config.vlm_model_id`` on first use, so constructing a
                preprocessor never touches the network.
        """
        super().__init__()
        self.config = config
        self._processor = processor
        state_features, action_features = split_features(features)
        norm_map = normalization_map(config.normalization_mode)

        self._normalizer: FeatureNormalizeTransform | None = None
        if state_features or action_features:
            self._normalizer = FeatureNormalizeTransform({**state_features, **action_features}, norm_map)

    @property
    def processor(self) -> Any:  # noqa: ANN401 - see __init__
        """Qwen3-VL processor, loaded lazily on first use.

        Returns:
            The processor for ``config.vlm_model_id``.
        """
        if self._processor is None:
            from transformers import AutoProcessor  # noqa: PLC0415  # heavy import, deferred

            self._processor = AutoProcessor.from_pretrained(self.config.vlm_model_id)
        return self._processor

    def build_prompt(self, instruction: str, num_images: int) -> str:
        """Render the chat prompt for one sample.

        Each image is preceded by its view name so the backbone can distinguish a
        wrist camera from an overhead one.

        Args:
            instruction: Natural-language task description.
            num_images: Number of images present for this sample.

        Returns:
            The rendered prompt string.
        """
        content: list[dict[str, str]] = []
        for index in range(num_images):
            view = self.config.camera_views[index] if index < len(self.config.camera_views) else f"view{index}"
            content.extend((
                {"type": "text", "text": f"{view.replace('_', ' ')} view:"},
                {"type": "image"},
            ))
        content.append({"type": "text", "text": instruction})

        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _ordered_images(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        """Collect camera tensors in the configured view order.

        Args:
            batch: Observation dict.

        Returns:
            One ``(batch, channels, height, width)`` tensor per view.
        """
        images = batch.get(IMAGES)
        if images is None:
            return []
        if isinstance(images, torch.Tensor):
            return [self._prepare_images(images)]

        ordered: list[torch.Tensor] = [
            self._prepare_images(images[view]) for view in self.config.camera_views if view in images
        ]
        if not ordered:
            # Fall back to whatever the dataset provides, in its own order, so a
            # dataset whose cameras are named differently still trains.
            ordered = [self._prepare_images(value) for value in images.values()]
        return ordered

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        """Letterbox a camera tensor to the configured resolution.

        Args:
            images: Tensor of shape ``(batch, channels, height, width)``.

        Returns:
            Resized tensor in ``float32`` in ``[0, 1]``.
        """
        images = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
        height, width = self.config.image_resolution
        return resize_with_pad(images, height, width)

    @staticmethod
    def _instructions(batch: dict[str, Any], batch_size: int) -> list[str]:
        """Extract one instruction string per sample.

        Args:
            batch: Observation dict.
            batch_size: Number of samples.

        Returns:
            Instruction strings, defaulting to a generic one when absent.
        """
        task = batch.get(TASK)
        if task is None:
            return [""] * batch_size
        if isinstance(task, str):
            return [task] * batch_size
        if isinstance(task, dict):
            task = next(iter(task.values()))
        if isinstance(task, (list, tuple)):
            return [str(item) for item in task]
        return [str(task)] * batch_size

    def _prepare_state(self, batch: dict[str, Any]) -> torch.Tensor:
        """Normalize, pad and shape the robot state.

        Args:
            batch: Observation dict.

        Returns:
            State tensor of shape ``(batch, state_len, max_state_dim)``.

        Raises:
            KeyError: If the batch carries no state.
        """
        state = batch.get(STATE)
        if state is None:
            msg = "XR1 requires a state feature in the observation batch"
            raise KeyError(msg)
        if isinstance(state, dict):
            state = torch.cat([value.flatten(start_dim=1) for value in state.values()], dim=-1)

        state = state.flatten(start_dim=1)
        state = pad_vector(state, self.config.max_state_dim)
        return state[:, None, :].expand(-1, self.config.state_len, -1).contiguous()

    def _prepare_action(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Normalize, pad and mask the target action chunk.

        Args:
            batch: Observation dict.

        Returns:
            ``(action, action_mask)``, or ``None`` when the batch has no action.
        """
        action = batch.get(ACTION)
        if action is None:
            return None
        if isinstance(action, dict):
            action = next(iter(action.values()))
        if action.ndim == 2:  # noqa: PLR2004 - (batch, action_dim) for a single step
            action = action[:, None, :]

        valid_dim = action.shape[-1]
        padded = pad_vector(action, self.config.max_action_dim)
        mask = build_action_mask(padded, valid_dim, batch.get("action_is_pad_inverse"))
        return padded, mask

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess an observation batch.

        Args:
            batch: Observation dict, typically ``Observation.to_dict()``.

        Returns:
            Dict with ``input_ids``, ``attention_mask``, optional ``pixel_values``
            and ``image_grid_thw``, ``state`` and, when supervision is present,
            ``action`` and ``action_mask``.
        """
        batch = dict(batch)
        if self._normalizer is not None:
            batch = self._normalizer(batch)

        state = self._prepare_state(batch)
        batch_size = state.shape[0]
        view_tensors = self._ordered_images(batch)

        prompts = [
            self.build_prompt(instruction, len(view_tensors)) for instruction in self._instructions(batch, batch_size)
        ]
        per_sample_images = (
            [[views[index] for views in view_tensors] for index in range(batch_size)] if view_tensors else None
        )

        encoded = self.processor(
            text=prompts,
            images=per_sample_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.tokenizer_max_length,
        )

        # The processor always returns CPU tensors, while the model lives on the
        # accelerator; the state tensor carries the device the batch arrived on.
        device = state.device
        processed: dict[str, Any] = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            STATE: state,
        }
        for key in ("pixel_values", "image_grid_thw"):
            if key in encoded:
                processed[key] = encoded[key].to(device)

        action = self._prepare_action(batch)
        if action is not None:
            processed[ACTION], processed["action_mask"] = action
        if "action_prefix" in batch:
            processed["action_prefix"] = pad_vector(batch["action_prefix"], self.config.max_action_dim)
        if "prefix_length" in batch:
            processed["prefix_length"] = batch["prefix_length"]

        return processed


class XR1Postprocessor(nn.Module):
    """Denormalize predicted actions and drop the padded dimensions."""

    def __init__(self, config: XR1Config, features: dict[str, Feature] | None = None) -> None:
        """Initialize the postprocessor.

        Args:
            config: Model configuration, used for the normalization mode.
            features: Action feature schema with normalization statistics.
        """
        super().__init__()
        self.config = config
        _, action_features = split_features(features)
        self._action_dim = None
        self._denormalizer: FeatureNormalizeTransform | None = None

        if action_features:
            norm_map = normalization_map(config.normalization_mode)
            self._denormalizer = FeatureNormalizeTransform(action_features, norm_map, inverse=True)
            first = next(iter(action_features.values()))
            self._action_dim = int(first.shape[-1]) if first.shape else None

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Postprocess predicted actions.

        Args:
            batch: Dict carrying an ``action`` tensor of shape
                ``(batch, horizon, max_action_dim)``.

        Returns:
            The same dict with actions trimmed to the dataset width and
            denormalized.
        """
        batch = dict(batch)
        action = batch.get(ACTION)
        if action is None:
            return batch

        if self._action_dim is not None:
            action = action[..., : self._action_dim]
        batch[ACTION] = action

        if self._denormalizer is not None:
            batch = self._denormalizer(batch)
        return batch


def make_xr1_preprocessors(
    config: XR1Config,
    features: dict[str, Feature] | None = None,
    processor: Any = None,  # noqa: ANN401 - see XR1Preprocessor
) -> tuple[XR1Preprocessor, XR1Postprocessor]:
    """Build a matched preprocessor and postprocessor pair.

    Args:
        config: Model configuration.
        features: Feature schema with normalization statistics.
        processor: Optional pre-built Qwen3-VL processor.

    Returns:
        ``(preprocessor, postprocessor)``.
    """
    return XR1Preprocessor(config, features, processor), XR1Postprocessor(config, features)
