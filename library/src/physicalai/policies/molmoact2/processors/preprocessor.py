# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 preprocessing orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from physicalai.data.constants import EXTRA, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK

from .inputs import MolmoAct2InputLayout, build_model_inputs
from .joint_transform import JointFrameTransform
from .preprocess_steps import (
    ActionExtractor,
    ActionPadder,
    ImagePacker,
    ImageResizeNormalizer,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)

if TYPE_CHECKING:
    from transformers import Qwen2Tokenizer

    from .image import MolmoAct2ImageProcessor
    from .normalization import MolmoAct2NormalizeTransform
    from .tokenizers import MolmoAct2Tokenizers


class MolmoAct2Preprocessor(torch.nn.Module):
    """Convert observation batches into model-ready MolmoAct2 tensors.

    Steps:
        1. Validate required state, task, and image inputs.
        2. Optionally map SO-101 joints into the checkpoint frame.
        3. Normalize configured state and action features.
        4. Extract state, task text, and ordered camera images.
        5. Encode and tokenize prompts from task and discrete state values.
        6. Resize, normalize, and pack camera images.
        7. Build the text and vision model inputs.
        8. Pad optional training actions and attach their masks.
    """

    def __init__(
        self,
        *,
        normalizer: MolmoAct2NormalizeTransform,
        extractor: StateTaskImageExtractor,
        prompt_encoder: RobotPromptEncoder,
        image_resize: ImageResizeNormalizer,
        image_packer: ImagePacker,
        image_processor: MolmoAct2ImageProcessor,
        tokenizers: MolmoAct2Tokenizers,
        action_padder: ActionPadder,
        input_layout: MolmoAct2InputLayout,
        adapt_to_so101: bool = False,
    ) -> None:
        """Store focused preprocessing components."""
        super().__init__()
        self._normalizer = normalizer
        self._extractor = extractor
        self._prompt_encoder = prompt_encoder
        self._image_resize = image_resize
        self._image_packer = image_packer
        self._image_processor = image_processor
        self._tokenizers = tokenizers
        self._action_padder = action_padder
        self._input_layout = input_layout
        self._joint_transform = JointFrameTransform() if adapt_to_so101 else None

    @property
    def tokenizer(self) -> Qwen2Tokenizer:
        """The tokenizer view used by export."""
        return self._tokenizers.tokenizer

    @property
    def max_token_len(self) -> int:
        """The tokenizer output width before BOS insertion."""
        return self._tokenizers.max_token_len - 1

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Preprocess one training or inference batch.

        Returns:
            Model-ready input tensors.
        """
        self._validate(batch)
        normalized = self._normalizer(self._to_checkpoint(dict(batch)))
        bundle = self._extractor.extract(normalized)
        prompts = self._prompt_encoder.encode(bundle)
        input_ids, attention_mask = self._tokenizers.tokenize_prompts(prompts.prompt_texts)
        images, _ = self._image_packer(self._image_resize(bundle.images_by_example))
        packed = {
            TOKENIZED_PROMPT: input_ids.to(bundle.state.device),
            TOKENIZED_PROMPT_MASK: attention_mask.to(bundle.state.device),
            IMAGES: images,
        }
        output = build_model_inputs(
            packed,
            layout=self._input_layout,
            image_processor=self._image_processor,
            pad_token_id=self._tokenizers.pad_token_id,
        )
        self._add_action(output, normalized)
        return output

    def _preprocess_action(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Preprocess optional action targets.

        Returns:
            Padded action tensors and masks, or an empty dictionary.
        """
        output: dict[str, torch.Tensor] = {}
        self._add_action(output, batch)
        return output

    @staticmethod
    def _validate(batch: dict[str, Any]) -> None:
        """Require state, task, and image inputs.

        Raises:
            TypeError: If the batch is not a dictionary.
            ValueError: If a required input is absent.
        """
        if not isinstance(batch, dict):
            msg = f"MolmoAct2Preprocessor expects a dictionary, got {type(batch)}."
            raise TypeError(msg)
        if batch.get(STATE) is None:
            msg = f"{STATE} is required."
            raise ValueError(msg)
        if batch.get(TASK) is None:
            msg = f"{TASK} is required."
            raise ValueError(msg)
        nested_images = batch.get(IMAGES)
        has_nested_images = (
            any(value is not None for value in nested_images.values())
            if isinstance(nested_images, dict)
            else nested_images is not None
        )
        has_flattened_images = any(
            str(key).startswith(f"{IMAGES}.") and value is not None for key, value in batch.items()
        )
        if not has_nested_images and not has_flattened_images:
            msg = f"{IMAGES} are required."
            raise ValueError(msg)

    def _to_checkpoint(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply optional SO-101 state and training-action frame conversion.

        Returns:
            The converted batch.
        """
        if self._joint_transform is None:
            return batch
        for key in (STATE, ACTION):
            value = batch.get(key)
            if torch.is_tensor(value):
                batch[key] = self._joint_transform.to_checkpoint(value)
        return batch

    @staticmethod
    def _action_horizon_mask(batch: dict[str, Any]) -> torch.Tensor | None:
        """Resolve an action horizon mask from either Observation dictionary form.

        Returns:
            The optional nested or flattened action horizon mask.
        """
        extra = batch.get(EXTRA)
        if isinstance(extra, dict):
            mask = extra.get("action_is_pad")
            return mask if torch.is_tensor(mask) else None
        mask = batch.get(f"{EXTRA}.action_is_pad")
        return mask if torch.is_tensor(mask) else None

    def _add_action(self, output: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        """Attach padded normalized action targets when present."""
        action = ActionExtractor.extract(batch)
        if action is None:
            return
        padded, horizon_mask, dimension_mask = self._action_padder(
            action,
            self._action_horizon_mask(batch),
        )
        output[ACTION] = padded
        output["action_horizon_is_pad"] = horizon_mask
        output["action_dim_is_pad"] = dimension_mask
