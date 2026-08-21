# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL vision-language backbone interface for VLA-JEPA.

Ported from LeRobot's ``lerobot.policies.vla_jepa.qwen_interface``. Wraps
``Qwen3VLForConditionalGeneration`` with the special action-token vocabulary expansion and the
chat-template prompt construction the published VLA-JEPA checkpoints were trained with.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from physicalai.policies.vla_jepa.config import VLAJEPAConfig

logger = logging.getLogger(__name__)


def _lazy_import_transformers() -> tuple:
    """Lazy import the transformers classes the Qwen3-VL backbone needs.

    Returns:
        Tuple containing (AutoProcessor, Qwen3VLForConditionalGeneration).

    Raises:
        ImportError: If transformers is not installed.
    """
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: PLC0415
    except ImportError as e:
        msg = (
            "VLA-JEPA requires the transformers library.\n\nInstall with:\n"
            "    uv pip install 'physicalai-train[vla_jepa]'"
        )
        raise ImportError(msg) from e
    else:
        return AutoProcessor, Qwen3VLForConditionalGeneration


_GRAYSCALE_CHANNELS = 1


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map a config dtype name to a torch dtype.

    Args:
        dtype_name: One of "float32", "float16" or "bfloat16".

    Returns:
        The matching torch dtype, defaulting to bfloat16.
    """
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    return torch.bfloat16


class Qwen3VLInterface(torch.nn.Module):
    """Qwen3-VL backbone wrapper.

    Owns the processor, the special action-token vocabulary and the chat-template prompt
    construction the published VLA-JEPA checkpoints were trained with.
    """

    def __init__(self, config: VLAJEPAConfig) -> None:
        """Load the backbone and its processor.

        Args:
            config: Policy configuration naming the backbone and its dtype.
        """
        super().__init__()
        auto_processor_cls, qwen_cls = _lazy_import_transformers()
        self.config = config
        self.model = qwen_cls.from_pretrained(
            config.qwen_model_name,
            dtype=resolve_torch_dtype(config.torch_dtype),
        )
        self.processor = auto_processor_cls.from_pretrained(config.qwen_model_name)
        self.processor.tokenizer.padding_side = config.tokenizer_padding_side
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def expand_tokenizer(self) -> tuple[list[str], list[int], int]:
        # starVLA/JEVLA checkpoints expand action tokens as action_horizon * 4,
        # independent of vj2 num_action_tokens_per_timestep. Keeping this count
        # is required for Qwen embedding/lm_head checkpoint shapes to match.
        """Add the special action tokens to the tokenizer vocabulary.

        starVLA/JEVLA checkpoints expand action tokens as ``chunk_size * 4``, independently of
        `num_action_tokens_per_timestep`. Keeping that count is required for the Qwen embedding and
        lm_head checkpoint shapes to match.

        Returns:
            Tuple of (action token strings, their ids, the embodied-action token id).
        """
        max_action_tokens = self.config.chunk_size * 4
        tokenizer = self.processor.tokenizer
        action_tokens = []
        action_token_ids = []
        for idx in range(max_action_tokens):
            token = self.config.special_action_token.format(idx)
            action_tokens.append(token)
            if token not in tokenizer.get_vocab():
                tokenizer.add_tokens([token], special_tokens=True)
            action_token_ids.append(tokenizer.convert_tokens_to_ids(token))

        embodied_action_token = self.config.embodied_action_token
        if embodied_action_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([embodied_action_token], special_tokens=True)
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        # Qwen3-VL-2B ships 267 spare embedding rows, so the `chunk_size * 4 + 1` added tokens fit
        # without a resize up to chunk_size=66. Past that, resizing changes the `embed_tokens` /
        # `lm_head` shapes and checkpoints stop loading across chunk sizes unless those prefixes are
        # in `reinit_modules` — warn instead of failing silently.
        current_rows = self.model.get_input_embeddings().weight.size(0)
        if current_rows < len(tokenizer):
            logger.warning(
                "chunk_size=%s needs %d added tokens, which exceeds the %d embedding rows of %s. "
                "Resizing to %d rows changes the shapes of "
                "`model.qwen.model.model.language_model.embed_tokens` and `model.qwen.model.lm_head`, "
                "so this model will not load from a checkpoint trained with a different chunk_size "
                "unless those prefixes are in `reinit_modules`.",
                self.config.chunk_size,
                max_action_tokens + 1,
                current_rows,
                self.config.qwen_model_name,
                len(tokenizer),
            )
            self.model.resize_token_embeddings(len(tokenizer))
        return action_tokens, action_token_ids, embodied_action_token_id

    def build_inputs(
        self,
        images: Sequence[Sequence[torch.Tensor]],
        instructions: Sequence[str],
        action_prompt: str,
        embodied_prompt: str,
    ) -> dict[str, torch.Tensor]:
        """Build the tokenized chat-template inputs for a batch.

        Args:
            images: Per-sample, per-view image tensors.
            instructions: Per-sample language instruction.
            action_prompt: The per-timestep action-token placeholder string.
            embodied_prompt: The embodied-action token placeholder string.

        Returns:
            Tokenized inputs on the backbone's device.
        """
        messages = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            prompt = self.config.prompt_template.format(
                instruction=instruction,
                actions=action_prompt,
                e_actions=embodied_prompt,
            )
            content = [{"type": "image", "image": img} for img in sample_images]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        # The Qwen image processor is a torchvision-backed fast processor: passing the
        # images as GPU tensors (with `device`) keeps the whole vision pipeline on-device
        # and avoids a GPU->CPU->GPU roundtrip. The image tensors are forwarded through
        # apply_chat_template untouched into Qwen3VLProcessor.__call__.
        # do_rescale=False: images already arrive as float in [0, 1] (the dataset decoder
        # yields float32/255 and VISUAL normalization is IDENTITY), so we skip the
        # processor's /255 rescale instead of round-tripping through uint8.
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "padding": True,
                "return_tensors": "pt",
                "device": self.model.device,
                "do_rescale": False,
            },
        )
        return batch_inputs.to(self.model.device)

    @staticmethod
    def to_pixel_values(image_tensor: torch.Tensor) -> torch.Tensor:
        """Prepare an image/video tensor for the fast processors (used with do_rescale=False).

        The dataset decoder yields float32 in [0, 1] (channels-first) and VISUAL
        normalization is IDENTITY, so the tensor already arrives in [0, 1]; we pass it
        through as float and let the processors normalize (no rescale, no uint8
        quantization). A single channel is expanded to 3 to match the RGB processors.

        Works for any channels-first layout (channel dim is -3): [C, H, W], [B, C, H, W],
        [T, C, H, W], [B, V, T, C, H, W], ...

        Args:
            image_tensor: Channels-first image or video tensor with values in [0, 1].

        Returns:
            The detached float tensor, with a single channel expanded to three.
        """
        image = image_tensor.detach().float()
        if image.shape[-3] == _GRAYSCALE_CHANNELS:
            repeats = [1] * image.ndim
            repeats[-3] = 3
            image = image.repeat(*repeats)
        return image
