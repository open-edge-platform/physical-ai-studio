# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-VL conversation interface for EO-1.

Ported from the two processor steps in LeRobot's ``lerobot.policies.eo1.processor_eo1``:
``EO1ConversationTemplateStep`` (which renders a robot-control sample as a multimodal chat) and
``EO1QwenProcessorStep`` (which tokenizes it). Studio keeps them next to the model rather than in
the preprocessor pipeline, because the tokenized batch has to land on the backbone's device and the
placeholder token ids are read back by the model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from physicalai.policies.eo1.config import EO1Config

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = "You are a helpful physical assistant."

# EO-1 special tokens. The `_pad` variants are placeholders whose embeddings are replaced by the
# projected state and by the noisy action chunk; the rest only delimit them in the prompt.
ACTION_START_TOKEN = "<|action_start|>"  # noqa: S105
DEFAULT_ACTION_TOKEN = "<|action_pad|>"  # noqa: S105
ACTION_END_TOKEN = "<|action_end|>"  # noqa: S105
STATE_START_TOKEN = "<|state_start|>"  # noqa: S105
DEFAULT_STATE_TOKEN = "<|state_pad|>"  # noqa: S105
STATE_END_TOKEN = "<|state_end|>"  # noqa: S105
TASK_VLA_TOKEN = "<|vla|>"  # noqa: S105

EO1_SPECIAL_TOKENS = [
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    ACTION_END_TOKEN,
    STATE_START_TOKEN,
    DEFAULT_STATE_TOKEN,
    STATE_END_TOKEN,
    TASK_VLA_TOKEN,
]

_GRAYSCALE_CHANNELS = 1
_UINT8_MAX = 255.0


def _lazy_import_transformers() -> tuple:
    """Lazy import the transformers classes the EO-1 prompt builder needs.

    Returns:
        Tuple of (AutoImageProcessor, Qwen2_5_VLProcessor).

    Raises:
        ImportError: If transformers is not installed.
    """
    try:
        from transformers import AutoImageProcessor  # noqa: PLC0415
        from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor  # noqa: PLC0415
    except ImportError as e:
        msg = "EO-1 requires the transformers library.\n\nInstall with:\n    uv pip install 'physicalai-train[eo1]'"
        raise ImportError(msg) from e
    else:
        return AutoImageProcessor, Qwen2_5_VLProcessor


def to_uint8_image(image: torch.Tensor) -> torch.Tensor:
    """Convert a Studio image tensor to the uint8 RGB layout Qwen2.5-VL expects.

    Studio visual observations arrive as float32 in [0, 1] (VISUAL normalization is identity), while
    the Qwen2.5-VL image processor is built for uint8 in [0, 255]. Keeping the quantization matches
    what the published EO-1 checkpoints were trained on. Already-integer tensors pass through, so
    the conversion is idempotent.

    Works for any channels-first layout (the channel dim is -3): ``[C, H, W]``, ``[B, C, H, W]``.

    Args:
        image: Channels-first image tensor.

    Returns:
        A uint8 tensor with a single channel expanded to three.
    """
    if not image.is_floating_point():
        converted = image.to(torch.uint8)
    else:
        converted = image.detach().clamp(0, 1).mul(_UINT8_MAX).round().to(torch.uint8)
    if converted.shape[-3] == _GRAYSCALE_CHANNELS:
        repeats = [1] * converted.ndim
        repeats[-3] = 3
        converted = converted.repeat(*repeats)
    return converted


class EO1QwenInterface:
    """Owns the Qwen2.5-VL processor, the EO-1 token vocabulary and the prompt construction.

    Deliberately a plain object rather than a :class:`torch.nn.Module`: the processor holds no
    parameters, and keeping it out of the module tree means it contributes no state-dict keys, so
    published EO-1 checkpoints map onto the model unchanged.

    Args:
        config: Policy configuration naming the backbone and its image pixel budget.
    """

    def __init__(self, config: EO1Config) -> None:
        """Load the processor and register the EO-1 special tokens.

        Args:
            config: Policy configuration naming the backbone and its image pixel budget.
        """
        auto_image_processor_cls, processor_cls = _lazy_import_transformers()
        self.config = config
        # transformers 5.x replaced `use_fast` with the image processor's `backend`, but the
        # processor-level kwarg trips over Qwen's video processor, so pick the image processor
        # explicitly and hand it over.
        image_processor = auto_image_processor_cls.from_pretrained(
            config.vlm_base,
            backend="torchvision" if config.use_fast_processor else "pil",
        )
        self.processor = processor_cls.from_pretrained(config.vlm_base, image_processor=image_processor)
        self.processor.tokenizer.add_tokens(EO1_SPECIAL_TOKENS, special_tokens=True)
        self.state_token_id: int = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_STATE_TOKEN)
        self.action_token_id: int = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_ACTION_TOKEN)

    def __len__(self) -> int:
        """Return the tokenizer vocabulary size after the EO-1 tokens were added.

        Returns:
            Number of tokens the tokenizer can emit.
        """
        return len(self.processor.tokenizer)

    def maybe_resize_embeddings(self, model: Any) -> None:  # noqa: ANN401
        """Grow the backbone embedding table when the EO-1 tokens do not fit in its spare rows.

        Qwen2.5-VL ships more embedding rows than its tokenizer uses, so the seven added tokens
        normally fit without a resize. Resizing changes the ``embed_tokens`` and ``lm_head`` shapes,
        which would stop published checkpoints from loading, so say so when it happens.

        Args:
            model: The Qwen backbone. Untyped because transformers is imported lazily, so its real
                class is not nameable here.
        """
        current_rows = model.get_input_embeddings().weight.size(0)
        if current_rows >= len(self):
            return
        logger.warning(
            "%s has %d embedding rows but the tokenizer needs %d after adding the EO-1 special "
            "tokens. Resizing changes the shapes of `vlm_backbone.model.language_model.embed_tokens` "
            "and `vlm_backbone.lm_head`, so published EO-1 checkpoints will not load into this model.",
            self.config.vlm_base,
            current_rows,
            len(self),
        )
        model.resize_token_embeddings(len(self))

    def build_messages(
        self,
        images: Sequence[Sequence[torch.Tensor]],
        tasks: Sequence[str],
    ) -> list[list[dict[str, Any]]]:
        """Render one robot-control conversation per sample.

        The user turn carries every camera frame plus the state placeholder and the language task;
        the assistant turn carries the `chunk_size` action placeholders the flow head denoises.

        Args:
            images: Per-sample, per-camera image tensors, channels-first in [0, 1] or uint8.
            tasks: Per-sample language instruction.

        Returns:
            One chat-template message list per sample.
        """
        messages = []
        for sample_images, task in zip(images, tasks, strict=True):
            content: list[dict[str, Any]] = [
                {"type": "image", "image": to_uint8_image(image)} for image in sample_images
            ]
            content.append({
                "type": "text",
                "text": f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}{task}{TASK_VLA_TOKEN}",
            })
            action_text = f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN * self.config.chunk_size}{ACTION_END_TOKEN}"
            messages.append([
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": action_text}]},
            ])
        return messages

    def build_inputs(
        self,
        images: Sequence[Sequence[torch.Tensor]],
        tasks: Sequence[str],
        *,
        padding_side: str,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of robot-control conversations.

        Args:
            images: Per-sample, per-camera image tensors.
            tasks: Per-sample language instruction.
            padding_side: ``"right"`` for supervised batches, ``"left"`` for rollouts so the action
                span stays at the same offset in every sample.

        Returns:
            Dict with ``input_ids``, ``attention_mask``, ``pixel_values``, ``image_grid_thw`` and
            ``mm_token_type_ids``.
        """
        messages = self.build_messages(images, tasks)

        processor_kwargs: dict[str, Any] = {"padding": True, "padding_side": padding_side}
        if self.config.image_min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.image_min_pixels
        if self.config.image_max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.image_max_pixels

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )
        return {
            key: inputs[key]
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
        }
