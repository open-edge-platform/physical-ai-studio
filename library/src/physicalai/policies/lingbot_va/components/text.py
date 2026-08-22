# Copyright 2024-2025 The Robbyant Team Authors.
# Copyright 2026 The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Frozen UMT5 text-encoder plumbing for LingBot-VA.

The task string is encoded once per episode by a frozen UMT5-XXL encoder (~11 GB). It is
loaded lazily from ``config.wan_pretrained_path`` and, by default, kept on CPU so the 5B
transformer and the VAE fit on a single GPU.
"""

from __future__ import annotations

import html
import re
from typing import Any

import torch


def _lazy_import_transformers() -> tuple[Any, Any]:
    """Import the UMT5 encoder and its tokenizer from transformers.

    Returns:
        Tuple of ``(T5TokenizerFast, UMT5EncoderModel)``.

    Raises:
        ImportError: If transformers is not installed.
    """
    try:
        # T5TokenizerFast is re-exported by transformers but not visible to the type checker.
        from transformers import (  # noqa: PLC0415
            T5TokenizerFast,  # pyrefly: ignore[missing-module-attribute]
            UMT5EncoderModel,
        )
    except ImportError as e:
        msg = "LingBot-VA requires transformers.\n\nInstall with:\n    uv pip install 'physicalai-train[lingbot_va]'"
        raise ImportError(msg) from e
    return T5TokenizerFast, UMT5EncoderModel


def load_text_encoder(
    text_encoder_path: str,
    torch_dtype: torch.dtype,
    torch_device: str | torch.device,
    subfolder: str | None = None,
) -> Any:  # noqa: ANN401
    """Load the frozen UMT5 text encoder.

    Args:
        text_encoder_path: HuggingFace repo id or local directory.
        torch_dtype: Dtype to load the weights in.
        torch_device: Device to place the encoder on.
        subfolder: Sub-folder inside ``text_encoder_path`` holding the encoder.

    Returns:
        The loaded ``UMT5EncoderModel``.
    """
    _, umt5_encoder_model = _lazy_import_transformers()
    text_encoder = umt5_encoder_model.from_pretrained(  # nosec B615
        text_encoder_path,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path: str, subfolder: str | None = None) -> Any:  # noqa: ANN401
    """Load the UMT5 tokenizer.

    Args:
        tokenizer_path: HuggingFace repo id or local directory.
        subfolder: Sub-folder inside ``tokenizer_path`` holding the tokenizer.

    Returns:
        The loaded ``T5TokenizerFast``.
    """
    t5_tokenizer_fast, _ = _lazy_import_transformers()
    return t5_tokenizer_fast.from_pretrained(tokenizer_path, subfolder=subfolder)  # nosec B615


def clean_prompt(text: str) -> str:
    """Normalize a task prompt (HTML-unescape, then collapse whitespace).

    Mirrors diffusers' Wan ``prompt_clean`` minus ``ftfy.fix_text``, which is a no-op for
    the ASCII task strings used here, so the extra ``ftfy`` dependency is avoided.

    Args:
        text: Raw task string.

    Returns:
        The normalized prompt.
    """
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


def encode_prompt(
    prompts: list[str],
    tokenizer: Any,  # noqa: ANN401
    text_encoder: Any,  # noqa: ANN401
    max_sequence_length: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """UMT5-encode task strings into padded prompt embeddings.

    Padding tokens are zeroed out (rather than left as encoder output) to match the
    upstream Wan pipeline.

    Args:
        prompts: Task strings, one per batch element.
        tokenizer: The UMT5 tokenizer.
        text_encoder: The frozen UMT5 encoder.
        max_sequence_length: Padded prompt length.
        dtype: Dtype of the returned embeddings.
        device: Device of the returned embeddings.

    Returns:
        Prompt embeddings of shape ``[B, max_sequence_length, text_dim]``.
    """
    cleaned = [clean_prompt(prompt) for prompt in prompts]

    text_inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    encoder_device = next(text_encoder.parameters()).device
    prompt_embeds = text_encoder(text_input_ids.to(encoder_device), mask.to(encoder_device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    trimmed = [u[:v] for u, v in zip(prompt_embeds, seq_lens, strict=False)]
    padded = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in trimmed],
        dim=0,
    )
    return padded.to(device)
