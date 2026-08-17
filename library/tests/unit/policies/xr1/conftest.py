# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the XR1 unit tests.

Every fixture is deliberately tiny and randomly initialized: the released XR1
configuration is 5.04B parameters, so the tests build a ~1M-parameter backbone
instead of downloading a 9.4 GiB checkpoint.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.policies.xr1 import XR1Config
from physicalai.policies.xr1.vlm import XR1Qwen3VL

VLM_HIDDEN = 128
VLM_LAYERS = 4
VLM_HEAD_DIM = 32
VLM_KV_HEADS = 2
VOCAB_SIZE = 200

DIT_HIDDEN = 256
CHUNK_SIZE = 4
STATE_DIM = 8
ACTION_DIM = 8
DATASET_STATE_DIM = 5
DATASET_ACTION_DIM = 6


@pytest.fixture
def tiny_vlm() -> XR1Qwen3VL:
    """Build a small randomly initialized Qwen3-VL backbone.

    Function-scoped on purpose: tests that exercise freezing call
    ``requires_grad_(False)``, which would leak into later tests through a shared
    instance. The backbone is ~1.2M parameters, so rebuilding it is cheap.

    Returns:
        A backbone whose depth and KV geometry match the tiny XR1 config.
    """
    from transformers import Qwen3VLConfig

    # ``text_config`` and ``vision_config`` are keyword-only parameters that
    # transformers injects dynamically, so type checkers cannot see them even though
    # the runtime signature accepts them.
    config = Qwen3VLConfig(  # type: ignore[call-arg]
        text_config={
            "hidden_size": VLM_HIDDEN,
            "num_hidden_layers": VLM_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": VLM_KV_HEADS,
            "head_dim": VLM_HEAD_DIM,
            "intermediate_size": 256,
            "vocab_size": VOCAB_SIZE,
        },
        vision_config={
            "hidden_size": 64,
            "depth": 2,
            "num_heads": 2,
            "intermediate_size": 128,
            "out_hidden_size": VLM_HIDDEN,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
        },
    )
    return XR1Qwen3VL._from_config(  # noqa: SLF001 - no public build-from-config entry point
        config,
        attn_implementation="sdpa",
        dtype=torch.float32,
    )


@pytest.fixture
def tiny_config() -> XR1Config:
    """Return a config matching the tiny backbone.

    Returns:
        A config small enough to run on CPU in a unit test.
    """
    return XR1Config(
        vlm_pretrained=False,
        dtype="float32",
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        max_state_dim=STATE_DIM,
        max_action_dim=ACTION_DIM,
        dit_num_layers=VLM_LAYERS,
        dit_hidden_size=DIT_HIDDEN,
        dit_head_dim=VLM_HEAD_DIM,
        dit_kv_heads=VLM_KV_HEADS,
        num_inference_steps=2,
        training_repeat=1,
        image_resolution=(64, 64),
        camera_views=("top", "wrist"),
        gradient_checkpointing=False,
    )


class StubProcessor:
    """Minimal stand-in for the Qwen3-VL processor.

    The real processor would download tokenizer files; the tests only need stable
    shapes and a rendered prompt string.
    """

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render a prompt by concatenating text parts and image placeholders.

        Args:
            messages: Chat messages in the transformers format.
            tokenize: Unused, matches the real signature.
            add_generation_prompt: Unused, matches the real signature.

        Returns:
            The rendered prompt.
        """
        del tokenize, add_generation_prompt
        parts = [part.get("text", "<image>") for part in messages[0]["content"]]
        return " ".join(parts)

    def __call__(
        self,
        text: list[str],
        images: list[list[torch.Tensor]] | None = None,
        **kwargs: Any,  # noqa: ANN401 - matches the real processor's signature
    ) -> dict[str, torch.Tensor]:
        """Return fixed-shape token and pixel tensors.

        Args:
            text: One prompt per sample.
            images: Per-sample image lists, when images are present.
            **kwargs: Ignored.

        Returns:
            Encoded batch with ``input_ids``, ``attention_mask`` and, when images are
            given, ``pixel_values`` and ``image_grid_thw``.
        """
        del kwargs
        batch_size = len(text)
        encoded = {
            "input_ids": torch.randint(0, VOCAB_SIZE, (batch_size, 9)),
            "attention_mask": torch.ones(batch_size, 9, dtype=torch.long),
        }
        if images:
            num_images = batch_size * len(images[0])
            encoded["pixel_values"] = torch.randn(num_images, 3, 16, 16)
            encoded["image_grid_thw"] = torch.ones(num_images, 3, dtype=torch.long)
        return encoded


@pytest.fixture
def stub_processor() -> StubProcessor:
    """Return a processor stub.

    Returns:
        The stub instance.
    """
    return StubProcessor()


@pytest.fixture
def dataset_features() -> dict[str, Feature]:
    """Return a state/action feature schema with unit statistics.

    Returns:
        Feature schema keyed by feature name.
    """
    return {
        "state": Feature(
            name="state",
            ftype=FeatureType.STATE,
            shape=(DATASET_STATE_DIM,),
            normalization_data=NormalizationParameters(
                mean=[0.0] * DATASET_STATE_DIM,
                std=[1.0] * DATASET_STATE_DIM,
            ),
        ),
        "action": Feature(
            name="action",
            ftype=FeatureType.ACTION,
            shape=(DATASET_ACTION_DIM,),
            normalization_data=NormalizationParameters(
                mean=[0.0] * DATASET_ACTION_DIM,
                std=[1.0] * DATASET_ACTION_DIM,
            ),
        ),
    }


@pytest.fixture
def observation_batch() -> dict[str, Any]:
    """Return a two-sample observation batch in dict form.

    Returns:
        Batch with state, action, two camera views and per-sample instructions.
    """
    return {
        "state": torch.randn(2, DATASET_STATE_DIM),
        "action": torch.randn(2, CHUNK_SIZE, DATASET_ACTION_DIM),
        "images": {
            "top": torch.rand(2, 3, 96, 128),
            "wrist": torch.rand(2, 3, 96, 96),
        },
        "task": ["transfer the cube", "transfer the cube"],
    }


@pytest.fixture
def model_batch() -> dict[str, torch.Tensor]:
    """Return a batch already in the model's input format.

    Returns:
        Batch with token ids, attention mask, state, action and action mask.
    """
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (2, 7)),
        "attention_mask": torch.ones(2, 7, dtype=torch.long),
        "state": torch.randn(2, 1, STATE_DIM),
        "action": torch.randn(2, CHUNK_SIZE, ACTION_DIM),
        "action_mask": torch.ones(2, CHUNK_SIZE, ACTION_DIM),
    }
