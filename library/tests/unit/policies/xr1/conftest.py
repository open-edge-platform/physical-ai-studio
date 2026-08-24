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
from physicalai.policies.xr1 import XR1, XR1Config
from physicalai.policies.xr1.vla import XR1Model
from physicalai.policies.xr1.vlm import XR1Qwen3VL

VLM_HIDDEN = 128
VLM_LAYERS = 4
VLM_HEAD_DIM = 32
VLM_KV_HEADS = 2
VOCAB_SIZE = 200
#: The stock id is 151655, which does not fit the shrunken vocabulary.
IMAGE_TOKEN_ID = VOCAB_SIZE - 1
PROMPT_LENGTH = 9

VISION_PATCH_SIZE = 14
VISION_TEMPORAL_PATCH = 2
VISION_MERGE_SIZE = 2
#: One patch row per (t, h, w) cell; h and w must be multiples of the merge size.
VISION_GRID = (1, VISION_MERGE_SIZE, VISION_MERGE_SIZE)
VISION_PATCH_DIM = 3 * VISION_TEMPORAL_PATCH * VISION_PATCH_SIZE**2

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
        image_token_id=IMAGE_TOKEN_ID,
        vision_config={
            "hidden_size": 64,
            "depth": 2,
            "num_heads": 2,
            "intermediate_size": 128,
            "out_hidden_size": VLM_HIDDEN,
            "patch_size": VISION_PATCH_SIZE,
            "temporal_patch_size": VISION_TEMPORAL_PATCH,
            "spatial_merge_size": VISION_MERGE_SIZE,
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


TINY_KWARGS: dict[str, Any] = {
    "vlm_pretrained": False,
    "dtype": "float32",
    "chunk_size": CHUNK_SIZE,
    "n_action_steps": CHUNK_SIZE,
    "max_state_dim": STATE_DIM,
    "max_action_dim": ACTION_DIM,
    "dit_num_layers": VLM_LAYERS,
    "dit_hidden_size": DIT_HIDDEN,
    "dit_head_dim": VLM_HEAD_DIM,
    "dit_kv_heads": VLM_KV_HEADS,
    "num_inference_steps": 2,
    "training_repeat": 1,
    "image_resolution": (64, 64),
    "camera_views": ("top",),
    "gradient_checkpointing": False,
}


@pytest.fixture
def dataset_stats() -> dict[str, dict[str, Any]]:
    """Return dataset statistics in the shape ``Dataset.stats`` produces.

    ``type`` is a plain string, which is what LeRobot writes and therefore what ends
    up in a checkpoint's hyperparameters. Storing the enum instead would make the
    checkpoint unloadable under ``torch.load(weights_only=True)``.

    Returns:
        Statistics for one state feature, one camera and the action.
    """
    return {
        "observation.state": {
            "name": "state",
            "type": str(FeatureType.STATE),
            "shape": (DATASET_STATE_DIM,),
            "mean": [0.0] * DATASET_STATE_DIM,
            "std": [1.0] * DATASET_STATE_DIM,
        },
        "observation.images.top": {
            "name": "top",
            "type": str(FeatureType.VISUAL),
            "shape": (3, 96, 96),
            "mean": [0.0] * 3,
            "std": [1.0] * 3,
        },
        "action": {
            "name": "action",
            "type": str(FeatureType.ACTION),
            "shape": (DATASET_ACTION_DIM,),
            "mean": [0.0] * DATASET_ACTION_DIM,
            "std": [1.0] * DATASET_ACTION_DIM,
        },
    }


@pytest.fixture
def offline_backbone(monkeypatch: pytest.MonkeyPatch, tiny_vlm: XR1Qwen3VL) -> XR1Qwen3VL:
    """Make model construction use the tiny random backbone instead of the Hub.

    Args:
        monkeypatch: Pytest patcher.
        tiny_vlm: The small backbone.

    Returns:
        The backbone that will be injected.
    """
    monkeypatch.setattr(XR1Model, "_build_vlm", staticmethod(lambda _config: tiny_vlm))
    return tiny_vlm


@pytest.fixture
def policy(
    offline_backbone: XR1Qwen3VL,
    dataset_stats: dict[str, dict[str, Any]],
    stub_processor: Any,
) -> XR1:
    """Build an eagerly initialized policy with a stubbed processor.

    Args:
        offline_backbone: Ensures no Hub download happens.
        dataset_stats: Statistics driving normalization and schemas.
        stub_processor: Processor stand-in.

    Returns:
        The policy, ready for forward and inference calls.
    """
    del offline_backbone
    policy = XR1(dataset_stats=dataset_stats, **TINY_KWARGS)
    policy._preprocessor._processor = stub_processor  # noqa: SLF001 - avoids a tokenizer download
    return policy


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
            given, ``pixel_values``, ``image_grid_thw`` and ``mm_token_type_ids``.
        """
        del kwargs
        batch_size = len(text)
        views = len(images[0]) if images else 0
        # Text ids stay strictly below the image token id so no random id is mistaken
        # for an image placeholder.
        input_ids = torch.randint(0, IMAGE_TOKEN_ID, (batch_size, PROMPT_LENGTH))
        mm_token_type_ids = torch.zeros(batch_size, PROMPT_LENGTH, dtype=torch.int32)
        encoded = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(batch_size, PROMPT_LENGTH, dtype=torch.long),
        }
        if views:
            # transformers checks that the number of image placeholders equals the
            # number of merged vision features, so both sides are derived from the
            # same grid: (t * h * w) / merge ** 2 merged tokens per image.
            merged_per_image = VISION_GRID[0] * VISION_GRID[1] * VISION_GRID[2] // VISION_MERGE_SIZE**2
            placeholders = views * merged_per_image
            input_ids[:, 1 : 1 + placeholders] = IMAGE_TOKEN_ID
            mm_token_type_ids[:, 1 : 1 + placeholders] = 1

            num_images = batch_size * views
            # Shapes the real Qwen3-VL image processor would produce: one row per
            # patch, each of width in_channels * temporal_patch * patch ** 2.
            patches = num_images * VISION_GRID[1] * VISION_GRID[2]
            encoded["pixel_values"] = torch.randn(patches, VISION_PATCH_DIM)
            encoded["image_grid_thw"] = torch.tensor([VISION_GRID] * num_images, dtype=torch.long)
            encoded["mm_token_type_ids"] = mm_token_type_ids
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
