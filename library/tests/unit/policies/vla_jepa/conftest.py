# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small, offline VLA-JEPA fixtures."""

from dataclasses import replace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Qwen3VLConfig, VJEPA2Config

from physicalai.data import Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies.vla_jepa import VLAJEPAConfig


@pytest.fixture
def tiny_config() -> VLAJEPAConfig:
    """Return a small real Qwen/DiT architecture with a local test tokenizer."""
    tokens = [
        "<|endoftext|>",
        "<|im_end|>",
        "[UNK]",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    tokens += [f"<|action_{index}|>" for index in range(28)] + ["<|embodied_action|>"]
    tokenizer = Tokenizer(
        models.WordLevel(dict(zip(tokens, range(len(tokens)), strict=True)), unk_token="[UNK]"),  # noqa: S106 - test vocabulary.
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(tokens)
    qwen = Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=5,
        video_token_id=6,
        vision_start_token_id=3,
        vision_end_token_id=4,
    )
    inputs = [
        Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 32, 32)),
        Feature(
            name="state",
            ftype=FeatureType.STATE,
            shape=(2,),
            normalization_data=NormalizationParameters(mean=[0.0, 0.0], std=[1.0, 1.0]),
        ),
    ]
    outputs = [
        Feature(
            name="action",
            ftype=FeatureType.ACTION,
            shape=(3,),
            normalization_data=NormalizationParameters(min=[-1.0] * 3, max=[1.0] * 3),
        ),
    ]
    template = (
        "{% for message in messages %}{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %}"
    )
    return VLAJEPAConfig(
        enable_world_model=False,
        input_features=inputs,
        output_features=outputs,
        qwen_config=qwen.to_dict(),
        dtype="float32",
        tokenizer_json=tokenizer.to_str(),
        chat_template=template,
        embodied_action_token_id=35,
        image_processor_config={
            "size": {"shortest_edge": 1024, "longest_edge": 1024},
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
        },
        chunk_size=3,
        n_action_steps=2,
        action_dim=3,
        state_dim=2,
        action_hidden_size=32,
        action_num_layers=2,
        action_num_heads=2,
        action_attention_head_dim=8,
        action_dropout=0.0,
        num_target_vision_tokens=2,
        num_embodied_action_tokens=2,
        num_inference_steps=2,
        repeated_diffusion_steps=2,
        image_resolution=(32, 32),
    )


@pytest.fixture
def jepa_config(tiny_config: VLAJEPAConfig) -> VLAJEPAConfig:
    """Return a tiny joint-training configuration with a real frozen V-JEPA2 encoder."""
    encoder = VJEPA2Config(
        crop_size=32,
        patch_size=16,
        frames_per_clip=8,
        tubelet_size=2,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        mlp_ratio=2,
        pred_hidden_size=16,
        pred_num_attention_heads=2,
        pred_num_hidden_layers=1,
    )
    return replace(
        tiny_config,
        enable_world_model=True,
        world_model_config=encoder.to_dict(),
        video_processor_config={
            "size": {"shortest_edge": 32},
            "crop_size": {"height": 32, "width": 32},
            "do_resize": True,
            "do_center_crop": True,
        },
        predictor_depth=2,
        predictor_num_heads=4,
    )


@pytest.fixture
def video_observation(observation: Observation) -> Observation:
    """Return eight-frame clips with distinct future frames and matching state history."""
    observation.images = {"front": torch.rand(2, 8, 3, 32, 32)}
    observation.state = torch.randn(2, 8, 2)
    return observation


@pytest.fixture
def observation() -> Observation:
    """Return two language-conditioned samples with one episode-end padding step."""
    return Observation(
        images={"front": torch.rand(2, 3, 32, 32)},
        state=torch.randn(2, 2),
        action=torch.rand(2, 3, 3) * 2 - 1,
        task=["pick up the object", "move the object"],
        extra={"action_is_pad": torch.tensor([[False, False, True], [False, False, False]])},
    )
