# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small shared inputs for model and policy wiring tests."""

import pytest
import torch
from torch import nn

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.policies.patch_policy import model
from physicalai.policies.patch_policy.config import PatchPolicyConfig


class TinyEncoder(nn.Module):
    """Turn each constant image into two distinguishable four-wide patch tokens."""

    def __init__(self, name, output, *, encoder_config=None, image_size=(4, 4)):
        super().__init__()
        self.output_dim = 4
        self.num_tokens = 2
        self.image_size = image_size
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)
        self.compile_calls = 0
        self.weight_loads = 0

    def forward(self, images):
        value = images.mean(dim=(-3, -2, -1)) * self.weight
        patches = torch.stack((value, value + 1), dim=-1)
        return patches.unsqueeze(-1).expand(*patches.shape, self.output_dim)

    def compile(self):
        self.compile_calls += 1

    def load_pretrained_weights(self, weights):
        self.weight_loads += 1
        with torch.no_grad():
            self.weight.copy_(weights["weight"])


@pytest.fixture
def tiny_encoder(monkeypatch):
    monkeypatch.setattr(model, "TimmFeatureEncoder", TinyEncoder)
    return TinyEncoder


@pytest.fixture
def small_config():
    return PatchPolicyConfig(
        input_features=[Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 4, 4))],
        output_features=[
            Feature(
                name="action",
                ftype=FeatureType.ACTION,
                shape=(2,),
                normalization_data=NormalizationParameters(min=[0.0, 0.0], max=[2.0, 2.0]),
            ),
        ],
        action_dim=2,
        action_head="diffusion",
        n_obs_steps=2,
        chunk_size=2,
        n_action_steps=1,
        image_size=(4, 4),
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        num_train_timesteps=4,
        num_inference_steps=2,
        eta=0.0,
        use_random_input_noise=False,
        vq_latent_dim=4,
        vq_num_embeddings=2,
        vq_num_groups=1,
        vq_action_scale=1.0,
    )


@pytest.fixture
def model_batch():
    return {
        "images.front": torch.zeros(1, 2, 3, 4, 4),
        "action": torch.tensor([[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]]),
    }
