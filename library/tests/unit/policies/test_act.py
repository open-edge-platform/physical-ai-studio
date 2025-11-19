# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import pytest
import torch
import numpy as np

from getiaction.data import Feature, FeatureType, Observation
from getiaction.policies.utils.normalization import NormalizationParameters
from getiaction.policies import ACT, ACTConfig
from getiaction.policies.act.model import ACT as ACTModel


class TestACTolicy:
    """Tests for ACTPolicy and ACTModel."""

    @pytest.fixture
    def policy(self):
        config = ACTConfig({"image": Feature(normalization_data=NormalizationParameters(mean=np.array([0]*3), std=np.array([1]*3)), shape=(3, 64, 64), ftype=FeatureType.VISUAL),
                            "state": Feature(normalization_data=NormalizationParameters(mean=np.array([0]*3), std=np.array([1]*3)), shape=(3,), ftype=FeatureType.STATE)},
                           {"action": Feature(normalization_data=NormalizationParameters(mean=np.array([0]*3), std=np.array([1]*3)), shape=(3,), ftype=FeatureType.ACTION)},
                            chunk_size=100)
        model = ACTModel.from_config(config)
        return ACT(model)

    @pytest.fixture
    def batch(self):
        bs = 2
        return Observation(
            images=torch.randn(bs, 3, 64, 64),
            state=torch.randn(bs, 3),
            action=torch.randn(bs, 100, 3),  # 'bs' samples, 3 features, 100 action steps
            extra={"action_is_pad": torch.zeros(bs, 100, dtype=torch.bool)}
        )

    def test_initialization(self, policy):
        """Check model and action shape."""
        assert isinstance(policy.model, ACTModel)
        assert policy.model._input_normalizer is not None
        assert policy.model._output_denormalizer is not None

    def test_select_action_returns_tensor(self, policy, batch):
        """select_action returns a tensor of correct shape."""
        policy.model.eval()
        actions = policy.select_action(batch)

        assert isinstance(actions, torch.Tensor)
        assert len(actions.shape) == 2
        assert actions.shape[1] == 3

    def test_forward_training_and_eval(self, policy, batch):
        """Forward pass works in training and eval modes."""
        # Training
        policy.model.train()
        loss, loss_dict = policy.model(copy.deepcopy(batch).to_dict())
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0
        assert loss_dict["kld_loss"] >= 0

        # Evaluation
        policy.model.eval()
        actions = policy.model(batch.to_dict())
        assert isinstance(actions, torch.Tensor)
        assert actions.shape == batch.action.shape

    def test_training_step(self, policy, batch):
        policy.model.train()
        loss = policy.training_step(batch, 0)

        assert "loss" in loss
        assert loss["loss"] >= 0

    def test_predict_action_chunk_with_explain(self, policy, batch):
        """Test predict_action_chunk_with_explain method."""
        policy.model.eval()
        actions, explain = policy.model.predict_action_chunk_with_explain(batch.to_dict())

        assert isinstance(actions, torch.Tensor)
        assert actions.shape == batch.action.shape
        assert isinstance(explain, torch.Tensor)
        assert explain.shape[0] == batch.action.shape[0]
        assert explain.shape[1] == 1
        assert explain.shape[2] > 1
        assert explain.shape[3] > 1

    def test_sample_input(self, policy):
        """Test sample_input generation."""
        sample_input = policy.model.sample_input

        assert isinstance(sample_input, dict)
        assert "state" in sample_input
        assert "images" in sample_input
