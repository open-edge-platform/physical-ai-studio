# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import tempfile
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
        config = ACTConfig({"image": Feature(normalization_data=NormalizationParameters(mean=[0.0]*3, std=[1.0]*3), shape=(3, 64, 64), ftype=FeatureType.VISUAL),
                            "state": Feature(normalization_data=NormalizationParameters(mean=[0.0]*3, std=[1.0]*3), shape=(3,), ftype=FeatureType.STATE)},
                           {"action": Feature(normalization_data=NormalizationParameters(mean=[0.0]*3, std=[1.0]*3), shape=(3,), ftype=FeatureType.ACTION)},
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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtype_change(self, policy, batch, dtype):
        """Test model behavior with different input dtypes."""
        eval_policy = copy.deepcopy(policy)
        eval_policy = eval_policy.to(dtype).eval()

        input_batch = copy.deepcopy(batch).to_dict()
        input_batch["images"] = input_batch["images"].to(dtype)
        input_batch["state"] = input_batch["state"].to(dtype)

        actions = eval_policy.model(input_batch)
        assert isinstance(actions, torch.Tensor)
        assert actions.dtype == dtype

    def test_load_from_checkpoint(self, policy):
        """Test checkpoint save and load preserves model config and weights."""
        # Save checkpoint manually (simulating Lightning checkpoint)
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = f.name

        try:
            checkpoint = {"state_dict": policy.state_dict()}
            policy.on_save_checkpoint(checkpoint)
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            torch.save(checkpoint, checkpoint_path)

            # Load from checkpoint
            loaded_policy = ACT.load_from_checkpoint(checkpoint_path)

            # Verify model type
            assert isinstance(loaded_policy.model, ACTModel)

            # Verify config is preserved
            assert list(loaded_policy.model.config.input_features.keys()) == list(
                policy.model.config.input_features.keys()
            )
            assert list(loaded_policy.model.config.output_features.keys()) == list(
                policy.model.config.output_features.keys()
            )
            assert loaded_policy.model.config.chunk_size == policy.model.config.chunk_size

            # Verify weights are loaded correctly
            orig_params = list(policy.model.parameters())
            loaded_params = list(loaded_policy.model.parameters())
            assert len(orig_params) == len(loaded_params)
            for orig, loaded in zip(orig_params, loaded_params, strict=True):
                assert torch.allclose(orig, loaded), "Weights should match after loading"

        finally:
            import os
            os.unlink(checkpoint_path)
