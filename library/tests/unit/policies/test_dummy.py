# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
from collections import deque
from getiaction.data import Observation
from getiaction.policies import Dummy, DummyConfig
from getiaction.policies.dummy.model import Dummy as DummyModel

class TestDummyPolicy:
    """Tests for DummyPolicy and DummyModel."""

    @pytest.fixture
    def policy(self):
        config = DummyConfig(action_shape=(3,))
        return Dummy(config)

    @pytest.fixture
    def batch(self):
        return Observation(state=torch.randn(5, 4))  # 5 samples, 4 features

    @pytest.fixture
    def batch_dict(self):
        return {"obs": torch.randn(5, 4)}  # 5 samples, 4 features

    def test_initialization(self, policy):
        """Check model and action shape."""
        assert isinstance(policy.model, DummyModel)
        assert policy.model.action_shape == [3]

    def test_select_action_returns_tensor(self, policy, batch):
        """select_action returns a tensor of correct shape."""
        actions = policy.select_action(batch)
        assert isinstance(actions, torch.Tensor)
        assert list(actions.shape[1:]) == policy.model.action_shape

    def test_forward_training_and_eval(self, policy, batch_dict):
        """Forward pass works in training and eval modes."""
        # Training
        policy.model.train()
        loss, loss_dict = policy.model(batch_dict)
        assert isinstance(loss, torch.Tensor)
        assert loss_dict["loss_mse"].item() >= 0

        # Evaluation
        policy.model.eval()
        actions = policy.model(batch_dict)
        assert isinstance(actions, torch.Tensor)
        assert actions.shape[0] == batch_dict["obs"].shape[0]

    def test_training_step(self, policy):
        policy.model.train()
        batch = Observation(state=torch.randn(5, 4))
        loss = policy.training_step(batch, 0)

        assert "loss" in loss
        assert loss["loss"] >= 0

    def test_action_queue_and_reset(self):
        """Action queue fills and resets correctly."""
        model = DummyModel(action_shape=torch.Size([2]), n_action_steps=3)
        batch = {"obs": torch.randn(2, 4)}
        model.eval()

        a1 = model.select_action(batch)
        assert isinstance(a1, torch.Tensor)
        assert len(model._action_queue) > 0

        model.reset()
        assert len(model._action_queue) == 0

    def test_configure_optimizers_returns_adam(self, policy):
        """Optimizer is Adam and includes model parameters."""
        optimizer = policy.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert list(optimizer.param_groups[0]["params"]) == [policy.model.dummy_param]


class TestDummyPolicyValidation:
    """Tests for DummyPolicy validation and testing."""

    def test_evaluate_gym_method_exists(self):
        """Test that Policy.evaluate_gym method exists and is callable."""
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)

        assert hasattr(policy, "evaluate_gym")
        assert callable(policy.evaluate_gym)

    def test_validation_step_accepts_gym(self):
        """Test that validation_step accepts Gym environment directly."""
        from getiaction.gyms import PushTGym
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)
        gym = PushTGym()

        # This should not raise TypeError
        result = policy.validation_step(gym, batch_idx=0)

        # Should return a dict of metrics
        assert isinstance(result, dict)
        assert all(isinstance(v, (int, float, torch.Tensor)) for v in result.values())

    def test_test_step_accepts_gym(self):
        """Test that test_step accepts Gym environment directly."""
        from getiaction.gyms import PushTGym
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)
        gym = PushTGym()

        # This should not raise TypeError
        result = policy.test_step(gym, batch_idx=0)

        # Should return a dict of metrics
        assert isinstance(result, dict)
        assert all(isinstance(v, (int, float, torch.Tensor)) for v in result.values())

    def test_validation_metrics_have_correct_keys(self):
        """Test that validation returns expected metric keys."""
        from getiaction.gyms import PushTGym
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)
        gym = PushTGym()

        metrics = policy.validation_step(gym, batch_idx=0)

        # Check for expected keys
        expected_keys = ["val/gym/episode_length", "val/gym/sum_reward", "val/gym/success"]

        for key in expected_keys:
            assert key in metrics, f"Missing expected metric: {key}"

    def test_test_metrics_use_test_prefix(self):
        """Test that test_step returns metrics with 'test/' prefix."""
        from getiaction.gyms import PushTGym
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)
        gym = PushTGym()

        metrics = policy.test_step(gym, batch_idx=0)

        # All keys should start with 'test/'
        assert all(key.startswith("test/") for key in metrics.keys())


class TestDummyPolicyImportExport:
    """Tests for DummyPolicy import/export functionality."""

    def test_export_and_import_torch(self, tmp_path):
        """Test exporting to and importing from Torch format."""
        from getiaction.policies.dummy import Dummy, DummyConfig
        from getiaction.policies.dummy.model import Dummy as DummyModel

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)

        export_path = tmp_path / "dummy_policy.pth"
        policy.to_torch(export_path)

        assert export_path.exists()

        # Import the model back
        loaded_model = DummyModel.load_checkpoint(export_path)

        assert isinstance(loaded_model, DummyModel)
        assert loaded_model.action_shape == policy.model.action_shape

    def test_export_to_onnx(self, tmp_path):
        """Test exporting to ONNX format."""
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)

        export_path = tmp_path / "dummy_policy.onnx"
        policy.to_onnx(export_path)

        assert export_path.exists()

    def test_export_to_openvino(self, tmp_path):
        """Test exporting to OpenVINO format."""
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)

        export_path = tmp_path / "dummy_policy.xml"
        policy.to_openvino(export_path)

        assert export_path.exists()
