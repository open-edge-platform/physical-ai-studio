# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from getiaction.data import Observation
from getiaction.policies import Dummy, DummyConfig
from getiaction.policies.dummy.model import Dummy as DummyModel

_openvino_available = True
try:
    import openvino  # noqa: F401
except ImportError:
    _openvino_available = False


class TestDummyPolicy:
    """Tests for DummyPolicy and DummyModel."""

    @pytest.fixture
    def policy(self):
        config = DummyConfig(action_shape=(3,))
        return Dummy(DummyModel.from_config(config))

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
        policy = Dummy(DummyModel.from_config(config))

        assert hasattr(policy, "evaluate_gym")
        assert callable(policy.evaluate_gym)

    def test_validation_step_accepts_gym(self):
        """Test that validation_step accepts Gym environment directly."""
        from getiaction.gyms import PushTGym
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(DummyModel.from_config(config))
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
        policy = Dummy(DummyModel.from_config(config))
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
        policy = Dummy(DummyModel.from_config(config))
        gym = PushTGym()

        metrics = policy.validation_step(gym, batch_idx=0)

        # Check for expected keys
        expected_keys = ["val/gym/episode_length", "val/gym/sum_reward"]

        for key in expected_keys:
            assert key in metrics, f"Missing expected metric: {key}"

    def test_test_metrics_use_test_prefix(self):
        """Test that test_step returns metrics with 'test/' prefix."""
        from getiaction.gyms import PushTGym
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(DummyModel.from_config(config))
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
        policy = Dummy(DummyModel.from_config(config))

        export_path = tmp_path / "dummy_policy.pth"
        policy.to_torch(export_path)

        assert export_path.exists()

        # Import the model back
        loaded_policy = Dummy.load_from_checkpoint(export_path)

        assert isinstance(loaded_policy, Dummy)
        assert loaded_policy.model.action_shape == policy.model.action_shape

    def test_export_to_onnx(self, tmp_path):
        """Test exporting to ONNX format."""
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(DummyModel.from_config(config))

        export_path = tmp_path / "dummy_policy.onnx"
        policy.to_onnx(export_path)

        assert export_path.exists()

    @pytest.mark.skipif(not _openvino_available, reason="openvino not installed")
    def test_export_to_openvino(self, tmp_path):
        """Test exporting to OpenVINO format."""
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(DummyModel.from_config(config))

        export_path = tmp_path / "dummy_policy.xml"
        policy.to_openvino(export_path)

        assert export_path.exists()

    def test_export_to_torch_ir(self, tmp_path):
        """Test exporting to Torch IR format."""
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(DummyModel.from_config(config))

        export_path = tmp_path / "dummy_policy_torch_ir.ptir"
        policy.to_torch_export_ir(export_path)

        assert export_path.exists()


class TestDummySample:
    """Tests for dtype/min/max sampling in Dummy._sample."""

    def test_float_default_distribution(self):
        """Float dtype with no bounds uses torch.rand."""
        out = DummyModel._sample((4, 3), torch.float32, None, None)
        assert out.dtype == torch.float32
        assert out.shape == (4, 3)
        assert torch.all((0.0 <= out) & (out <= 1.0))  # torch.rand range

    def test_float_with_bounds(self):
        """Float dtype with explicit min/max samples inside range."""
        out = DummyModel._sample((5,), torch.float32, -2.0, 2.0)
        assert out.dtype == torch.float32
        assert out.shape == (5,)
        assert torch.all((out >= -2.0) & (out <= 2.0))  # uniform bounded

    def test_float_clamped_lower(self):
        """Float dtype with only min clamps values."""
        out = DummyModel._sample((6,), torch.float32, 0.5, None)
        assert out.dtype == torch.float32
        assert torch.all(out >= 0.5)  # lower bound enforced

    def test_float_clamped_upper(self):
        """Float dtype with only max clamps values."""
        out = DummyModel._sample((6,), torch.float32, None, 0.3)
        assert out.dtype == torch.float32
        assert torch.all(out <= 0.3)  # upper bound enforced

    def test_int_default_distribution(self):
        """Int dtype with no bounds samples full dtype range."""
        out = DummyModel._sample((3, 3), torch.int32, None, None)
        assert out.dtype == torch.int32
        assert out.shape == (3, 3)

    def test_int_with_bounds(self):
        """Int dtype with explicit min/max samples inside integer bounds."""
        out = DummyModel._sample((10,), torch.int32, 1, 4)
        assert out.dtype == torch.int32
        assert torch.all((out >= 1) & (out <= 4))  # inclusive integer range

    def test_int_clamped_lower(self):
        """Int dtype with only lower bound clamps values."""
        out = DummyModel._sample((8,), torch.int32, 5, None)
        assert torch.all(out >= 5)  #  lower clamp applied

    def test_int_clamped_upper(self):
        """Int dtype with only upper bound clamps values."""
        out = DummyModel._sample((8,), torch.int32, None, 7)
        assert torch.all(out <= 7)  #  upper clamp applied
