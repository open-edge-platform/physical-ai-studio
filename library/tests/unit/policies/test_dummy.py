import pytest
import torch
from torch import nn
from collections import deque
from action_trainer.policies import Dummy, DummyConfig
from action_trainer.policies.dummy.model import Dummy as DummyModel

class TestDummyPolicy:
    """Tests for DummyPolicy and DummyModel."""

    @pytest.fixture
    def policy(self):
        config = DummyConfig(action_shape=(3,))
        return Dummy(config)

    @pytest.fixture
    def batch(self):
        return {"obs": torch.randn(5, 4)}  # 5 samples, 4 features

    def test_initialization(self, policy):
        """Check model and action shape."""
        assert isinstance(policy.model, DummyModel)
        assert policy.model.action_shape == torch.Size([3])

    def test_select_action_returns_tensor(self, policy, batch):
        """select_action returns a tensor of correct shape."""
        actions = policy.select_action(batch)
        assert isinstance(actions, torch.Tensor)
        assert actions.shape[1:] == policy.model.action_shape

    def test_forward_training_and_eval(self, policy, batch):
        """Forward pass works in training and eval modes."""
        # Training
        policy.model.train()
        loss, loss_dict = policy.model(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss_dict["loss_mse"].item() >= 0

        # Evaluation
        policy.model.eval()
        actions = policy.model(batch)
        assert isinstance(actions, torch.Tensor)
        assert actions.shape[0] == batch["obs"].shape[0]

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
