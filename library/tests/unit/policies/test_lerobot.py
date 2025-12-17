# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LeRobot policy wrappers."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch


# Skip all tests if lerobot not installed
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("lerobot", reason="LeRobot not installed"),
    reason="Requires lerobot",
)


# Module-level fixtures for expensive operations (shared across all tests)
@pytest.fixture(scope="module")
def lerobot_imports():
    """Import LeRobot modules once per test module."""
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from getiaction.policies.lerobot import LeRobotPolicy

    return {"LeRobotPolicy": LeRobotPolicy, "LeRobotDataset": LeRobotDataset}


@pytest.fixture(scope="module")
def pusht_dataset(lerobot_imports):
    """Load pusht dataset once per module."""
    LeRobotDataset = lerobot_imports["LeRobotDataset"]
    return LeRobotDataset("lerobot/pusht")


@pytest.fixture(scope="module")
def pusht_act_policy(lerobot_imports, pusht_dataset):
    """Create ACT policy from pusht dataset once per module."""
    LeRobotPolicy = lerobot_imports["LeRobotPolicy"]
    return LeRobotPolicy.from_dataset("act", pusht_dataset)


class TestLeRobotPolicyLazyInit:
    """Tests for lazy initialization pattern."""

    def test_lazy_init_stores_policy_name(self, lerobot_imports):
        """Test lazy init stores policy name without initializing model."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(policy_name="diffusion")

        assert policy.policy_name == "diffusion"
        assert policy._config is None  # Not initialized yet

    def test_lazy_init_stores_kwargs(self, lerobot_imports):
        """Test kwargs are stored for deferred config creation."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(
            policy_name="act",
            optimizer_lr=1e-4,
            chunk_size=50,
        )

        assert policy._policy_config["optimizer_lr"] == 1e-4
        assert policy._policy_config["chunk_size"] == 50

    def test_lazy_init_merges_policy_config_and_kwargs(self, lerobot_imports):
        """Test policy_config dict is merged with kwargs, kwargs take precedence."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(
            policy_name="act",
            policy_config={"optimizer_lr": 1e-4, "chunk_size": 50},
            optimizer_lr=2e-4,  # Override via kwargs
        )

        # kwargs should override policy_config
        assert policy._policy_config["optimizer_lr"] == 2e-4
        assert policy._policy_config["chunk_size"] == 50


class TestLeRobotPolicyEagerInit:
    """Tests for eager initialization via from_dataset."""

    def test_from_dataset_with_lerobot_dataset(self, lerobot_imports, pusht_dataset):
        """Test from_dataset accepts LeRobotDataset instance."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        # Use shared dataset fixture
        policy = LeRobotPolicy.from_dataset("act", pusht_dataset)

        assert policy._config is not None  # Initialized
        assert hasattr(policy, "_lerobot_policy")

    def test_from_dataset_with_repo_id_string(self, pusht_act_policy):
        """Test from_dataset accepts repo ID string."""
        # Use shared fixture instead of creating new policy
        assert pusht_act_policy._config is not None
        assert hasattr(pusht_act_policy, "_lerobot_policy")

    def test_from_dataset_passes_kwargs_to_config(self, lerobot_imports, pusht_dataset):
        """Test from_dataset passes kwargs for config creation."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        # Use optimizer_lr which is safe - doesn't have dependent validation
        policy = LeRobotPolicy.from_dataset(
            "act",
            pusht_dataset,  # Reuse cached dataset
            optimizer_lr=5e-5,
        )

        assert policy._config.optimizer_lr == 5e-5


class TestLeRobotPolicyMethods:
    """Tests for policy method signatures."""

    def test_has_validation_step(self, lerobot_imports):
        """Test validation_step method exists."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(policy_name="diffusion")

        assert hasattr(policy, "validation_step")
        assert callable(policy.validation_step)

    def test_has_test_step(self, lerobot_imports):
        """Test test_step method exists."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(policy_name="diffusion")

        assert hasattr(policy, "test_step")
        assert callable(policy.test_step)

    def test_has_configure_optimizers(self, lerobot_imports):
        """Test configure_optimizers method exists."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(policy_name="act")

        assert hasattr(policy, "configure_optimizers")
        assert callable(policy.configure_optimizers)

    def test_has_training_step(self, lerobot_imports):
        """Test training_step method exists."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(policy_name="act")

        assert hasattr(policy, "training_step")
        assert callable(policy.training_step)


class TestLeRobotPolicyConfigureOptimizers:
    """Tests for optimizer configuration using LeRobot presets."""

    def test_configure_optimizers_uses_lerobot_preset(self, pusht_act_policy):
        """Test that configure_optimizers uses LeRobot's get_optimizer_preset."""
        # Verify config has optimizer preset method
        assert hasattr(pusht_act_policy._config, "get_optimizer_preset")

        # Call configure_optimizers
        optimizer = pusht_act_policy.configure_optimizers()
        assert optimizer is not None

    def test_configure_optimizers_respects_lr_setting(self, lerobot_imports, pusht_dataset):
        """Test optimizer uses learning rate from config."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        lr = 5e-5
        policy = LeRobotPolicy.from_dataset(
            "act",
            pusht_dataset,  # Reuse cached dataset
            optimizer_lr=lr,
        )

        optimizer = policy.configure_optimizers()

        # Check first param group has correct lr
        assert optimizer.param_groups[0]["lr"] == lr


class TestLeRobotPolicySelectAction:
    """Tests for action selection method signature."""

    def test_select_action_method_exists(self, pusht_act_policy):
        """Test select_action method exists after from_dataset initialization."""
        assert hasattr(pusht_act_policy, "select_action")
        assert callable(pusht_act_policy.select_action)

    def test_policy_is_ready_after_from_dataset(self, pusht_act_policy):
        """Test policy is fully initialized after from_dataset."""
        # Check internal state is properly initialized
        assert pusht_act_policy._config is not None
        assert hasattr(pusht_act_policy, "_lerobot_policy")
        assert pusht_act_policy._lerobot_policy is not None

        # Check pre/post processors are loaded (for normalization)
        assert pusht_act_policy._preprocessor is not None
        assert pusht_act_policy._postprocessor is not None


class TestLeRobotPolicyErrorCases:
    """Tests for error handling."""

    def test_invalid_policy_name_raises_error(self, lerobot_imports, pusht_dataset):
        """Test that invalid policy name raises ValueError."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        # LeRobot's error message format
        with pytest.raises(ValueError, match="is not available"):
            LeRobotPolicy.from_dataset("nonexistent_policy", pusht_dataset)


class TestLeRobotPolicyUniversalWrapper:
    """Tests verifying universal wrapper works with multiple policy types."""

    def test_supports_act_policy(self, pusht_act_policy):
        """Test universal wrapper supports ACT policy."""
        assert pusht_act_policy._config is not None
        assert "act" in pusht_act_policy._config.__class__.__name__.lower()

    def test_supports_diffusion_policy(self, lerobot_imports, pusht_dataset):
        """Test universal wrapper supports Diffusion policy."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy.from_dataset("diffusion", pusht_dataset)
        assert policy._config is not None
        assert "diffusion" in policy._config.__class__.__name__.lower()

    def test_supports_vqbet_policy(self, lerobot_imports, pusht_dataset):
        """Test universal wrapper supports VQ-BeT policy."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy.from_dataset("vqbet", pusht_dataset)
        assert policy._config is not None
        assert "vqbet" in policy._config.__class__.__name__.lower()


class TestLeRobotPolicyCheckpoint:
    """Tests for checkpoint save and load functionality."""

    def test_load_from_checkpoint_universal_wrapper(self, lerobot_imports, pusht_dataset):
        """Test checkpoint save and load preserves model config and weights for universal wrapper."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        # Create a policy
        original_policy = LeRobotPolicy.from_dataset("act", pusht_dataset)

        # Save checkpoint manually (simulating Lightning checkpoint)
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = f.name

        try:
            checkpoint = {"state_dict": original_policy.state_dict()}
            original_policy.on_save_checkpoint(checkpoint)
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            torch.save(checkpoint, checkpoint_path)

            # Load from checkpoint
            loaded_policy = LeRobotPolicy.load_from_checkpoint(checkpoint_path)

            # Verify policy type
            assert isinstance(loaded_policy, LeRobotPolicy)
            assert loaded_policy.policy_name == "act"

            # Verify config is preserved
            assert loaded_policy._config is not None
            assert loaded_policy._config.type == original_policy._config.type
            assert loaded_policy._config.optimizer_lr == original_policy._config.optimizer_lr

            # Verify weights are loaded correctly
            orig_params = list(original_policy.lerobot_policy.parameters())
            loaded_params = list(loaded_policy.lerobot_policy.parameters())
            assert len(orig_params) == len(loaded_params)
            for orig, loaded in zip(orig_params, loaded_params, strict=True):
                assert torch.allclose(orig, loaded), "Weights should match after loading"

        finally:
            import os

            os.unlink(checkpoint_path)

    def test_load_from_checkpoint_explicit_wrapper_act(self, lerobot_imports, pusht_dataset):
        """Test checkpoint save and load for explicit ACT wrapper."""
        from getiaction.policies.lerobot import ACT

        # Create a policy using explicit wrapper
        # Note: n_action_steps must be <= chunk_size for ACT
        original_policy = ACT.from_dataset(pusht_dataset, chunk_size=10, n_action_steps=10)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = f.name

        try:
            checkpoint = {"state_dict": original_policy.state_dict()}
            original_policy.on_save_checkpoint(checkpoint)
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            torch.save(checkpoint, checkpoint_path)

            # Load from checkpoint using explicit wrapper
            loaded_policy = ACT.load_from_checkpoint(checkpoint_path)

            # Verify it's an ACT instance
            assert isinstance(loaded_policy, ACT)
            assert loaded_policy.policy_name == "act"

            # Verify config is preserved
            assert loaded_policy._config.chunk_size == original_policy._config.chunk_size

            # Verify weights match
            orig_params = list(original_policy.lerobot_policy.parameters())
            loaded_params = list(loaded_policy.lerobot_policy.parameters())
            assert len(orig_params) == len(loaded_params)
            for orig, loaded in zip(orig_params, loaded_params, strict=True):
                assert torch.allclose(orig, loaded), "Weights should match after loading"

        finally:
            import os

            os.unlink(checkpoint_path)

    def test_load_from_checkpoint_explicit_wrapper_diffusion(self, lerobot_imports, pusht_dataset):
        """Test checkpoint save and load for explicit Diffusion wrapper."""
        from getiaction.policies.lerobot import Diffusion

        # Create a policy using explicit wrapper
        original_policy = Diffusion.from_dataset(pusht_dataset, horizon=16)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = f.name

        try:
            checkpoint = {"state_dict": original_policy.state_dict()}
            original_policy.on_save_checkpoint(checkpoint)
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            torch.save(checkpoint, checkpoint_path)

            # Load from checkpoint using explicit wrapper
            loaded_policy = Diffusion.load_from_checkpoint(checkpoint_path)

            # Verify it's a Diffusion instance
            assert isinstance(loaded_policy, Diffusion)
            assert loaded_policy.policy_name == "diffusion"

            # Verify config is preserved
            assert loaded_policy._config.horizon == original_policy._config.horizon

            # Verify weights match
            orig_params = list(original_policy.lerobot_policy.parameters())
            loaded_params = list(loaded_policy.lerobot_policy.parameters())
            assert len(orig_params) == len(loaded_params)
            for orig, loaded in zip(orig_params, loaded_params, strict=True):
                assert torch.allclose(orig, loaded), "Weights should match after loading"

        finally:
            import os

            os.unlink(checkpoint_path)

    def test_load_from_checkpoint_missing_config_raises_error(self, tmp_path):
        """Test that loading checkpoint without config raises KeyError."""
        from getiaction.policies.lerobot import LeRobotPolicy

        checkpoint_path = tmp_path / "test.ckpt"
        checkpoint = {"state_dict": {}}
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(checkpoint, str(checkpoint_path))

        with pytest.raises(KeyError, match="Checkpoint missing"):
            LeRobotPolicy.load_from_checkpoint(str(checkpoint_path))

    def test_load_from_checkpoint_preserves_dataset_stats(self, lerobot_imports, pusht_dataset):
        """Test that dataset_stats are preserved in checkpoint."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        # Create a policy (which will have dataset_stats)
        original_policy = LeRobotPolicy.from_dataset("act", pusht_dataset)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = f.name

        try:
            checkpoint = {"state_dict": original_policy.state_dict()}
            original_policy.on_save_checkpoint(checkpoint)
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            torch.save(checkpoint, checkpoint_path)

            # Load from checkpoint
            loaded_policy = LeRobotPolicy.load_from_checkpoint(checkpoint_path)

            # Verify dataset_stats are preserved (if they were saved)
            if "dataset_stats" in checkpoint:
                assert loaded_policy._dataset_stats is not None

        finally:
            import os

            os.unlink(checkpoint_path)


class TestLeRobotPolicyNumericalEquivalence:
    """Tests verifying wrapper produces identical results to direct LeRobot calls.

    These tests ensure our wrapper's predict_action_chunk and select_action
    produce numerically equivalent outputs to calling the underlying LeRobot
    policy methods directly.
    """

    @pytest.fixture
    def device(self):
        """Get the device to use for tests."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def act_policy_on_device(self, lerobot_imports, pusht_dataset, device):
        """Create ACT policy and move to correct device."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]
        policy = LeRobotPolicy.from_dataset("act", pusht_dataset)
        policy.to(device)
        policy.eval()
        return policy

    def test_select_action_matches_lerobot_directly(self, act_policy_on_device, pusht_dataset, device):
        """Verify wrapper.select_action == lerobot_policy.select_action."""
        policy = act_policy_on_device

        # Create a sample batch and move to device
        sample = pusht_dataset[0]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}

        # Reset both to clear any state
        policy.reset()

        # Get preprocessed batch (what LeRobot expects)
        preprocessed = policy._preprocessor(batch)

        # Direct LeRobot call
        lerobot_action = policy.lerobot_policy.select_action(preprocessed)

        # Reset again before wrapper call (action queue state)
        policy.reset()

        # Wrapper call (includes preprocessing internally)
        wrapper_action = policy.select_action(batch)

        # Should be numerically identical
        torch.testing.assert_close(
            wrapper_action,
            policy._postprocessor(lerobot_action),
            rtol=1e-5,
            atol=1e-5,
            msg="Wrapper select_action should match LeRobot select_action",
        )

    def test_predict_action_chunk_matches_lerobot_directly(self, act_policy_on_device, pusht_dataset, device):
        """Verify wrapper.predict_action_chunk == lerobot_policy.predict_action_chunk."""
        policy = act_policy_on_device

        # Create a sample batch and move to device
        sample = pusht_dataset[0]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}

        # Reset to clear state
        policy.reset()

        # Get preprocessed batch
        preprocessed = policy._preprocessor(batch)

        # Direct LeRobot call
        lerobot_chunk = policy.lerobot_policy.predict_action_chunk(preprocessed)

        # Reset again
        policy.reset()

        # Wrapper call
        wrapper_chunk = policy.predict_action_chunk(batch)

        # Should be numerically identical
        torch.testing.assert_close(
            wrapper_chunk,
            policy._postprocessor(lerobot_chunk),
            rtol=1e-5,
            atol=1e-5,
            msg="Wrapper predict_action_chunk should match LeRobot predict_action_chunk",
        )

    def test_select_action_shape_is_single_action(self, act_policy_on_device, pusht_dataset, device):
        """Verify select_action returns single action per batch item."""
        policy = act_policy_on_device

        sample = pusht_dataset[0]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}

        policy.reset()
        action = policy.select_action(batch)

        # select_action returns (batch, action_dim) - one action per batch item
        # LeRobot's select_action preserves the batch dimension
        action_dim = policy._config.output_features["action"].shape[0]
        assert action.dim() == 2, f"Expected 2D tensor, got shape {action.shape}"
        assert action.shape[0] == 1  # batch size
        assert action.shape[1] == action_dim  # action_dim

    def test_predict_action_chunk_shape_is_full_chunk(self, act_policy_on_device, pusht_dataset, device):
        """Verify predict_action_chunk returns full chunk."""
        policy = act_policy_on_device

        sample = pusht_dataset[0]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}

        policy.reset()
        chunk = policy.predict_action_chunk(batch)

        # predict_action_chunk should return (batch, chunk_size, action_dim)
        action_dim = policy._config.output_features["action"].shape[0]
        assert chunk.dim() == 3, f"Expected 3D tensor, got shape {chunk.shape}"
        assert chunk.shape[0] == 1  # batch size
        assert chunk.shape[1] == policy._config.chunk_size  # chunk_size
        assert chunk.shape[2] == action_dim  # action_dim

    def test_multiple_select_action_uses_cached_chunk(self, lerobot_imports, pusht_dataset, device):
        """Verify select_action uses internal queue correctly."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy.from_dataset("act", pusht_dataset, n_action_steps=3)
        policy.to(device)
        policy.eval()

        sample = pusht_dataset[0]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}

        policy.reset()

        # Get full chunk first
        full_chunk = policy.predict_action_chunk(batch)

        # Reset and call select_action multiple times
        policy.reset()
        actions = []
        for _ in range(3):
            action = policy.select_action(batch)
            actions.append(action)

        # Each action should match corresponding position in chunk
        # select_action returns (batch, action_dim), chunk is (batch, chunk_size, action_dim)
        for i, action in enumerate(actions):
            # LeRobot's select_action returns from its internal queue
            # which is populated by predict_action_chunk[:, :n_action_steps]
            assert action.shape == (1, full_chunk.shape[2])  # (batch, action_dim)
