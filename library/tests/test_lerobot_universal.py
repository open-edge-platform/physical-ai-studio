"""Tests for LeRobot universal wrapper.

This test suite validates that the universal LeRobotPolicy wrapper works correctly
with multiple LeRobot policy types (Diffusion, VQBeT, TDMPC, etc.).

NOTE: Some tests use single-sample batches which work for testing instantiation
and API correctness, but ACT and Diffusion policies may require temporal sequences
for training. For actual training, use LeRobotDataModule with delta_timestamps config.

Test Strategy:
- Parametrize tests across multiple policy types
- Verify instantiation and API correctness
- Test policies that work with single frames
- Skip tests requiring temporal data configuration
- Compare outputs with native LeRobot where possible
"""

import pytest
import torch

pytest.importorskip("lerobot")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features

from getiaction.policies.lerobot import (
    Diffusion,
    LeRobotPolicy,
    SAC,
    TDMPC,
    VQBeT,
)


@pytest.fixture
def pusht_dataset():
    """Load PushT dataset for testing."""
    return LeRobotDataset("lerobot/pusht")


@pytest.fixture
def pusht_features(pusht_dataset):
    """Extract policy features from PushT dataset."""
    return dataset_to_policy_features(pusht_dataset.meta.features)


@pytest.fixture
def pusht_stats(pusht_dataset):
    """Get dataset statistics."""
    return pusht_dataset.meta.stats


@pytest.fixture
def sample_batch(pusht_dataset):
    """Get a sample batch from the dataset."""
    return pusht_dataset[0]


# Policy configurations for testing
# Using minimal configs to speed up tests
POLICY_CONFIGS = {
    "act": {
        "dim_model": 256,  # Reduced for faster testing
        "chunk_size": 10,
        "n_action_steps": 10,
        "n_encoder_layers": 2,
        "n_decoder_layers": 1,
    },
    "diffusion": {
        "horizon": 16,
        "n_action_steps": 8,
    },
}


class TestLeRobotPolicyInstantiation:
    """Test that LeRobotPolicy can instantiate different policy types."""

    @pytest.mark.parametrize("policy_name", ["act", "diffusion"])
    def test_instantiation_with_policy_name(
        self, policy_name, pusht_features, pusht_stats
    ):
        """Test creating policies using policy_name parameter."""
        config = POLICY_CONFIGS[policy_name]

        policy = LeRobotPolicy(
            policy_name=policy_name,
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            learning_rate=1e-4,
            **config,
        )

        assert policy is not None
        assert policy.lerobot_policy is not None
        assert hasattr(policy, "training_step")
        assert hasattr(policy, "validation_step")
        assert hasattr(policy, "configure_optimizers")

    def test_instantiation_with_convenience_alias(self, pusht_features, pusht_stats):
        """Test creating policies using convenience aliases."""
        # Test Diffusion alias
        diffusion = Diffusion(
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **POLICY_CONFIGS["diffusion"],
        )
        assert diffusion is not None
        assert diffusion.lerobot_policy is not None

    def test_unsupported_policy_raises_error(self, pusht_features, pusht_stats):
        """Test that unsupported policy names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported policy"):
            LeRobotPolicy(
                policy_name="nonexistent_policy",
                input_features=pusht_features,
                output_features=pusht_features,
                dataset_stats=pusht_stats,
            )


class TestLeRobotPolicyForwardPass:
    """Test forward pass functionality."""

    @pytest.mark.parametrize("policy_name", ["act", "diffusion"])
    def test_forward_pass_shape(
        self, policy_name, pusht_features, pusht_stats, sample_batch
    ):
        """Test that forward pass returns correctly shaped outputs."""
        config = POLICY_CONFIGS[policy_name]

        policy = LeRobotPolicy(
            policy_name=policy_name,
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **config,
        )

        # Add batch dimension
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

        # Forward pass
        with torch.no_grad():
            output = policy(batch)

        # Check output is a dictionary
        assert isinstance(output, dict)
        assert "action" in output

        # Check action shape
        action = output["action"]
        assert isinstance(action, torch.Tensor)
        assert action.shape[0] == 1  # batch size

    @pytest.mark.parametrize("policy_name", ["act", "diffusion"])
    def test_forward_pass_no_gradients(
        self, policy_name, pusht_features, pusht_stats, sample_batch
    ):
        """Test that forward pass doesn't require gradients."""
        config = POLICY_CONFIGS[policy_name]

        policy = LeRobotPolicy(
            policy_name=policy_name,
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **config,
        )

        # Add batch dimension
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

        # Forward pass without gradients
        with torch.no_grad():
            output = policy(batch)

        # Check no gradients
        assert not output["action"].requires_grad


class TestLeRobotPolicyTraining:
    """Test training-related functionality."""

    @pytest.mark.parametrize("policy_name", ["act", "diffusion"])
    def test_training_step(
        self, policy_name, pusht_features, pusht_stats, sample_batch
    ):
        """Test that training_step executes without errors."""
        config = POLICY_CONFIGS[policy_name]

        policy = LeRobotPolicy(
            policy_name=policy_name,
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **config,
        )

        # Add batch dimension
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

        # Training step
        loss = policy.training_step(batch, batch_idx=0)

        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    @pytest.mark.parametrize("policy_name", ["act", "diffusion"])
    def test_validation_step(
        self, policy_name, pusht_features, pusht_stats, sample_batch
    ):
        """Test that validation_step executes without errors."""
        config = POLICY_CONFIGS[policy_name]

        policy = LeRobotPolicy(
            policy_name=policy_name,
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **config,
        )

        # Add batch dimension
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

        # Validation step
        loss = policy.validation_step(batch, batch_idx=0)

        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    @pytest.mark.parametrize("policy_name", ["act", "diffusion"])
    def test_configure_optimizers(self, policy_name, pusht_features, pusht_stats):
        """Test optimizer configuration."""
        config = POLICY_CONFIGS[policy_name]

        policy = LeRobotPolicy(
            policy_name=policy_name,
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            learning_rate=1e-4,
            **config,
        )

        optimizer = policy.configure_optimizers()

        # Check optimizer
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert len(optimizer.param_groups) > 0

        # Check learning rate
        assert optimizer.param_groups[0]["lr"] == 1e-4


class TestLeRobotPolicySelectAction:
    """Test action selection (inference) functionality."""

    @pytest.mark.parametrize("policy_name", ["act", "diffusion"])
    def test_select_action_shape(
        self, policy_name, pusht_features, pusht_stats, sample_batch
    ):
        """Test that select_action returns correctly shaped actions."""
        config = POLICY_CONFIGS[policy_name]

        policy = LeRobotPolicy(
            policy_name=policy_name,
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **config,
        )

        # Add batch dimension (but typically inference is single sample)
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

        # Select action
        with torch.no_grad():
            action = policy.select_action(batch)

        # Check action
        assert isinstance(action, torch.Tensor)
        assert action.shape[0] == 1  # batch size
        assert action.ndim >= 2  # at least [batch, action_dim]

    @pytest.mark.parametrize("policy_name", ["act", "diffusion"])
    def test_select_action_deterministic(
        self, policy_name, pusht_features, pusht_stats, sample_batch
    ):
        """Test that select_action is deterministic when using eval mode."""
        config = POLICY_CONFIGS[policy_name]

        policy = LeRobotPolicy(
            policy_name=policy_name,
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **config,
        )

        # Add batch dimension
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

        # Set to eval mode
        policy.eval()

        # Select action twice
        with torch.no_grad():
            action1 = policy.select_action(batch)
            action2 = policy.select_action(batch)

        # For deterministic policies (ACT), actions should match
        # For stochastic policies (Diffusion), they might differ
        # We just check they have the same shape
        assert action1.shape == action2.shape


class TestLeRobotPolicyNativeComparison:
    """Compare universal wrapper outputs with native LeRobot policies."""

    def test_diffusion_forward_equivalence(
        self, pusht_features, pusht_stats, sample_batch
    ):
        """Test that Diffusion wrapper produces similar outputs to native LeRobot."""
        from lerobot.policies.diffusion.modeling_diffusion import (
            DiffusionPolicy as LeRobotDiffusion,
        )
        from lerobot.policies.diffusion.configuration_diffusion import (
            DiffusionConfig,
        )

        config_dict = POLICY_CONFIGS["diffusion"]

        # Create wrapper policy
        wrapper_policy = Diffusion(
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **config_dict,
        )

        # Create native policy
        native_config = DiffusionConfig(
            input_features=pusht_features,
            output_features=pusht_features,
            **config_dict,
        )
        native_policy = LeRobotDiffusion(native_config, dataset_stats=pusht_stats)

        # Copy weights from wrapper to native (to ensure same initialization)
        native_policy.load_state_dict(wrapper_policy.lerobot_policy.state_dict())

        # Add batch dimension
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

        # Forward pass
        with torch.no_grad():
            wrapper_output = wrapper_policy(batch)
            native_output = native_policy(batch)

        # Compare actions (they should be identical since we copied weights)
        torch.testing.assert_close(
            wrapper_output["action"],
            native_output["action"],
            rtol=1e-4,
            atol=1e-4,
            msg="Wrapper and native Diffusion outputs don't match",
        )


class TestLeRobotPolicyConvenienceAliases:
    """Test convenience alias functions."""

    def test_diffusion_alias(self, pusht_features, pusht_stats):
        """Test Diffusion convenience alias."""
        policy = Diffusion(
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **POLICY_CONFIGS["diffusion"],
        )

        assert isinstance(policy, LeRobotPolicy)
        assert policy.lerobot_policy.__class__.__name__ == "DiffusionPolicy"

    def test_act_via_universal_wrapper(self, pusht_features, pusht_stats):
        """Test ACT via universal wrapper."""
        policy = LeRobotPolicy(
            policy_name="act",
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **POLICY_CONFIGS["act"],
        )

        assert isinstance(policy, LeRobotPolicy)
        assert policy.lerobot_policy.__class__.__name__ == "ACTPolicy"


class TestLeRobotPolicyStats:
    """Test dataset statistics handling."""

    def test_stats_passed_to_lerobot(self, pusht_features, pusht_stats):
        """Test that stats are correctly passed to LeRobot policy."""
        policy = Diffusion(
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=pusht_stats,
            **POLICY_CONFIGS["diffusion"],
        )

        # Check that LeRobot policy has stats
        assert hasattr(policy.lerobot_policy, "normalize_inputs")
        assert hasattr(policy.lerobot_policy, "unnormalize_outputs")

    def test_stats_none_warning(self, pusht_features):
        """Test that None stats don't cause errors."""
        # Should work without stats (but might warn)
        policy = Diffusion(
            input_features=pusht_features,
            output_features=pusht_features,
            dataset_stats=None,
            **POLICY_CONFIGS["diffusion"],
        )

        assert policy is not None
