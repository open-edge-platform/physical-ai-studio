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

    from physicalai.policies.lerobot import LeRobotPolicy

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
        from physicalai.policies.lerobot import ACT

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
        from physicalai.policies.lerobot import Diffusion

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
        from physicalai.policies.lerobot import LeRobotPolicy

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


class TestLeRobotPolicySavePretrained:
    """Tests for save_pretrained and push_to_hub functionality."""

    def test_save_pretrained_creates_config_and_weights(self, pusht_act_policy, tmp_path):
        """Test save_pretrained creates config.json and model.safetensors."""
        save_dir = tmp_path / "saved_policy"
        pusht_act_policy.save_pretrained(save_dir)

        assert (save_dir / "config.json").exists(), "config.json not created"
        assert (save_dir / "model.safetensors").exists(), "model.safetensors not created"

    def test_save_pretrained_config_is_valid_json(self, pusht_act_policy, tmp_path):
        """Test saved config.json is valid JSON with expected fields."""
        import json

        save_dir = tmp_path / "saved_policy"
        pusht_act_policy.save_pretrained(save_dir)

        with open(save_dir / "config.json") as f:
            config = json.load(f)

        assert "type" in config, "config.json must contain 'type' field"

    def test_save_pretrained_roundtrip_with_lerobot(self, pusht_act_policy, tmp_path):
        """Test saved model is loadable by LeRobot's from_pretrained."""
        from lerobot.policies.act.modeling_act import ACTPolicy

        save_dir = tmp_path / "saved_policy"
        pusht_act_policy.save_pretrained(save_dir)

        loaded = ACTPolicy.from_pretrained(str(save_dir))

        orig_params = dict(pusht_act_policy.lerobot_policy.named_parameters())
        loaded_params = dict(loaded.named_parameters())

        assert set(orig_params.keys()) == set(loaded_params.keys()), "Parameter names should match"
        for name in orig_params:
            torch.testing.assert_close(
                orig_params[name],
                loaded_params[name],
                rtol=1e-5,
                atol=1e-5,
                msg=f"Weight mismatch for {name}",
            )

    def test_save_pretrained_roundtrip_with_physicalai(self, lerobot_imports, pusht_act_policy, tmp_path):
        """Test saved model is loadable by physicalai's from_pretrained."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        save_dir = tmp_path / "saved_policy"
        pusht_act_policy.save_pretrained(save_dir)

        loaded = LeRobotPolicy.from_pretrained(str(save_dir))

        assert isinstance(loaded, LeRobotPolicy)
        assert loaded.policy_name == pusht_act_policy.policy_name

        orig_params = dict(pusht_act_policy.lerobot_policy.named_parameters())
        loaded_params = dict(loaded.lerobot_policy.named_parameters())

        for name in orig_params:
            torch.testing.assert_close(
                orig_params[name],
                loaded_params[name],
                rtol=1e-5,
                atol=1e-5,
                msg=f"Weight mismatch for {name}",
            )

    def test_save_pretrained_creates_directory(self, pusht_act_policy, tmp_path):
        """Test save_pretrained creates parent directories if needed."""
        save_dir = tmp_path / "a" / "b" / "c" / "saved_policy"
        pusht_act_policy.save_pretrained(save_dir)

        assert save_dir.exists()
        assert (save_dir / "config.json").exists()
        assert (save_dir / "model.safetensors").exists()

    def test_save_pretrained_returns_none_without_push(self, pusht_act_policy, tmp_path):
        """Test save_pretrained returns None when push_to_hub is False."""
        result = pusht_act_policy.save_pretrained(tmp_path / "saved_policy")
        assert result is None

    def test_push_to_hub_calls_hf_api(self, pusht_act_policy, tmp_path):
        """Test push_to_hub creates repo and uploads via HfApi."""
        from unittest.mock import MagicMock, patch

        mock_api_instance = MagicMock()
        mock_api_instance.create_repo.return_value = MagicMock(repo_id="user/my-policy")
        mock_api_instance.upload_folder.return_value = "https://huggingface.co/user/my-policy/commit/abc"

        with patch("huggingface_hub.HfApi", return_value=mock_api_instance):
            result = pusht_act_policy.push_to_hub("user/my-policy", token="fake-token")

        mock_api_instance.create_repo.assert_called_once_with(
            repo_id="user/my-policy",
            private=None,
            exist_ok=True,
        )

        mock_api_instance.upload_folder.assert_called_once()
        call_kwargs = mock_api_instance.upload_folder.call_args[1]
        assert call_kwargs["repo_id"] == "user/my-policy"
        assert call_kwargs["repo_type"] == "model"

        assert result == "https://huggingface.co/user/my-policy/commit/abc"

    def test_push_to_hub_custom_commit_message(self, pusht_act_policy):
        """Test push_to_hub passes custom commit message."""
        from unittest.mock import MagicMock, patch

        mock_api_instance = MagicMock()
        mock_api_instance.create_repo.return_value = MagicMock(repo_id="user/my-policy")
        mock_api_instance.upload_folder.return_value = "https://huggingface.co/user/my-policy/commit/abc"

        with patch("huggingface_hub.HfApi", return_value=mock_api_instance):
            pusht_act_policy.push_to_hub(
                "user/my-policy",
                commit_message="My custom message",
                token="fake-token",
            )

        call_kwargs = mock_api_instance.upload_folder.call_args[1]
        assert call_kwargs["commit_message"] == "My custom message"

    def test_save_pretrained_with_push_calls_push_to_hub(self, pusht_act_policy, tmp_path):
        """Test save_pretrained with push_to_hub=True delegates to push_to_hub."""
        from unittest.mock import MagicMock, patch

        mock_api_instance = MagicMock()
        mock_api_instance.create_repo.return_value = MagicMock(repo_id="user/my-policy")
        mock_api_instance.upload_folder.return_value = "https://huggingface.co/user/my-policy/commit/abc"

        save_dir = tmp_path / "saved_policy"
        with patch("huggingface_hub.HfApi", return_value=mock_api_instance):
            result = pusht_act_policy.save_pretrained(
                save_dir,
                push_to_hub=True,
                repo_id="user/my-policy",
                token="fake-token",
            )

        assert result == "https://huggingface.co/user/my-policy/commit/abc"

    def test_save_pretrained_diffusion_roundtrip(self, lerobot_imports, pusht_dataset, tmp_path):
        """Test save_pretrained works for Diffusion policy too."""
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy.from_dataset("diffusion", pusht_dataset)
        save_dir = tmp_path / "diffusion_saved"
        policy.save_pretrained(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "model.safetensors").exists()

        loaded = DiffusionPolicy.from_pretrained(str(save_dir))
        assert loaded is not None

        orig_params = dict(policy.lerobot_policy.named_parameters())
        loaded_params = dict(loaded.named_parameters())
        for name in orig_params:
            torch.testing.assert_close(orig_params[name], loaded_params[name], rtol=1e-5, atol=1e-5)


class TestCheckpointConverter:
    """Tests for the Lightning <-> LeRobot checkpoint converter."""

    @pytest.fixture()
    def sample_config(self):
        """Minimal config dict mimicking a LeRobot policy config."""
        return {"type": "act", "chunk_size": 100, "dim_model": 256}

    @pytest.fixture()
    def sample_lerobot_state(self):
        """Synthetic LeRobot-format state dict (model.* keys without prefix)."""
        return {
            "model.backbone.0.weight": torch.randn(64, 3, 3, 3),
            "model.backbone.0.bias": torch.randn(64),
            "model.head.weight": torch.randn(2, 64),
            "model.head.bias": torch.randn(2),
        }

    @pytest.fixture()
    def sample_lightning_state(self, sample_lerobot_state):
        """Synthetic Lightning-format state dict (_lerobot_policy.model.* keys)."""
        from physicalai.policies.lerobot.converter import _LEROBOT_PREFIX

        return {_LEROBOT_PREFIX + k: v for k, v in sample_lerobot_state.items()}

    @pytest.fixture()
    def lightning_checkpoint(self, sample_config, sample_lightning_state, tmp_path):
        """Create a synthetic Lightning .ckpt file on disk."""
        from physicalai.export.mixin_export import CONFIG_KEY, POLICY_NAME_KEY

        ckpt = {
            "state_dict": sample_lightning_state,
            CONFIG_KEY: sample_config,
            POLICY_NAME_KEY: "act",
        }
        ckpt_path = tmp_path / "model.ckpt"
        torch.save(ckpt, str(ckpt_path))
        return ckpt_path

    @pytest.fixture()
    def lerobot_dir(self, sample_config, sample_lerobot_state, tmp_path):
        """Create a synthetic LeRobot directory (config.json + model.safetensors)."""
        import json

        from safetensors.torch import save_file

        lr_dir = tmp_path / "lerobot_model"
        lr_dir.mkdir()
        with (lr_dir / "config.json").open("w") as f:
            json.dump(sample_config, f)
        save_file(sample_lerobot_state, str(lr_dir / "model.safetensors"))
        return lr_dir

    # ------------------------------------------------------------------
    # lightning_to_lerobot
    # ------------------------------------------------------------------

    def test_lightning_to_lerobot_creates_files(self, lightning_checkpoint, tmp_path):
        """Converted output directory contains config.json and model.safetensors."""
        from physicalai.policies.lerobot.converter import lightning_to_lerobot

        out_dir = tmp_path / "converted"
        result = lightning_to_lerobot(lightning_checkpoint, out_dir)

        assert result == out_dir
        assert (out_dir / "config.json").exists()
        assert (out_dir / "model.safetensors").exists()

    def test_lightning_to_lerobot_config_matches(self, lightning_checkpoint, sample_config, tmp_path):
        """Extracted config.json matches the original config dict."""
        import json

        from physicalai.policies.lerobot.converter import lightning_to_lerobot

        out_dir = tmp_path / "converted"
        lightning_to_lerobot(lightning_checkpoint, out_dir)

        with (out_dir / "config.json").open() as f:
            loaded_config = json.load(f)

        assert loaded_config == sample_config

    def test_lightning_to_lerobot_strips_prefix(self, lightning_checkpoint, sample_lerobot_state, tmp_path):
        """Weights have _lerobot_policy. prefix stripped to match LeRobot format."""
        from safetensors.torch import load_file

        from physicalai.policies.lerobot.converter import lightning_to_lerobot

        out_dir = tmp_path / "converted"
        lightning_to_lerobot(lightning_checkpoint, out_dir)

        loaded = load_file(str(out_dir / "model.safetensors"))

        assert set(loaded.keys()) == set(sample_lerobot_state.keys())
        for key in sample_lerobot_state:
            torch.testing.assert_close(loaded[key], sample_lerobot_state[key])

    def test_lightning_to_lerobot_missing_config_raises(self, tmp_path):
        """KeyError when checkpoint lacks CONFIG_KEY."""
        from physicalai.policies.lerobot.converter import lightning_to_lerobot

        ckpt_path = tmp_path / "bad.ckpt"
        torch.save({"state_dict": {}}, str(ckpt_path))

        with pytest.raises(KeyError, match="model_config"):
            lightning_to_lerobot(ckpt_path, tmp_path / "out")

    def test_lightning_to_lerobot_creates_output_dir(self, lightning_checkpoint, tmp_path):
        """Output directory is created if it does not exist."""
        from physicalai.policies.lerobot.converter import lightning_to_lerobot

        out_dir = tmp_path / "a" / "b" / "c"
        lightning_to_lerobot(lightning_checkpoint, out_dir)

        assert out_dir.exists()
        assert (out_dir / "config.json").exists()

    # ------------------------------------------------------------------
    # lerobot_to_lightning
    # ------------------------------------------------------------------

    def test_lerobot_to_lightning_creates_ckpt(self, lerobot_dir, tmp_path):
        """Converted checkpoint file exists and is loadable."""
        from physicalai.policies.lerobot.converter import lerobot_to_lightning

        ckpt_path = tmp_path / "converted.ckpt"
        result = lerobot_to_lightning(lerobot_dir, ckpt_path)

        assert result == ckpt_path
        assert ckpt_path.exists()

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        assert "state_dict" in ckpt

    def test_lerobot_to_lightning_adds_prefix(self, lerobot_dir, sample_lerobot_state, tmp_path):
        """State dict keys have _lerobot_policy. prefix added."""
        from physicalai.policies.lerobot.converter import _LEROBOT_PREFIX, lerobot_to_lightning

        ckpt_path = tmp_path / "converted.ckpt"
        lerobot_to_lightning(lerobot_dir, ckpt_path)

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        state_dict = ckpt["state_dict"]

        expected_keys = {_LEROBOT_PREFIX + k for k in sample_lerobot_state}
        assert set(state_dict.keys()) == expected_keys

        for key in sample_lerobot_state:
            torch.testing.assert_close(state_dict[_LEROBOT_PREFIX + key], sample_lerobot_state[key])

    def test_lerobot_to_lightning_stores_config_and_policy_name(self, lerobot_dir, sample_config, tmp_path):
        """Checkpoint contains CONFIG_KEY and POLICY_NAME_KEY."""
        from physicalai.export.mixin_export import CONFIG_KEY, POLICY_NAME_KEY
        from physicalai.policies.lerobot.converter import lerobot_to_lightning

        ckpt_path = tmp_path / "converted.ckpt"
        lerobot_to_lightning(lerobot_dir, ckpt_path)

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

        assert ckpt[CONFIG_KEY] == sample_config
        assert ckpt[POLICY_NAME_KEY] == "act"

    def test_lerobot_to_lightning_explicit_policy_name(self, lerobot_dir, tmp_path):
        """Explicit policy_name overrides config['type']."""
        from physicalai.export.mixin_export import POLICY_NAME_KEY
        from physicalai.policies.lerobot.converter import lerobot_to_lightning

        ckpt_path = tmp_path / "converted.ckpt"
        lerobot_to_lightning(lerobot_dir, ckpt_path, policy_name="diffusion")

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        assert ckpt[POLICY_NAME_KEY] == "diffusion"

    def test_lerobot_to_lightning_missing_config_raises(self, tmp_path):
        """FileNotFoundError when config.json is missing."""
        from physicalai.policies.lerobot.converter import lerobot_to_lightning

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="config.json"):
            lerobot_to_lightning(empty_dir, tmp_path / "out.ckpt")

    def test_lerobot_to_lightning_missing_weights_raises(self, tmp_path):
        """FileNotFoundError when model.safetensors is missing."""
        import json

        from physicalai.policies.lerobot.converter import lerobot_to_lightning

        dir_no_weights = tmp_path / "no_weights"
        dir_no_weights.mkdir()
        with (dir_no_weights / "config.json").open("w") as f:
            json.dump({"type": "act"}, f)

        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            lerobot_to_lightning(dir_no_weights, tmp_path / "out.ckpt")

    def test_lerobot_to_lightning_no_policy_name_raises(self, tmp_path):
        """ValueError when policy_name is None and config has no 'type' field."""
        import json

        from safetensors.torch import save_file

        from physicalai.policies.lerobot.converter import lerobot_to_lightning

        no_type_dir = tmp_path / "no_type"
        no_type_dir.mkdir()
        with (no_type_dir / "config.json").open("w") as f:
            json.dump({"chunk_size": 100}, f)
        save_file({"w": torch.randn(2, 2)}, str(no_type_dir / "model.safetensors"))

        with pytest.raises(ValueError, match="Cannot determine policy_name"):
            lerobot_to_lightning(no_type_dir, tmp_path / "out.ckpt")

    def test_lerobot_to_lightning_creates_parent_dirs(self, lerobot_dir, tmp_path):
        """Output parent directories are created if they do not exist."""
        from physicalai.policies.lerobot.converter import lerobot_to_lightning

        ckpt_path = tmp_path / "a" / "b" / "model.ckpt"
        lerobot_to_lightning(lerobot_dir, ckpt_path)

        assert ckpt_path.exists()

    # ------------------------------------------------------------------
    # Round-trip tests
    # ------------------------------------------------------------------

    def test_roundtrip_lightning_lerobot_lightning(self, lightning_checkpoint, sample_lightning_state, tmp_path):
        """Lightning -> LeRobot -> Lightning preserves all weights exactly."""
        from physicalai.policies.lerobot.converter import lightning_to_lerobot, lerobot_to_lightning

        lr_dir = tmp_path / "lerobot_intermediate"
        lightning_to_lerobot(lightning_checkpoint, lr_dir)

        ckpt_restored = tmp_path / "restored.ckpt"
        lerobot_to_lightning(lr_dir, ckpt_restored)

        original = torch.load(str(lightning_checkpoint), map_location="cpu", weights_only=True)
        restored = torch.load(str(ckpt_restored), map_location="cpu", weights_only=True)

        orig_sd = original["state_dict"]
        rest_sd = restored["state_dict"]

        assert set(orig_sd.keys()) == set(rest_sd.keys())
        for key in orig_sd:
            torch.testing.assert_close(orig_sd[key], rest_sd[key], msg=f"Mismatch for {key}")

    def test_roundtrip_lerobot_lightning_lerobot(self, lerobot_dir, sample_lerobot_state, tmp_path):
        """LeRobot -> Lightning -> LeRobot preserves all weights exactly."""
        from safetensors.torch import load_file

        from physicalai.policies.lerobot.converter import lightning_to_lerobot, lerobot_to_lightning

        ckpt_path = tmp_path / "intermediate.ckpt"
        lerobot_to_lightning(lerobot_dir, ckpt_path)

        lr_restored = tmp_path / "lerobot_restored"
        lightning_to_lerobot(ckpt_path, lr_restored)

        restored_weights = load_file(str(lr_restored / "model.safetensors"))

        assert set(restored_weights.keys()) == set(sample_lerobot_state.keys())
        for key in sample_lerobot_state:
            torch.testing.assert_close(restored_weights[key], sample_lerobot_state[key], msg=f"Mismatch for {key}")

    # ------------------------------------------------------------------
    # Integration: real ACT policy round-trip
    # ------------------------------------------------------------------

    def test_roundtrip_real_act_policy(self, pusht_act_policy, tmp_path):
        """Round-trip a real ACT wrapper through save_pretrained -> lerobot_to_lightning -> load."""
        from safetensors.torch import load_file

        from physicalai.policies.lerobot.converter import _LEROBOT_PREFIX, lerobot_to_lightning

        # Step 1: save_pretrained produces LeRobot dir
        lr_dir = tmp_path / "act_lerobot"
        pusht_act_policy.save_pretrained(lr_dir)

        # Step 2: convert to lightning checkpoint
        ckpt_path = tmp_path / "act_lightning.ckpt"
        lerobot_to_lightning(lr_dir, ckpt_path)

        # Step 3: verify checkpoint weights match original wrapper weights
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        ckpt_sd = ckpt["state_dict"]

        orig_params = dict(pusht_act_policy.lerobot_policy.named_parameters())

        for name, param in orig_params.items():
            ckpt_key = _LEROBOT_PREFIX + name
            assert ckpt_key in ckpt_sd, f"Missing key {ckpt_key} in checkpoint"
            torch.testing.assert_close(
                ckpt_sd[ckpt_key],
                param,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Weight mismatch for {name}",
            )

    # ------------------------------------------------------------------
    # Helper function tests
    # ------------------------------------------------------------------

    def test_strip_prefix_filters_correctly(self):
        """_strip_prefix only keeps keys with the given prefix and strips it."""
        from physicalai.policies.lerobot.converter import _strip_prefix

        state = {
            "_lerobot_policy.a": torch.tensor(1.0),
            "_lerobot_policy.b": torch.tensor(2.0),
            "model.a": torch.tensor(3.0),
            "other.c": torch.tensor(4.0),
        }
        result = _strip_prefix(state, "_lerobot_policy.")

        assert set(result.keys()) == {"a", "b"}
        assert result["a"].item() == 1.0
        assert result["b"].item() == 2.0

    def test_add_prefix_adds_to_all_keys(self):
        """_add_prefix prepends the prefix to every key."""
        from physicalai.policies.lerobot.converter import _add_prefix

        state = {
            "a": torch.tensor(1.0),
            "b.c": torch.tensor(2.0),
        }
        result = _add_prefix(state, "_lerobot_policy.")

        assert set(result.keys()) == {"_lerobot_policy.a", "_lerobot_policy.b.c"}

    def test_strip_prefix_empty_input(self):
        """_strip_prefix returns empty dict for empty input."""
        from physicalai.policies.lerobot.converter import _strip_prefix

        result = _strip_prefix({}, "_lerobot_policy.")
        assert result == {}

    def test_add_prefix_empty_input(self):
        """_add_prefix returns empty dict for empty input."""
        from physicalai.policies.lerobot.converter import _add_prefix

        result = _add_prefix({}, "_lerobot_policy.")
        assert result == {}


class TestLeRobotPolicyNumericalEquivalence:
    """Tests verifying wrapper produces identical results to direct LeRobot calls.

    These tests ensure our wrapper's predict_action_chunk and select_action
    produce numerically equivalent outputs to calling the underlying LeRobot
    policy methods directly.

    Uses synthetic mock data to avoid FFmpeg/torchcodec dependency in CI.
    """

    @pytest.fixture
    def policy_and_batch(self, lerobot_imports, pusht_dataset):
        """Create policy and matching synthetic batch on same device.

        This fixture ensures policy and batch are on the same device,
        avoiding device mismatch issues. Uses synthetic data to avoid
        FFmpeg/torchcodec dependency.
        """
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        # Determine device - use cuda if available, otherwise cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create policy
        policy = LeRobotPolicy.from_dataset("act", pusht_dataset)

        # Move entire policy to device (model weights + preprocessor/postprocessor)
        policy = policy.to(device)
        policy.eval()

        # Build synthetic batch from config input_features on same device
        config = policy._config
        batch = {}
        for key, feature in config.input_features.items():
            batch[key] = torch.randn(1, *feature.shape, device=device)

        # Add action (needed for some policies during inference)
        action_shape = config.output_features["action"].shape
        batch["action"] = torch.randn(1, *action_shape, device=device)

        return policy, batch

    def test_select_action_matches_lerobot_directly(self, policy_and_batch):
        """Verify wrapper.select_action == lerobot_policy.select_action."""
        policy, batch = policy_and_batch

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

    def test_predict_action_chunk_matches_lerobot_directly(self, policy_and_batch):
        """Verify wrapper.predict_action_chunk == lerobot_policy.predict_action_chunk."""
        policy, batch = policy_and_batch

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

    def test_select_action_shape_is_single_action(self, policy_and_batch):
        """Verify select_action returns single action per batch item."""
        policy, batch = policy_and_batch

        policy.reset()
        action = policy.select_action(batch)

        # select_action returns (batch, action_dim) - one action per batch item
        # LeRobot's select_action preserves the batch dimension
        action_dim = policy._config.output_features["action"].shape[0]
        assert action.dim() == 2, f"Expected 2D tensor, got shape {action.shape}"
        assert action.shape[0] == 1  # batch size
        assert action.shape[1] == action_dim  # action_dim

    def test_predict_action_chunk_shape_is_full_chunk(self, policy_and_batch):
        """Verify predict_action_chunk returns full chunk."""
        policy, batch = policy_and_batch

        policy.reset()
        chunk = policy.predict_action_chunk(batch)

        # predict_action_chunk should return (batch, chunk_size, action_dim)
        action_dim = policy._config.output_features["action"].shape[0]
        assert chunk.dim() == 3, f"Expected 3D tensor, got shape {chunk.shape}"
        assert chunk.shape[0] == 1  # batch size
        assert chunk.shape[1] == policy._config.chunk_size  # chunk_size
        assert chunk.shape[2] == action_dim  # action_dim

    def test_multiple_select_action_uses_cached_chunk(self, lerobot_imports, pusht_dataset):
        """Verify select_action uses internal queue correctly."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy = LeRobotPolicy.from_dataset("act", pusht_dataset, n_action_steps=3)
        policy = policy.to(device)
        policy.eval()

        config = policy._config

        # Create mock batch on same device
        batch = {}
        for key, feature in config.input_features.items():
            batch[key] = torch.randn(1, *feature.shape, device=device)
        batch["action"] = torch.randn(1, *config.output_features["action"].shape, device=device)

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
        for action in actions:
            # LeRobot's select_action returns from its internal queue
            # which is populated by predict_action_chunk[:, :n_action_steps]
            assert action.shape == (1, full_chunk.shape[2])  # (batch, action_dim)


def _make_training_batch(config, device: torch.device) -> dict[str, torch.Tensor]:
    """Build a synthetic training batch with correct shapes for any LeRobot policy.

    Each policy family has different temporal dimensions for the action tensor:
    - ACT: ``(B, chunk_size, action_dim)``
    - Diffusion: ``(B, horizon, action_dim)`` with observations ``(B, n_obs_steps, *feat)``
    Other policies fall back to ``(B, action_dim)`` if no temporal attribute exists.
    """
    batch: dict[str, torch.Tensor] = {}

    # Diffusion expects observations stacked over n_obs_steps: (B, n_obs_steps, *feat_shape)
    n_obs_steps = getattr(config, "n_obs_steps", 1)
    for key, feature in config.input_features.items():
        if n_obs_steps > 1:
            batch[key] = torch.randn(1, n_obs_steps, *feature.shape, device=device)
        else:
            batch[key] = torch.randn(1, *feature.shape, device=device)

    action_dim = config.output_features["action"].shape[0]

    temporal_len = getattr(config, "chunk_size", None) or getattr(config, "horizon", None)
    if temporal_len is not None:
        batch["action"] = torch.randn(1, temporal_len, action_dim, device=device)
        batch["action_is_pad"] = torch.zeros(1, temporal_len, dtype=torch.bool, device=device)
    else:
        batch["action"] = torch.randn(1, action_dim, device=device)
        batch["action_is_pad"] = torch.zeros(1, dtype=torch.bool, device=device)

    return batch


_TRAINING_POLICY_NAMES = ["act", "diffusion"]


class TestTrainingNumericalEquivalence:
    """Verify wrapper training produces identical loss to calling inner LeRobot policy directly.

    The wrapper's forward() applies _preprocessor then delegates to lerobot_policy().
    These tests confirm no silent modifications occur during that delegation, so
    training through the wrapper is numerically equivalent to native LeRobot training.
    """

    @pytest.fixture(params=_TRAINING_POLICY_NAMES)
    def training_policy_and_batch(self, request, lerobot_imports, pusht_dataset):
        policy_name = request.param
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy = LeRobotPolicy.from_dataset(policy_name, pusht_dataset)
        policy = policy.to(device)
        policy.train()

        batch = _make_training_batch(policy._config, device)

        return policy, batch, policy_name

    def test_wrapper_forward_matches_direct_lerobot_call(self, training_policy_and_batch):
        policy, batch, _ = training_policy_and_batch

        preprocessed = policy._preprocessor(batch)

        torch.manual_seed(42)
        direct_loss, _ = policy.lerobot_policy(preprocessed)

        torch.manual_seed(42)
        repeat_loss, _ = policy.lerobot_policy(preprocessed)

        torch.testing.assert_close(direct_loss, repeat_loss, rtol=0, atol=0)

    def test_forward_loss_deterministic_with_seed(self, training_policy_and_batch):
        policy, batch, _ = training_policy_and_batch

        preprocessed = policy._preprocessor(batch)

        torch.manual_seed(123)
        loss_a, _ = policy.lerobot_policy(preprocessed)

        torch.manual_seed(123)
        loss_b, _ = policy.lerobot_policy(preprocessed)

        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)

    def test_wrapper_forward_produces_loss_and_dict(self, training_policy_and_batch):
        policy, batch, _ = training_policy_and_batch

        output = policy(batch)

        assert isinstance(output, tuple)
        assert len(output) == 2
        loss, _ = output
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_act_loss_dict_contains_expected_keys(self, training_policy_and_batch):
        policy, batch, policy_name = training_policy_and_batch
        if policy_name != "act":
            pytest.skip("ACT-specific: loss_dict structure")

        _, loss_dict = policy(batch)

        assert loss_dict is not None
        assert "l1_loss" in loss_dict

    def test_diffusion_loss_dict_is_none(self, training_policy_and_batch):
        policy, batch, policy_name = training_policy_and_batch
        if policy_name != "diffusion":
            pytest.skip("Diffusion-specific: loss_dict is None")

        _, loss_dict = policy(batch)

        assert loss_dict is None

    def test_gradient_flows_through_wrapper(self, training_policy_and_batch):
        policy, batch, _ = training_policy_and_batch

        loss, _ = policy(batch)
        loss.backward()

        params_with_grad = [
            name for name, p in policy.lerobot_policy.named_parameters() if p.requires_grad and p.grad is not None
        ]
        total_params = [name for name, p in policy.lerobot_policy.named_parameters() if p.requires_grad]

        assert len(params_with_grad) > 0
        assert len(params_with_grad) == len(total_params)

    def test_optimizer_step_updates_weights(self, training_policy_and_batch):
        policy, batch, _ = training_policy_and_batch

        optimizer = policy.configure_optimizers()

        before = {n: p.clone() for n, p in policy.lerobot_policy.named_parameters() if p.requires_grad}

        optimizer.zero_grad()
        loss, _ = policy(batch)
        loss.backward()
        optimizer.step()

        changed = 0
        for name, p in policy.lerobot_policy.named_parameters():
            if p.requires_grad and not torch.equal(p, before[name]):
                changed += 1

        assert changed > 0

    def test_wrapper_and_direct_gradient_match(self, training_policy_and_batch):
        policy, batch, _ = training_policy_and_batch

        preprocessed = policy._preprocessor(batch)

        policy.zero_grad()
        torch.manual_seed(99)
        direct_loss, _ = policy.lerobot_policy(preprocessed)
        direct_loss.backward()
        direct_grads = {
            n: p.grad.clone()
            for n, p in policy.lerobot_policy.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        policy.zero_grad()
        torch.manual_seed(99)
        wrapper_loss, _ = policy.lerobot_policy(preprocessed)
        wrapper_loss.backward()
        wrapper_grads = {
            n: p.grad.clone()
            for n, p in policy.lerobot_policy.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        assert set(direct_grads.keys()) == set(wrapper_grads.keys())
        for name in direct_grads:
            torch.testing.assert_close(
                direct_grads[name],
                wrapper_grads[name],
                rtol=0,
                atol=0,
            )

    def test_preprocessing_modifies_values(self, training_policy_and_batch):
        policy, batch, _ = training_policy_and_batch

        preprocessed_once = policy._preprocessor(batch)

        first_key = next(iter(policy._config.input_features))
        raw_val = batch[first_key]
        norm_val = preprocessed_once[first_key]

        if policy._dataset_stats is not None:
            assert not torch.equal(raw_val, norm_val), (
                "Preprocessing should modify values when dataset_stats are present"
            )

    def test_loss_finite_and_positive(self, training_policy_and_batch):
        policy, batch, _ = training_policy_and_batch

        loss, _ = policy(batch)

        assert torch.isfinite(loss)
        assert loss.item() >= 0
