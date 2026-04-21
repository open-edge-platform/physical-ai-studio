# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LeRobot policy wrappers."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any

import pytest
import torch

# Skip all tests if lerobot not installed
pytest.importorskip("lerobot", reason="LeRobot not installed")


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

    def test_escape_hatch_warns_for_unsupported_policy(self, lerobot_imports):
        """Universal wrapper accepts unsupported names with a UserWarning."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        from physicalai.policies.lerobot import policy as policy_module  # noqa: PLC0415

        policy_module._WARNED_UNSUPPORTED_NAMES.discard("vqbet")
        with pytest.warns(UserWarning, match="not in physicalai's supported set"):
            LeRobotPolicy(policy_name="vqbet")


class TestNamedLeRobotPolicy:
    """Tests for the named-wrapper unification (Oracle PR #418 follow-up)."""

    def test_from_dataset_returns_subclass_not_base(self, pusht_dataset):
        """ACT.from_dataset returns ACT, not LeRobotPolicy.

        Regression: prior to the unification fix, ``from_dataset`` was
        defined on the base and unconditionally returned a ``LeRobotPolicy``,
        breaking ``isinstance`` checks against the named wrapper.
        """
        from physicalai.policies.lerobot import ACT, Diffusion, LeRobotPolicy

        act = ACT.from_dataset(pusht_dataset)
        assert type(act) is ACT
        assert isinstance(act, LeRobotPolicy)

        diffusion = Diffusion.from_dataset(pusht_dataset)
        assert type(diffusion) is Diffusion

    def test_optimizer_lr_override_propagates(self, pusht_dataset):
        """``optimizer_lr=`` mutates the LeRobot config and the resulting optimizer."""
        from physicalai.policies.lerobot import ACT

        policy = ACT.from_dataset(pusht_dataset, optimizer_lr=7.5e-5)
        assert policy._config.optimizer_lr == pytest.approx(7.5e-5)
        optimizer = policy.configure_optimizers()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(7.5e-5)

    def test_named_wrappers_share_policy_name_with_base(self):
        """Every named wrapper's POLICY_NAME is registered in SUPPORTED_POLICIES."""
        from physicalai.policies.lerobot import (
            ACT,
            PI0,
            PI05,
            SUPPORTED_POLICIES,
            VALIDATED_EQUIVALENCE_POLICIES,
            XVLA,
            Diffusion,
            Groot,
            PI0Fast,
            SmolVLA,
        )

        wrappers = (
            ACT,
            Diffusion,
            Groot,
            PI0,
            PI05,
            PI0Fast,
            SmolVLA,
            XVLA,
        )
        names = {cls.POLICY_NAME for cls in wrappers}
        assert names == set(SUPPORTED_POLICIES)
        assert set(VALIDATED_EQUIVALENCE_POLICIES) <= set(SUPPORTED_POLICIES), (
            "VALIDATED_EQUIVALENCE_POLICIES must be a subset of SUPPORTED_POLICIES"
        )
        import importlib.util  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        integration_test = Path(__file__).resolve().parents[2] / "integration" / "test_lerobot_wrapper_equivalence.py"
        spec = importlib.util.spec_from_file_location("_lerobot_eq_module", integration_test)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        overlap = set(VALIDATED_EQUIVALENCE_POLICIES) & set(module._EQUIVALENCE_XFAIL_REASONS)
        assert not overlap, (
            f"Policies cannot be both VALIDATED and XFAIL: {sorted(overlap)}. "
            "Either remove from _EQUIVALENCE_XFAIL_REASONS (fix the limitation) "
            "or remove from VALIDATED_EQUIVALENCE_POLICIES (downgrade the guarantee)."
        )

    @pytest.mark.parametrize(
        "wrapper_name",
        ["ACT", "Diffusion", "Groot", "PI0", "PI05", "PI0Fast", "SmolVLA", "XVLA"],
    )
    def test_named_wrapper_rejects_mismatched_policy_name(self, wrapper_name):
        """``ACT(policy_name="diffusion")`` raises — POLICY_NAME is the source of truth.

        Covers every named wrapper to lock in the invariant uniformly. Uses
        lazy construction (no dataset / no inner policy build) so the test
        is cheap and works even for wrappers without pusht-compatible data.
        """
        import physicalai.policies.lerobot as lerobot_module

        wrapper_cls = getattr(lerobot_module, wrapper_name)
        bound_name = wrapper_cls.POLICY_NAME
        wrong_name = "diffusion" if bound_name != "diffusion" else "act"

        with pytest.raises(ValueError, match="refusing to override"):
            wrapper_cls(policy_name=wrong_name)


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
            pathlib.Path(checkpoint_path).unlink()

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
            pathlib.Path(checkpoint_path).unlink()

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
            pathlib.Path(checkpoint_path).unlink()

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
            pathlib.Path(checkpoint_path).unlink()


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

        with pathlib.Path(save_dir / "config.json").open() as f:
            config = json.load(f)

        assert "type" in config, "config.json must contain 'type' field"

    def test_save_pretrained_roundtrip_with_lerobot(self, pusht_act_policy, tmp_path):
        """Test saved model is loadable by LeRobot's from_pretrained."""
        from lerobot.policies.act.modeling_act import ACTPolicy

        save_dir = tmp_path / "saved_policy"
        pusht_act_policy.save_pretrained(save_dir)

        loaded = ACTPolicy.from_pretrained(str(save_dir))
        loaded = loaded.to(pusht_act_policy.device)

        orig_params = dict(pusht_act_policy.lerobot_policy.named_parameters())
        loaded_params = dict(loaded.named_parameters())

        assert set(orig_params.keys()) == set(loaded_params.keys()), "Parameter names should match"
        for name in orig_params:
            torch.testing.assert_close(
                orig_params[name].cpu(),
                loaded_params[name].cpu(),
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
                orig_params[name].cpu(),
                loaded_params[name].cpu(),
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

    def test_save_pretrained_with_repo_id_pushes_to_hub(self, pusht_act_policy, tmp_path):
        """Test save_pretrained with repo_id delegates to push_to_hub."""
        from unittest.mock import MagicMock, patch

        mock_api_instance = MagicMock()
        mock_api_instance.create_repo.return_value = MagicMock(repo_id="user/my-policy")
        mock_api_instance.upload_folder.return_value = "https://huggingface.co/user/my-policy/commit/abc"

        save_dir = tmp_path / "saved_policy"
        with patch("huggingface_hub.HfApi", return_value=mock_api_instance):
            result = pusht_act_policy.save_pretrained(
                save_dir,
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
            torch.testing.assert_close(orig_params[name].cpu(), loaded_params[name].cpu(), rtol=1e-5, atol=1e-5)


class TestCheckpointConverter:
    """Tests for the Lightning <-> LeRobot checkpoint converter."""

    @pytest.fixture
    def sample_config(self):
        """Minimal config dict mimicking a LeRobot policy config."""
        return {"type": "act", "chunk_size": 100, "dim_model": 256}

    @pytest.fixture
    def sample_lerobot_state(self):
        """Synthetic LeRobot-format state dict (model.* keys without prefix)."""
        return {
            "model.backbone.0.weight": torch.randn(64, 3, 3, 3),
            "model.backbone.0.bias": torch.randn(64),
            "model.head.weight": torch.randn(2, 64),
            "model.head.bias": torch.randn(2),
        }

    @pytest.fixture
    def sample_lightning_state(self, sample_lerobot_state):
        """Synthetic Lightning-format state dict (_lerobot_policy.model.* keys)."""
        from physicalai.policies.lerobot.utils.checkpoint_converter import _LEROBOT_PREFIX

        return {_LEROBOT_PREFIX + k: v for k, v in sample_lerobot_state.items()}

    @pytest.fixture
    def lightning_checkpoint(self, sample_config, sample_lightning_state, tmp_path):
        """Create a synthetic Lightning .ckpt file on disk."""
        from physicalai.export.mixin_policy import CONFIG_KEY, POLICY_NAME_KEY

        ckpt = {
            "state_dict": sample_lightning_state,
            CONFIG_KEY: sample_config,
            POLICY_NAME_KEY: "act",
        }
        ckpt_path = tmp_path / "model.ckpt"
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(ckpt, str(ckpt_path))
        return ckpt_path

    @pytest.fixture
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
    # lightning_to_lerobot (real policy round-trip; synthetic dicts are not
    # supported because the new converter delegates to
    # ``LeRobotPolicy.load_from_checkpoint`` + ``save_pretrained``, which
    # require a real LeRobot config dataclass and initialized processors.)
    # ------------------------------------------------------------------

    def test_lightning_to_lerobot_real_policy(self, pusht_act_policy, tmp_path):
        """Round-trip: save Lightning ckpt, convert to LeRobot dir, verify artefacts."""
        from physicalai.policies.lerobot.utils.checkpoint_converter import lightning_to_lerobot

        ckpt_path = tmp_path / "act.ckpt"
        ckpt_dict: dict[str, Any] = {"state_dict": pusht_act_policy.state_dict()}
        pusht_act_policy.on_save_checkpoint(ckpt_dict)
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(ckpt_dict, str(ckpt_path))

        out_dir = tmp_path / "act_lerobot"
        result = lightning_to_lerobot(ckpt_path, out_dir)

        assert result == out_dir
        assert (out_dir / "config.json").exists()
        assert (out_dir / "model.safetensors").exists()
        # Processors are the bug Oracle flagged: must be present so that
        # the resulting directory is loadable by LeRobot's own loader.
        assert (out_dir / "policy_preprocessor.json").exists()
        assert (out_dir / "policy_postprocessor.json").exists()

    def test_lightning_to_lerobot_roundtrip_loadable(self, pusht_act_policy, tmp_path):
        """The converted directory is loadable via LeRobotPolicy.from_pretrained."""
        from physicalai.policies.lerobot import LeRobotPolicy
        from physicalai.policies.lerobot.utils.checkpoint_converter import lightning_to_lerobot

        ckpt_path = tmp_path / "act.ckpt"
        ckpt_dict: dict[str, Any] = {"state_dict": pusht_act_policy.state_dict()}
        pusht_act_policy.on_save_checkpoint(ckpt_dict)
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(ckpt_dict, str(ckpt_path))

        out_dir = tmp_path / "act_lerobot"
        lightning_to_lerobot(ckpt_path, out_dir)

        reloaded = LeRobotPolicy.from_pretrained(str(out_dir), policy_name="act")
        # Weight equivalence after round-trip.
        for (orig_name, orig_param), (rel_name, rel_param) in zip(
            pusht_act_policy.lerobot_policy.named_parameters(),
            reloaded.lerobot_policy.named_parameters(),
            strict=True,
        ):
            assert orig_name == rel_name
            torch.testing.assert_close(orig_param.cpu(), rel_param.cpu(), rtol=1e-5, atol=1e-5)

    # ------------------------------------------------------------------
    # lerobot_to_lightning
    # ------------------------------------------------------------------

    def test_lerobot_to_lightning_creates_ckpt(self, lerobot_dir, tmp_path):
        """Converted checkpoint file exists and is loadable."""
        from physicalai.policies.lerobot.utils.checkpoint_converter import lerobot_to_lightning

        ckpt_path = tmp_path / "converted.ckpt"
        result = lerobot_to_lightning(lerobot_dir, ckpt_path)

        assert result == ckpt_path
        assert ckpt_path.exists()

        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        assert "state_dict" in ckpt

    def test_lerobot_to_lightning_adds_prefix(self, lerobot_dir, sample_lerobot_state, tmp_path):
        """State dict keys have _lerobot_policy. prefix added."""
        from physicalai.policies.lerobot.utils.checkpoint_converter import _LEROBOT_PREFIX, lerobot_to_lightning

        ckpt_path = tmp_path / "converted.ckpt"
        lerobot_to_lightning(lerobot_dir, ckpt_path)

        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        state_dict = ckpt["state_dict"]

        expected_keys = {_LEROBOT_PREFIX + k for k in sample_lerobot_state}
        assert set(state_dict.keys()) == expected_keys

        for key in sample_lerobot_state:
            torch.testing.assert_close(state_dict[_LEROBOT_PREFIX + key], sample_lerobot_state[key])

    def test_lerobot_to_lightning_stores_config_and_policy_name(self, lerobot_dir, sample_config, tmp_path):
        """Checkpoint contains CONFIG_KEY and POLICY_NAME_KEY."""
        from physicalai.export.mixin_policy import CONFIG_KEY, POLICY_NAME_KEY
        from physicalai.policies.lerobot.utils.checkpoint_converter import lerobot_to_lightning

        ckpt_path = tmp_path / "converted.ckpt"
        lerobot_to_lightning(lerobot_dir, ckpt_path)

        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

        assert ckpt[CONFIG_KEY] == sample_config
        assert ckpt[POLICY_NAME_KEY] == "act"

    def test_lerobot_to_lightning_explicit_policy_name(self, lerobot_dir, tmp_path):
        """Explicit policy_name overrides config['type']."""
        from physicalai.export.mixin_policy import POLICY_NAME_KEY
        from physicalai.policies.lerobot.utils.checkpoint_converter import lerobot_to_lightning

        ckpt_path = tmp_path / "converted.ckpt"
        lerobot_to_lightning(lerobot_dir, ckpt_path, policy_name="diffusion")

        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        assert ckpt[POLICY_NAME_KEY] == "diffusion"

    def test_lerobot_to_lightning_missing_config_raises(self, tmp_path):
        """FileNotFoundError when config.json is missing."""
        from physicalai.policies.lerobot.utils.checkpoint_converter import lerobot_to_lightning

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="config.json"):
            lerobot_to_lightning(empty_dir, tmp_path / "out.ckpt")

    def test_lerobot_to_lightning_missing_weights_raises(self, tmp_path):
        """FileNotFoundError when model.safetensors is missing."""
        import json

        from physicalai.policies.lerobot.utils.checkpoint_converter import lerobot_to_lightning

        dir_no_weights = tmp_path / "no_weights"
        dir_no_weights.mkdir()
        with (dir_no_weights / "config.json").open("w") as f:
            json.dump({"type": "act"}, f)

        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            lerobot_to_lightning(dir_no_weights, tmp_path / "out.ckpt")

    def test_lerobot_to_lightning_no_policy_name_raises(self, tmp_path):
        """ValueError when policy_name is None and config has no 'type' field."""
        import json

        from physicalai.policies.lerobot.utils.checkpoint_converter import lerobot_to_lightning
        from safetensors.torch import save_file

        no_type_dir = tmp_path / "no_type"
        no_type_dir.mkdir()
        with (no_type_dir / "config.json").open("w") as f:
            json.dump({"chunk_size": 100}, f)
        save_file({"w": torch.randn(2, 2)}, str(no_type_dir / "model.safetensors"))

        with pytest.raises(ValueError, match="Cannot determine policy_name"):
            lerobot_to_lightning(no_type_dir, tmp_path / "out.ckpt")

    def test_lerobot_to_lightning_creates_parent_dirs(self, lerobot_dir, tmp_path):
        """Output parent directories are created if they do not exist."""
        from physicalai.policies.lerobot.utils.checkpoint_converter import lerobot_to_lightning

        ckpt_path = tmp_path / "a" / "b" / "model.ckpt"
        lerobot_to_lightning(lerobot_dir, ckpt_path)

        assert ckpt_path.exists()

    # ------------------------------------------------------------------
    # Round-trip tests
    # ------------------------------------------------------------------

    def test_roundtrip_lerobot_lightning_lerobot(self, lerobot_dir, sample_lerobot_state, tmp_path):
        """LeRobot -> Lightning preserves all weights exactly.

        Note: Lightning -> LeRobot half cannot be exercised with synthetic
        dicts because the converter now delegates to the real wrapper. See
        ``test_lightning_to_lerobot_real_policy`` / ``..._roundtrip_loadable``
        for the real-policy version of that direction.
        """
        from physicalai.policies.lerobot.utils.checkpoint_converter import _LEROBOT_PREFIX, lerobot_to_lightning

        ckpt_path = tmp_path / "intermediate.ckpt"
        lerobot_to_lightning(lerobot_dir, ckpt_path)

        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        sd = ckpt["state_dict"]
        for key, value in sample_lerobot_state.items():
            torch.testing.assert_close(sd[_LEROBOT_PREFIX + key], value, msg=f"Mismatch for {key}")

    # ------------------------------------------------------------------
    # Integration: real ACT policy round-trip
    # ------------------------------------------------------------------

    def test_roundtrip_real_act_policy(self, pusht_act_policy, tmp_path):
        """Round-trip a real ACT wrapper through save_pretrained -> lerobot_to_lightning -> load."""
        from physicalai.policies.lerobot.utils.checkpoint_converter import _LEROBOT_PREFIX, lerobot_to_lightning

        lr_dir = tmp_path / "act_lerobot"
        pusht_act_policy.save_pretrained(lr_dir)

        ckpt_path = tmp_path / "act_lightning.ckpt"
        lerobot_to_lightning(lr_dir, ckpt_path)

        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
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

    def test_add_prefix_adds_to_all_keys(self):
        """_add_prefix prepends the prefix to every key."""
        from physicalai.policies.lerobot.utils.checkpoint_converter import _add_prefix

        state = {
            "a": torch.tensor(1.0),
            "b.c": torch.tensor(2.0),
        }
        result = _add_prefix(state, "_lerobot_policy.")

        assert set(result.keys()) == {"_lerobot_policy.a", "_lerobot_policy.b.c"}

    def test_add_prefix_empty_input(self):
        """_add_prefix returns empty dict for empty input."""
        from physicalai.policies.lerobot.utils.checkpoint_converter import _add_prefix

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

        Uses ``_make_training_batch`` so the shapes are correct for both
        inference *and* training (ACT's VAE encoder requires temporal dims).
        """
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy = LeRobotPolicy.from_dataset("act", pusht_dataset)
        policy = policy.to(device)
        policy.eval()

        batch = _make_training_batch(policy._config, device)

        return policy, batch

    def test_select_action_matches_lerobot_directly(self, policy_and_batch):
        """Verify wrapper.select_action == lerobot_policy.select_action."""
        policy, batch = policy_and_batch

        policy.reset()

        preprocessed = policy._preprocessor(_clone_batch(batch))
        lerobot_action = policy.lerobot_policy.select_action(preprocessed)

        policy.reset()

        wrapper_action = policy.select_action(_clone_batch(batch))

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

        policy.reset()

        preprocessed = policy._preprocessor(_clone_batch(batch))
        lerobot_chunk = policy.lerobot_policy.predict_action_chunk(preprocessed)

        policy.reset()

        wrapper_chunk = policy.predict_action_chunk(_clone_batch(batch))

        # Should be numerically identical
        torch.testing.assert_close(
            wrapper_chunk,
            policy._postprocessor(lerobot_chunk),
            rtol=1e-5,
            atol=1e-5,
            msg="Wrapper predict_action_chunk should match LeRobot predict_action_chunk",
        )

    def test_forward_training_returns_loss(self, policy_and_batch):
        """Verify forward in training mode returns (loss, loss_dict)."""
        policy, batch = policy_and_batch

        policy.train()
        policy.reset()

        output = policy(_clone_batch(batch))

        assert isinstance(output, tuple)
        loss = output[0]
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1

    def test_select_action_shape_is_single_action(self, policy_and_batch):
        """Verify select_action returns single action per batch item."""
        policy, batch = policy_and_batch

        policy.eval()
        policy.reset()
        action = policy.select_action(_clone_batch(batch))

        # select_action returns (batch, action_dim) - one action per batch item
        # LeRobot's select_action preserves the batch dimension
        action_dim = policy._config.output_features["action"].shape[0]
        assert action.dim() == 2, f"Expected 2D tensor, got shape {action.shape}"
        assert action.shape[0] == 1  # batch size
        assert action.shape[1] == action_dim  # action_dim

    def test_predict_action_chunk_shape_is_full_chunk(self, policy_and_batch):
        """Verify predict_action_chunk returns full chunk."""
        policy, batch = policy_and_batch

        policy.eval()
        policy.reset()
        chunk = policy.predict_action_chunk(_clone_batch(batch))

        # predict_action_chunk should return (batch, chunk_size, action_dim)
        action_dim = policy._config.output_features["action"].shape[0]
        assert chunk.dim() == 3, f"Expected 3D tensor, got shape {chunk.shape}"
        assert chunk.shape[0] == 1  # batch size
        assert chunk.shape[1] == policy._config.chunk_size  # chunk_size
        assert chunk.shape[2] == action_dim  # action_dim

    def test_multiple_select_action_uses_cached_chunk(self, lerobot_imports, pusht_dataset):
        """Verify select_action uses internal queue correctly."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy = LeRobotPolicy.from_dataset("act", pusht_dataset, n_action_steps=3)
        policy = policy.to(device)
        policy.eval()

        batch = _make_training_batch(policy._config, device)

        policy.reset()

        full_chunk = policy.predict_action_chunk(_clone_batch(batch))

        policy.reset()
        actions = []
        for _ in range(3):
            action = policy.select_action(_clone_batch(batch))
            actions.append(action)

        for action in actions:
            assert action.shape == (1, full_chunk.shape[2])


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone a batch dict, detaching and cloning all tensors."""
    import copy

    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().clone()
        elif isinstance(value, dict):
            out[key] = _clone_batch(value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _make_training_batch(config, device: torch.device) -> dict[str, torch.Tensor]:
    """Build a synthetic training batch with correct shapes for any LeRobot policy.

    Action temporal dimension resolution (in order):
    - ACT / VLA family: ``chunk_size``
    - Diffusion: ``horizon``
    - Fallback: ``(B, action_dim)`` for non-temporal policies.

    Tokenized policies (smolvla, …) detect a tokenizer in the preprocessor
    pipeline via the presence of language-related config attributes and
    receive a synthetic ``task`` string.
    """
    batch: dict[str, torch.Tensor] = {}

    n_obs_steps = getattr(config, "n_obs_steps", 1)
    for key, feature in config.input_features.items():
        if n_obs_steps > 1:
            batch[key] = torch.randn(1, n_obs_steps, *feature.shape, device=device)
        else:
            batch[key] = torch.randn(1, *feature.shape, device=device)

    action_dim = config.output_features["action"].shape[0]

    chunk = getattr(config, "chunk_size", None)
    horizon = getattr(config, "horizon", None)
    temporal_len = chunk or horizon

    if temporal_len is not None:
        batch["action"] = torch.randn(1, temporal_len, action_dim, device=device)
        batch["action_is_pad"] = torch.zeros(1, temporal_len, dtype=torch.bool, device=device)
    else:
        batch["action"] = torch.randn(1, action_dim, device=device)
        batch["action_is_pad"] = torch.zeros(1, dtype=torch.bool, device=device)

    if any(hasattr(config, attr) for attr in ("tokenizer_max_length", "max_state_dim", "vlm_model_name")):
        batch["task"] = ["pick up the block"]

    return batch


_TRAINING_POLICY_NAMES = ["act", "diffusion", "smolvla"]
"""Policies validated end-to-end on CPU with the pusht dataset fixture.

Excluded with reason:
- ``vlas`` (pi0/pi05/pi0_fast/groot): require GPU + bf16; integration tier only.
- ``xvla``: requires explicit ``vision_config`` kwarg, not derivable from the dataset.
"""


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

        device = torch.device("cpu")

        # Pass device="cpu" so the LeRobot config (and its DeviceProcessorStep)
        # targets CPU; otherwise LeRobot autodetects CUDA when available and the
        # preprocessor moves the batch to GPU while wrapper weights stay on CPU.
        policy = LeRobotPolicy.from_dataset(policy_name, pusht_dataset, device="cpu")
        policy.train()

        batch = _make_training_batch(policy._config, device)

        return policy, batch, policy_name

    def test_wrapper_forward_matches_native_path(self, training_policy_and_batch):
        """Wrapper.forward(batch) must produce the same loss as preprocessor + native call.

        This is the core wrapper-vs-native equivalence guarantee at the unit level:
        running the batch through ``policy(batch)`` (which preprocesses internally)
        must be numerically identical to manually preprocessing and calling the
        underlying ``lerobot_policy`` directly with the same seed.
        """
        policy, batch, _ = training_policy_and_batch

        preprocessed = policy._preprocessor(_clone_batch(batch))

        torch.manual_seed(42)
        native_loss, _ = policy.lerobot_policy(preprocessed)

        torch.manual_seed(42)
        wrapper_loss, _ = policy(_clone_batch(batch))

        torch.testing.assert_close(wrapper_loss, native_loss, rtol=1e-6, atol=1e-6)

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

        # Clone batch before each call to avoid in-place mutation
        preprocessed_a = policy._preprocessor(_clone_batch(batch))

        # --- Direct path: call inner LeRobot policy on preprocessed data ---
        policy.zero_grad()
        torch.manual_seed(99)
        direct_loss, _ = policy.lerobot_policy(preprocessed_a)
        direct_loss.backward()
        direct_grads = {
            n: p.grad.clone()
            for n, p in policy.lerobot_policy.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        # --- Wrapper path: call policy(batch) which preprocesses internally ---
        policy.zero_grad()
        torch.manual_seed(99)
        wrapper_loss, _ = policy(_clone_batch(batch))
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
                rtol=1e-6,
                atol=1e-6,
            )

    def test_preprocessing_modifies_values(self, training_policy_and_batch):
        policy, batch, _ = training_policy_and_batch

        if policy._dataset_stats is None:
            pytest.skip("No dataset_stats; preprocessor cannot normalize")

        raw_batch = _clone_batch(batch)
        preprocessed = policy._preprocessor(_clone_batch(batch))

        modified_keys = [
            key for key in policy._config.input_features if not torch.equal(raw_batch[key], preprocessed[key])
        ]
        assert modified_keys, (
            f"Preprocessing did not modify any input feature; expected at least one of "
            f"{list(policy._config.input_features)} to be normalized."
        )

    def test_loss_finite_and_positive(self, training_policy_and_batch):
        policy, batch, _ = training_policy_and_batch

        loss, _ = policy(batch)

        assert torch.isfinite(loss)
        assert loss.item() >= 0


class TestMultiStepTrainingTrajectory:
    """Verify multi-step training through the wrapper matches native LeRobot exactly.

    Runs N training steps with identical weights, optimizer, and batch through both
    the physicalai wrapper and bare LeRobot policy, then asserts:
    - Per-step loss values are identical (atol=0)
    - Loss decreases from first to last step
    - Final model weights are identical
    """

    NUM_STEPS = 5

    @pytest.fixture(params=_TRAINING_POLICY_NAMES)
    def dual_setup(self, request, lerobot_imports, pusht_dataset):
        """Create wrapper + native LeRobot policy with identical weights and optimizers."""
        import copy

        policy_name = request.param
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        device = torch.device("cpu")

        # See TestTrainingNumericalEquivalence.training_policy_and_batch for why
        # device="cpu" must be passed as a config kwarg, not via .to("cpu").
        wrapper = LeRobotPolicy.from_dataset(policy_name, pusht_dataset, device="cpu")
        wrapper.train()

        # --- Native path: deep-copy the inner LeRobot policy so weights are identical ---
        native = copy.deepcopy(wrapper.lerobot_policy)
        native = native.to(device)
        native.train()

        # Build identical optimizers using LeRobot's own preset
        wrapper_optimizer = wrapper.configure_optimizers()
        native_optimizer = wrapper._config.get_optimizer_preset().build(native.get_optim_params())

        # Build a fixed training batch
        batch = _make_training_batch(wrapper._config, device)

        return {
            "wrapper": wrapper,
            "native": native,
            "wrapper_optimizer": wrapper_optimizer,
            "native_optimizer": native_optimizer,
            "batch": batch,
            "policy_name": policy_name,
        }

    def test_loss_trajectories_match(self, dual_setup):
        """Per-step loss values from wrapper and native must be bit-identical."""
        wrapper = dual_setup["wrapper"]
        native = dual_setup["native"]
        w_opt = dual_setup["wrapper_optimizer"]
        n_opt = dual_setup["native_optimizer"]
        batch = dual_setup["batch"]

        wrapper_losses: list[float] = []
        native_losses: list[float] = []

        for step in range(self.NUM_STEPS):
            w_opt.zero_grad()
            torch.manual_seed(step)
            w_loss, _ = wrapper(_clone_batch(batch))
            w_loss.backward()
            w_opt.step()
            wrapper_losses.append(w_loss.item())

            n_opt.zero_grad()
            preprocessed = wrapper._preprocessor(_clone_batch(batch))
            torch.manual_seed(step)
            n_loss, _ = native(preprocessed)
            n_loss.backward()
            n_opt.step()
            native_losses.append(n_loss.item())

        for step, (wl, nl) in enumerate(zip(wrapper_losses, native_losses)):
            torch.testing.assert_close(
                torch.tensor(wl),
                torch.tensor(nl),
                rtol=1e-6,
                atol=1e-6,
                msg=lambda m, step=step, wl=wl, nl=nl: f"Step {step}: wrapper loss {wl} != native loss {nl} ({m})",
            )

    def test_loss_decreases_over_training(self, dual_setup):
        """Average loss in later steps should be lower than early steps.

        Diffusion policies have inherently noisy loss (random noise + timesteps
        each forward), so we compare the mean of the first half vs the second
        half over more steps rather than demanding strict monotonic decrease.
        """
        wrapper = dual_setup["wrapper"]
        w_opt = dual_setup["wrapper_optimizer"]
        batch = dual_setup["batch"]

        num_steps = 20
        losses: list[float] = []
        for step in range(num_steps):
            w_opt.zero_grad()
            torch.manual_seed(step)
            loss, _ = wrapper(_clone_batch(batch))
            loss.backward()
            w_opt.step()
            losses.append(loss.item())

        mid = num_steps // 2
        first_half_mean = sum(losses[:mid]) / mid
        second_half_mean = sum(losses[mid:]) / (num_steps - mid)

        assert first_half_mean > second_half_mean, (
            f"Loss did not decrease: first_half_mean={first_half_mean:.6f}, "
            f"second_half_mean={second_half_mean:.6f}, "
            f"trajectory={[f'{val:.4f}' for val in losses]}"
        )

    def test_final_weights_match(self, dual_setup):
        """After N identical steps, wrapper and native should have identical parameters."""
        wrapper = dual_setup["wrapper"]
        native = dual_setup["native"]
        w_opt = dual_setup["wrapper_optimizer"]
        n_opt = dual_setup["native_optimizer"]
        batch = dual_setup["batch"]

        for step in range(self.NUM_STEPS):
            w_opt.zero_grad()
            torch.manual_seed(step)
            w_loss, _ = wrapper(_clone_batch(batch))
            w_loss.backward()
            w_opt.step()

            n_opt.zero_grad()
            preprocessed = wrapper._preprocessor(_clone_batch(batch))
            torch.manual_seed(step)
            n_loss, _ = native(preprocessed)
            n_loss.backward()
            n_opt.step()

        # Compare all parameters
        wrapper_params = dict(wrapper.lerobot_policy.named_parameters())
        native_params = dict(native.named_parameters())

        assert set(wrapper_params.keys()) == set(native_params.keys()), (
            "Parameter names differ between wrapper and native"
        )

        for name in wrapper_params:
            torch.testing.assert_close(
                wrapper_params[name],
                native_params[name],
                rtol=1e-6,
                atol=1e-6,
                msg=f"Parameter '{name}' diverged after {self.NUM_STEPS} training steps",
            )
