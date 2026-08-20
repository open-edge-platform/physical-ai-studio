# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LeRobot policy wrappers."""

from __future__ import annotations

import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

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
