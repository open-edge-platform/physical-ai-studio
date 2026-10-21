"""Studio discovers VLA-JEPA without claiming unverified capabilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from training.job import TrainingJobSpec, build_policy
from training.policy_catalog import available_policy_classes, policy_catalog


def test_vla_jepa_is_native_lazy_and_available() -> None:
    """The real Studio construction path needs neither downloads nor feature metadata."""
    with patch("physicalai.policies.list_physicalai_policies", return_value=["vla_jepa"]):
        classes = available_policy_classes(missing_dependencies=lambda _name: [])
    catalog = policy_catalog(classes)
    assert catalog.available[0].id == "vla_jepa"
    assert catalog.available[0].min_vram is None
    assert catalog.backends == {"vla_jepa": ["torch"]}
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained") as resolve:
        policy = build_policy(TrainingJobSpec(policy="vla_jepa"))
    assert policy.__class__.__module__ == "physicalai.policies.vla_jepa.policy"
    assert policy.model is None
    assert policy.pretrained_name_or_path == "lerobot/VLA-JEPA-Pretrain"
    assert policy.enable_world_model is True
    resolve.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_requested_training_mode_reaches_native_policy(enabled: bool) -> None:
    """The shared job runner honors both full and action-only training requests."""
    policy = build_policy(TrainingJobSpec(policy="vla_jepa", enable_world_model=enabled))
    assert policy.enable_world_model is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_world_model_options_are_rejected_for_other_policies(enabled: bool) -> None:
    """A policy-specific option cannot be silently ignored on ACT."""
    with pytest.raises(ValidationError, match="only supported"):
        TrainingJobSpec(policy="act", enable_world_model=enabled)


@pytest.mark.parametrize("enabled", [None, False])
def test_resume_preserves_or_explicitly_disables_world_model(tmp_path: Path, enabled: bool | None) -> None:
    """Resuming without an override retains the saved checkpoint mode."""
    cls = MagicMock()
    path = tmp_path / "model.ckpt"
    with patch("physicalai.policies.get_physicalai_policy_class", return_value=cls):
        build_policy(TrainingJobSpec(policy="vla_jepa", enable_world_model=enabled), resume_from=path)
    expected = {} if enabled is None else {"enable_world_model": enabled}
    cls.load_from_checkpoint.assert_called_once_with(str(path), **expected)


def test_missing_vla_jepa_dependencies_hide_the_policy() -> None:
    """Remote and local catalogs both require the optional policy dependencies."""
    with patch("physicalai.policies.list_physicalai_policies", return_value=["vla_jepa"]):
        assert available_policy_classes(missing_dependencies=lambda _name: ["diffusers"]) == {}


@pytest.mark.parametrize("use_dora", [False, True])
def test_vla_jepa_supports_lora(use_dora: bool) -> None:
    """Studio passes the adapter settings to the lazy native policy."""
    spec = TrainingJobSpec(policy="vla_jepa", lora_enabled=True, lora_rank=4, lora_use_dora=use_dora)
    policy = build_policy(spec)
    assert policy.model is None
    assert policy.hparams.lora_enabled is True
    assert policy.hparams.lora_rank == 4
    assert policy.hparams.lora_use_dora is use_dora
