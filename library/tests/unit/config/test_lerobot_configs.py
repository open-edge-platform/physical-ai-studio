# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the curated LeRobot starter YAML configs.

These tests operate on the YAML files in ``library/configs/lerobot/`` as
data. They do not instantiate policies (which would download large model
checkpoints) but verify that every shipped starter config is:

1. Valid YAML.
2. Targets a real ``NamedLeRobotPolicy`` subclass via ``class_path``.
3. Has a self-consistent ``chunk_size`` ↔ ``delta_timestamps.action`` pair
   where both are declared. This catches the class of pre-existing bugs
   (missing ``delta_timestamps`` on Diffusion/Groot) that motivated this
   curation effort in the first place.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from physicalai.policies.lerobot.aliases import ACT, Diffusion, Groot, PI0, PI05, PI0Fast, SmolVLA, XVLA
from physicalai.policies.lerobot.policy import NamedLeRobotPolicy

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs" / "lerobot"

# Every named LeRobot policy alias must ship a starter YAML. Keeping this as
# an explicit mapping (rather than inferring from the directory) means
# adding a new alias without a YAML, or vice versa, fails loudly.
EXPECTED_CONFIGS: dict[str, type[NamedLeRobotPolicy]] = {
    "act.yaml": ACT,
    "diffusion.yaml": Diffusion,
    "smolvla.yaml": SmolVLA,
    "groot.yaml": Groot,
    "pi0.yaml": PI0,
    "pi05.yaml": PI05,
    "pi0fast.yaml": PI0Fast,
    "xvla.yaml": XVLA,
}


def _load(yaml_path: Path) -> dict:
    with yaml_path.open("r") as f:
        return yaml.safe_load(f)


def test_configs_dir_exists() -> None:
    assert CONFIGS_DIR.is_dir(), f"Configs dir missing: {CONFIGS_DIR}"


def test_all_expected_configs_present() -> None:
    """Every alias in aliases.py must have a matching starter YAML and vice versa."""
    on_disk = {p.name for p in CONFIGS_DIR.glob("*.yaml")}
    expected = set(EXPECTED_CONFIGS.keys())
    missing = expected - on_disk
    extra = on_disk - expected
    assert not missing, f"Missing starter YAML(s): {sorted(missing)}"
    assert not extra, f"Unexpected YAML(s) in configs/lerobot/: {sorted(extra)}"


@pytest.mark.parametrize("yaml_name", sorted(EXPECTED_CONFIGS.keys()))
def test_yaml_is_valid(yaml_name: str) -> None:
    """Each starter YAML parses and has the expected top-level structure."""
    config = _load(CONFIGS_DIR / yaml_name)
    assert isinstance(config, dict), f"{yaml_name}: top level must be a mapping"
    for key in ("model", "data", "trainer"):
        assert key in config, f"{yaml_name}: missing top-level '{key}' section"
    assert "class_path" in config["model"], f"{yaml_name}: model.class_path missing"
    assert "init_args" in config["model"], f"{yaml_name}: model.init_args missing"


@pytest.mark.parametrize(("yaml_name", "expected_cls"), sorted(EXPECTED_CONFIGS.items()))
def test_class_path_resolves_to_alias(yaml_name: str, expected_cls: type[NamedLeRobotPolicy]) -> None:
    """model.class_path must import to the NamedLeRobotPolicy alias for this YAML."""
    config = _load(CONFIGS_DIR / yaml_name)
    class_path: str = config["model"]["class_path"]
    module_name, _, attr = class_path.rpartition(".")
    assert module_name, f"{yaml_name}: class_path '{class_path}' is not dotted"
    module = importlib.import_module(module_name)
    resolved = getattr(module, attr)
    assert resolved is expected_cls, (
        f"{yaml_name}: class_path '{class_path}' resolves to {resolved!r}, expected {expected_cls!r}"
    )


@pytest.mark.parametrize("yaml_name", sorted(EXPECTED_CONFIGS.keys()))
def test_data_targets_lerobot_datamodule(yaml_name: str) -> None:
    """Starter YAMLs should all wire up LeRobotDataModule for consistency."""
    config = _load(CONFIGS_DIR / yaml_name)
    assert config["data"]["class_path"] == "physicalai.data.lerobot.LeRobotDataModule", (
        f"{yaml_name}: data.class_path should be physicalai.data.lerobot.LeRobotDataModule"
    )


# Action-horizon invariants per policy family. These reflect how each
# policy's ``__post_init__`` derives ``action_delta_indices`` upstream:
#
#   ACT / Diffusion / SmolVLA / PI0 / PI05 / PI0Fast / XVLA:
#       action_delta_indices = range(chunk_size)
#   Groot:
#       action_delta_indices = range(min(chunk_size, 16))  -- capped at 16
#
# If the wrong length is supplied, the collate step silently drops frames
# or torch raises an opaque shape error at forward pass time, which is
# exactly the class of bug this test guards against.
ACTION_HORIZON_FROM_CHUNK_SIZE: dict[str, callable] = {
    "act.yaml": lambda cs: cs,
    "diffusion.yaml": None,  # uses horizon, not chunk_size - checked separately
    "smolvla.yaml": lambda cs: cs,
    "groot.yaml": lambda cs: min(cs, 16),
    "pi0.yaml": lambda cs: cs,
    "pi05.yaml": lambda cs: cs,
    "pi0fast.yaml": lambda cs: cs,
    "xvla.yaml": lambda cs: cs,
}


@pytest.mark.parametrize(
    "yaml_name",
    sorted(name for name, fn in ACTION_HORIZON_FROM_CHUNK_SIZE.items() if fn is not None),
)
def test_action_delta_timestamps_match_chunk_size(yaml_name: str) -> None:
    """len(delta_timestamps.action) must match the policy's effective action horizon."""
    config = _load(CONFIGS_DIR / yaml_name)
    model_args = config["model"]["init_args"]
    data_args = config["data"]["init_args"]

    assert "chunk_size" in model_args, f"{yaml_name}: model.init_args.chunk_size required"
    chunk_size = model_args["chunk_size"]
    expected_len = ACTION_HORIZON_FROM_CHUNK_SIZE[yaml_name](chunk_size)

    delta_timestamps = data_args.get("delta_timestamps")
    assert delta_timestamps is not None, (
        f"{yaml_name}: data.init_args.delta_timestamps is required "
        f"(policy expects {expected_len} action frames)"
    )
    action_offsets = delta_timestamps.get("action")
    assert action_offsets is not None, f"{yaml_name}: delta_timestamps.action is required"
    assert len(action_offsets) == expected_len, (
        f"{yaml_name}: len(delta_timestamps.action)={len(action_offsets)} "
        f"does not match expected action horizon {expected_len} (chunk_size={chunk_size})"
    )


def test_diffusion_delta_timestamps_match_horizon_and_n_obs_steps() -> None:
    """Diffusion policy derives horizon and n_obs_steps from its own config.

    Unlike chunk_size-based policies, DiffusionConfig uses:
        observation_delta_indices = range(1 - n_obs_steps, 1)
        action_delta_indices      = range(1 - n_obs_steps, 1 - n_obs_steps + horizon)

    so the YAML must declare both observation.* and action offsets with
    matching lengths.
    """
    config = _load(CONFIGS_DIR / "diffusion.yaml")
    model_args = config["model"]["init_args"]
    data_args = config["data"]["init_args"]

    n_obs_steps = model_args.get("n_obs_steps", 2)  # LeRobot default
    horizon = model_args.get("horizon", 16)  # LeRobot default

    delta_timestamps = data_args.get("delta_timestamps")
    assert delta_timestamps is not None, "diffusion.yaml: delta_timestamps missing"

    # Must have at least one observation.* entry with n_obs_steps frames.
    obs_keys = [k for k in delta_timestamps if k.startswith("observation.")]
    assert obs_keys, "diffusion.yaml: at least one observation.* delta_timestamps entry required"
    for key in obs_keys:
        assert len(delta_timestamps[key]) == n_obs_steps, (
            f"diffusion.yaml: len(delta_timestamps['{key}'])={len(delta_timestamps[key])} "
            f"does not match n_obs_steps={n_obs_steps}"
        )

    action_offsets = delta_timestamps.get("action")
    assert action_offsets is not None, "diffusion.yaml: delta_timestamps.action missing"
    assert len(action_offsets) == horizon, (
        f"diffusion.yaml: len(delta_timestamps.action)={len(action_offsets)} "
        f"does not match horizon={horizon}"
    )
