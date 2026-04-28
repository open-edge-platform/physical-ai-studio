# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the curated LeRobot starter YAML configs.

These tests operate on the YAML files in ``library/configs/lerobot/`` as
data. They do not instantiate policies (which would download large model
checkpoints) but verify that every shipped starter config is:

1. Valid YAML with the expected top-level structure.
2. Targets a real ``NamedLeRobotPolicy`` subclass via ``class_path``.
3. Has ``delta_timestamps`` whose action offsets are EXACTLY the consecutive
   frame stamps the policy expects at the dataset's fps. Length-only checks
   are insufficient because using e.g. ``[0.0, 0.1, ..., 0.9]`` at 50 fps
   silently samples every fifth frame, breaking the policy's effective
   horizon without any runtime error.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import yaml

from physicalai.policies.lerobot import aliases as aliases_mod
from physicalai.policies.lerobot.policy import NamedLeRobotPolicy

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs" / "lerobot"


def _discover_named_aliases() -> dict[str, type[NamedLeRobotPolicy]]:
    """Return ``{<POLICY_NAME>.yaml: alias_cls}`` for every concrete alias.

    Deriving from ``aliases.py`` (rather than a hand-maintained dict) means
    adding a new alias without a starter YAML, or vice versa, fails loudly
    in ``test_all_expected_configs_present`` below.
    """
    discovered: dict[str, type[NamedLeRobotPolicy]] = {}
    for _, obj in inspect.getmembers(aliases_mod, inspect.isclass):
        if obj is NamedLeRobotPolicy or not issubclass(obj, NamedLeRobotPolicy):
            continue
        if inspect.getmodule(obj) is not aliases_mod:
            continue
        policy_name = getattr(obj, "POLICY_NAME", None)
        assert isinstance(policy_name, str) and policy_name, f"{obj.__name__} missing POLICY_NAME"
        discovered[f"{policy_name}.yaml"] = obj
    return discovered


EXPECTED_CONFIGS: dict[str, type[NamedLeRobotPolicy]] = _discover_named_aliases()


def _load(yaml_path: Path) -> dict:
    with yaml_path.open("r") as f:
        return yaml.safe_load(f)


def _approx_equal(actual: list[float], expected: list[float]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(abs(a - e) < 1e-9 for a, e in zip(actual, expected, strict=True))


# Default repo_id across every starter YAML. Pinned so the timestamp
# expectations below stay valid; if a starter switches datasets it must also
# revisit its delta_timestamps. The dataset is 50 fps (verified from
# ``meta/info.json`` of ``lerobot/aloha_sim_transfer_cube_human``).
EXPECTED_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
EXPECTED_FPS = 50
FRAME_STEP = 1.0 / EXPECTED_FPS


def _consecutive_offsets(start_frame: int, count: int) -> list[float]:
    """Return ``count`` consecutive frame offsets starting at ``start_frame``.

    Mirrors how ``LeRobotDataset`` converts integer delta indices to seconds:
    ``offset_seconds = frame_index / fps`` (see ``lerobot.datasets`` source).
    """
    return [round((start_frame + i) * FRAME_STEP, 9) for i in range(count)]


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
    """Every starter YAML wires LeRobotDataModule (no other DataModule subclasses allowed)."""
    config = _load(CONFIGS_DIR / yaml_name)
    assert config["data"]["class_path"] == "physicalai.data.lerobot.LeRobotDataModule", (
        f"{yaml_name}: data.class_path should be physicalai.data.lerobot.LeRobotDataModule"
    )


# Exact action offsets each starter YAML must declare, given the pinned 50 fps
# dataset. Each entry is ``(start_frame, count)``: action offsets equal
# ``range(start_frame, start_frame + count)`` divided by fps.
#
# Derivation per policy upstream (see configuration_*.py in lerobot/):
#   ACT / SmolVLA / PI0 / PI05 / PI0Fast / XVLA: action_delta_indices = range(chunk_size)
#   Diffusion: action_delta_indices = range(1 - n_obs_steps, 1 - n_obs_steps + horizon)
#   Groot:     action_delta_indices = range(min(chunk_size, 16))
EXPECTED_ACTION_OFFSETS: dict[str, tuple[int, int]] = {
    "diffusion.yaml": (-1, 16),
    "groot.yaml": (0, 16),
    "pi0.yaml": (0, 10),
    "pi05.yaml": (0, 10),
    "pi0_fast.yaml": (0, 10),
    "xvla.yaml": (0, 16),
}


@pytest.mark.parametrize("yaml_name", sorted(EXPECTED_ACTION_OFFSETS.keys()))
def test_pinned_repo_id_for_offset_assertions(yaml_name: str) -> None:
    """YAMLs whose action offsets are checked must use the pinned 50 fps dataset.

    EXPECTED_ACTION_OFFSETS hard-codes frame indices that are only valid for
    the pinned ``lerobot/aloha_sim_transfer_cube_human`` dataset. If a YAML
    switches datasets it must update both the YAML offsets and this test.
    """
    config = _load(CONFIGS_DIR / yaml_name)
    assert config["data"]["init_args"].get("repo_id") == EXPECTED_REPO_ID, (
        f"{yaml_name}: repo_id changed from pinned {EXPECTED_REPO_ID!r}; "
        f"delta_timestamps assumptions in this test must be revisited."
    )


@pytest.mark.parametrize("yaml_name", sorted(EXPECTED_ACTION_OFFSETS.keys()))
def test_action_delta_timestamps_are_exact_consecutive_frames(yaml_name: str) -> None:
    """delta_timestamps.action must equal the EXACT consecutive frame offsets."""
    config = _load(CONFIGS_DIR / yaml_name)
    delta_timestamps = config["data"]["init_args"].get("delta_timestamps")
    assert delta_timestamps is not None, f"{yaml_name}: delta_timestamps missing"

    action_offsets = delta_timestamps.get("action")
    assert action_offsets is not None, f"{yaml_name}: delta_timestamps.action missing"

    start, count = EXPECTED_ACTION_OFFSETS[yaml_name]
    expected = _consecutive_offsets(start, count)
    assert _approx_equal(action_offsets, expected), (
        f"{yaml_name}: delta_timestamps.action={action_offsets} does not match "
        f"the expected consecutive frame offsets at {EXPECTED_FPS} fps: {expected}. "
        f"This usually means the offsets were authored assuming the wrong fps "
        f"(e.g. [0.0, 0.1, ...] is correct at 10 fps but wrong at 50 fps)."
    )


def test_diffusion_observation_offsets_are_exact() -> None:
    """Diffusion's observation.* offsets must be the n_obs_steps frames ending at t=0."""
    config = _load(CONFIGS_DIR / "diffusion.yaml")
    n_obs_steps = config["model"]["init_args"].get("n_obs_steps", 2)
    delta_timestamps = config["data"]["init_args"]["delta_timestamps"]

    obs_keys = [k for k in delta_timestamps if k.startswith("observation.")]
    assert obs_keys, "diffusion.yaml: at least one observation.* delta_timestamps entry required"

    expected_obs = _consecutive_offsets(1 - n_obs_steps, n_obs_steps)
    for key in obs_keys:
        assert _approx_equal(delta_timestamps[key], expected_obs), (
            f"diffusion.yaml: delta_timestamps['{key}']={delta_timestamps[key]} does not "
            f"match expected {expected_obs} for n_obs_steps={n_obs_steps} at {EXPECTED_FPS} fps."
        )


def test_pi05_uses_physicalai_data_format() -> None:
    """PI0.5 normalization uses QUANTILES, which only the physicalai data_format auto-synthesizes.

    Selecting ``data_format: "lerobot"`` would crash in NormalizerProcessorStep
    because the default LeRobot dataset metadata has no q01/q99 stats.
    """
    config = _load(CONFIGS_DIR / "pi05.yaml")
    data_format = config["data"]["init_args"].get("data_format")
    assert data_format == "physicalai", (
        f"pi05.yaml: data_format must be 'physicalai' to auto-synthesize quantile "
        f"stats for QUANTILES normalization (got {data_format!r})"
    )
