# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generic export+inference round-trip for every supported LeRobot wrapper.

Builds each wrapper from a small dataset, exports via ``torch`` backend, reloads
through :class:`physicalai.inference.InferenceModel`, and asserts the exported
package produces a usable action chunk for a real dataset observation. This
closes the gap that ``test_first_party_e2e.py`` only covers first-party policies
and ``test_lerobot_wrapper_equivalence.py`` never invokes ``.export()``.
"""

# ruff: noqa: S101

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from physicalai.data import LeRobotDataModule
from physicalai.data.lerobot import FormatConverter, get_delta_timestamps_from_policy
from physicalai.devices import get_available_device
from physicalai.inference import InferenceModel
from physicalai.policies.lerobot import (
    ACT,
    PI0,
    PI05,
    SUPPORTED_POLICIES,
    Diffusion,
    Groot,
    PI0Fast,
    SmolVLA,
    XVLA,
)
from physicalai.policies.lerobot.policy import NamedLeRobotPolicy

pytest.importorskip("lerobot", reason="LeRobot not installed")

DATASET_REPO_ID = "lerobot/aloha_sim_insertion_human"

_NAMED_WRAPPER: dict[str, type[NamedLeRobotPolicy]] = {
    "act": ACT,
    "diffusion": Diffusion,
    "groot": Groot,
    "pi0": PI0,
    "pi0_fast": PI0Fast,
    "pi05": PI05,
    "smolvla": SmolVLA,
    "xvla": XVLA,
}

_VLA_POLICIES = {"pi0", "pi05", "pi0_fast", "groot", "smolvla", "xvla"}

_E2E_XFAIL_REASONS: dict[str, str] = {
    "groot": "hardcodes flash_attention_2 in eagle2_hg_model (upstream lerobot)",
    "xvla": "requires explicit vision_config kwarg not derivable from dataset",
}


def _policy_param(policy_name: str) -> Any:
    reason = _E2E_XFAIL_REASONS.get(policy_name)
    if reason is not None:
        return pytest.param(policy_name, marks=pytest.mark.xfail(strict=False, reason=reason))
    return policy_name


def _get_policy_kwargs(policy_name: str) -> dict[str, Any]:
    """Kwargs for ``from_dataset`` in this export round-trip test only.

    These overrides exist because this test builds scaffold policies from
    dataset metadata rather than loading tuned checkpoints.
    """
    if policy_name == "diffusion":
        return {"num_train_timesteps": 10, "num_inference_steps": 5}
    if policy_name == "groot":
        return {
            "tune_llm": False,
            "tune_visual": False,
            "tune_projector": True,
            "tune_diffusion_model": False,
        }
    if policy_name == "pi0_fast":
        return {
            "dtype": "bfloat16",
            # Random-init from_dataset cannot emit valid FAST "Action:" tokens.
            "validate_action_token_prefix": False,
        }
    if policy_name in _VLA_POLICIES:
        return {"dtype": "bfloat16"}
    return {}


def _vla_cpu_skip(policy_name: str) -> None:
    if policy_name in _VLA_POLICIES and get_available_device() == "cpu":
        pytest.skip(f"{policy_name} requires CUDA or XPU")


@pytest.fixture(scope="module")
def aloha_dataset() -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(DATASET_REPO_ID)
    for feature_stats in ds.meta.stats.values():
        if "min" in feature_stats and "q01" not in feature_stats:
            min_val = feature_stats["min"]
            max_val = feature_stats["max"]
            feature_stats["q01"] = min_val.copy() if hasattr(min_val, "copy") else min_val
            feature_stats["q99"] = max_val.copy() if hasattr(max_val, "copy") else max_val
    return ds


@pytest.fixture(scope="module")
def sample_observation_dict(aloha_dataset: Any) -> dict[str, Any]:
    policy_name = next(p for p in SUPPORTED_POLICIES if p not in _VLA_POLICIES)
    delta_timestamps = get_delta_timestamps_from_policy(policy_name)
    dm = LeRobotDataModule(
        repo_id=DATASET_REPO_ID,
        train_batch_size=1,
        episodes=[0],
        data_format="lerobot",
        delta_timestamps=delta_timestamps or None,
    )
    dm.setup("fit")
    sample_batch = next(iter(dm.train_dataloader()))
    batch_obs = FormatConverter.to_observation(sample_batch)
    return batch_obs[0:1].to("cpu").to_numpy().to_dict(flatten=False)


@pytest.mark.parametrize("policy_name", [_policy_param(p) for p in SUPPORTED_POLICIES])
def test_lerobot_wrapper_torch_export_roundtrip(
    policy_name: str,
    aloha_dataset: Any,
    sample_observation_dict: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Named wrapper builds, exports to torch, reloads via InferenceModel, predicts action chunk."""
    _vla_cpu_skip(policy_name)

    wrapper_cls = _NAMED_WRAPPER[policy_name]
    wrapper = wrapper_cls.from_dataset(aloha_dataset, **_get_policy_kwargs(policy_name))

    if "torch" not in wrapper.get_supported_export_backends():
        pytest.skip(f"{policy_name} wrapper does not support torch export")

    export_dir = tmp_path / f"{policy_name}_torch"
    wrapper.export(export_dir, backend="torch")

    assert (export_dir / "manifest.json").exists()
    assert any(export_dir.glob("*.pt"))

    inference_model = InferenceModel.load(export_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    assert inference_model.backend == "torch"

    with torch.no_grad():
        chunk = inference_model.predict_action_chunk(sample_observation_dict)

    action_dim = next(iter(wrapper._config.output_features.values())).shape[-1]
    assert chunk.ndim == 2
    assert chunk.shape[-1] == action_dim
