# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: native PyTorch ACT vs OpenVINO export numerical and closed-loop parity.

Loads a real pretrained LeRobot ACT checkpoint via
``physicalai.policies.act.ACT(pretrained_name_or_path=...)``, exports it to
OpenVINO, and validates that the export reproduces the native model's behaviour:

  1. **Numerical**: ``predict_action_chunk`` max-abs-diff and cosine similarity
     across observations sampled from the gym-aloha environment.
  2. **Closed-loop**: per-episode success/fail outcomes and success-rate delta on
     gym-aloha's ``AlohaTransferCube-v0`` task with matching seeds.

Both tests are marked ``@pytest.mark.slow`` because they require downloading a
real checkpoint and running many environment steps.  Run them explicitly with::

    pytest -m slow tests/integration/test_act_openvino_parity.py
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

gym_aloha = pytest.importorskip("gym_aloha", reason="gym-aloha not installed")
gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from physicalai.data.observation import Observation
from physicalai.inference import InferenceModel
from physicalai.policies.act.policy import ACT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_ID = "gym_aloha/AlohaTransferCube-v0"
_CHECKPOINT = "lerobot/act_aloha_sim_transfer_cube_human"
_MAX_ABS_DIFF_TOLERANCE = 0.05
_SUCCESS_RATE_DIFF_TOLERANCE_PCT = 15.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _obs_to_observation(obs: dict) -> Observation:
    state = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0)
    images = torch.from_numpy(obs["pixels"]["top"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return Observation(state=state, images=images)


def _obs_to_input(obs: dict) -> dict[str, np.ndarray]:
    image = obs["pixels"]["top"].astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[None, ...]
    state = obs["agent_pos"].astype(np.float32)[None, ...]
    return {"state": state, "images": image}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def native_policy() -> ACT:
    """Load the native ACT policy from the pretrained checkpoint once per module."""
    policy = ACT(pretrained_name_or_path=_CHECKPOINT)
    policy.eval()
    return policy


@pytest.fixture(scope="module")
def exported_model(native_policy: ACT, tmp_path_factory: pytest.TempPathFactory) -> InferenceModel:
    """Export the native policy to OpenVINO once per module and return an ``InferenceModel``."""
    export_dir = tmp_path_factory.mktemp("act_openvino_export")
    native_policy.export(export_dir, backend="openvino", compress_to_fp16=False)
    return InferenceModel(str(export_dir), device="CPU")


@pytest.fixture(scope="module")
def aloha_env():
    """Create the gym-aloha environment once per module."""
    import gym_aloha  # noqa: F401

    env = gym.make(_ENV_ID, obs_type="pixels_agent_pos")
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Numerical parity test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestACTOpenVINONumericalParity:
    """Verify predict_action_chunk outputs are numerically close between backends."""

    N_SAMPLES = 20
    SEED_OFFSET = 0
    GLOBAL_SEED = 42

    def test_max_abs_diff_within_tolerance(
        self,
        native_policy: ACT,
        exported_model: InferenceModel,
        aloha_env: Any,
    ) -> None:
        """Max absolute difference across all samples must be below tolerance."""
        _set_seed(self.GLOBAL_SEED)

        max_abs_diffs: list[float] = []
        for i in range(self.N_SAMPLES):
            obs, _ = aloha_env.reset(seed=self.SEED_OFFSET + i)

            native_policy.reset()
            with torch.inference_mode():
                native_chunk = native_policy.predict_action_chunk(_obs_to_observation(obs))
            native_arr = (
                native_chunk[0].cpu().numpy() if native_chunk.ndim == 3 else native_chunk.cpu().numpy()  # noqa: PLR2004
            )

            exported_chunk = np.asarray(exported_model.predict_action_chunk(_obs_to_input(obs)))
            exported_arr = exported_chunk[0] if exported_chunk.ndim == 3 else exported_chunk  # noqa: PLR2004

            max_abs_diffs.append(float(np.abs(native_arr - exported_arr).max()))

        overall_max = max(max_abs_diffs)
        assert overall_max <= _MAX_ABS_DIFF_TOLERANCE, (
            f"Max abs diff {overall_max:.6f} exceeds tolerance {_MAX_ABS_DIFF_TOLERANCE}"
        )

    def test_cosine_similarity_near_one(
        self,
        native_policy: ACT,
        exported_model: InferenceModel,
        aloha_env: Any,
    ) -> None:
        """Min cosine similarity across all samples must be close to 1."""
        _set_seed(self.GLOBAL_SEED)

        cosine_sims: list[float] = []
        for i in range(self.N_SAMPLES):
            obs, _ = aloha_env.reset(seed=self.SEED_OFFSET + i)

            native_policy.reset()
            with torch.inference_mode():
                native_chunk = native_policy.predict_action_chunk(_obs_to_observation(obs))
            native_flat = (
                native_chunk[0].cpu().numpy() if native_chunk.ndim == 3 else native_chunk.cpu().numpy()  # noqa: PLR2004
            ).flatten()

            exported_chunk = np.asarray(exported_model.predict_action_chunk(_obs_to_input(obs)))
            exported_flat = (exported_chunk[0] if exported_chunk.ndim == 3 else exported_chunk).flatten()  # noqa: PLR2004

            cosine_sims.append(
                float(np.dot(native_flat, exported_flat) / (np.linalg.norm(native_flat) * np.linalg.norm(exported_flat) + 1e-12))
            )

        assert min(cosine_sims) >= 0.99, (  # noqa: PLR2004
            f"Min cosine similarity {min(cosine_sims):.6f} is below 0.99"
        )


# ---------------------------------------------------------------------------
# Closed-loop parity test
# ---------------------------------------------------------------------------


def _run_native_episode(env: Any, policy: ACT, seed: int) -> dict[str, Any]:
    policy.reset()
    obs, _ = env.reset(seed=seed)
    done = False
    success = False
    steps = 0
    while not done:
        with torch.inference_mode():
            action = policy.select_action(_obs_to_observation(obs))
        obs, _reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        success = success or bool(info.get("is_success", False))
        done = terminated or truncated
        steps += 1
    return {"success": success, "steps": steps}


def _run_exported_episode(env: Any, model: InferenceModel, seed: int, n_action_steps: int) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    success = False
    steps = 0
    max_steps = getattr(env.spec, "max_episode_steps", None) or 10_000
    action_queue: list[np.ndarray] = []

    while steps < max_steps:
        if not action_queue:
            chunk = np.asarray(model.predict_action_chunk(_obs_to_input(obs)))
            if chunk.ndim == 3:  # noqa: PLR2004
                chunk = chunk[0]
            action_queue = list(chunk[:n_action_steps])

        obs, _reward, terminated, truncated, info = env.step(action_queue.pop(0))
        steps += 1
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break

    return {"success": success, "steps": steps}


@pytest.mark.slow
class TestACTOpenVINOClosedLoopParity:
    """Verify closed-loop episode outcomes are comparable between backends."""

    N_EPISODES = 10
    SEED_OFFSET = 0
    N_ACTION_STEPS = 100

    def test_success_rate_difference_within_tolerance(
        self,
        native_policy: ACT,
        exported_model: InferenceModel,
        aloha_env: Any,
    ) -> None:
        """Absolute success-rate difference must be within tolerance."""
        native_successes = 0
        exported_successes = 0

        for ep in range(self.N_EPISODES):
            seed = self.SEED_OFFSET + ep
            native_result = _run_native_episode(aloha_env, native_policy, seed)
            exported_result = _run_exported_episode(aloha_env, exported_model, seed, self.N_ACTION_STEPS)
            native_successes += int(native_result["success"])
            exported_successes += int(exported_result["success"])

        native_rate = native_successes / self.N_EPISODES * 100
        exported_rate = exported_successes / self.N_EPISODES * 100
        diff = abs(native_rate - exported_rate)

        assert diff <= _SUCCESS_RATE_DIFF_TOLERANCE_PCT, (
            f"Success-rate diff {diff:.1f}pp exceeds tolerance {_SUCCESS_RATE_DIFF_TOLERANCE_PCT}pp "
            f"(native={native_rate:.1f}%, openvino={exported_rate:.1f}%)"
        )
