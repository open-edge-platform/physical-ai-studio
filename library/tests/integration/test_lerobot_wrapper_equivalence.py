# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for LeRobot wrapper numerical equivalence.

Validates that training through physicalai's LeRobotPolicy wrapper via the
Lightning Trainer produces identical loss trajectories compared to running
native LeRobot training with the same data and config.

Weight-level equivalence is validated in unit tests (``test_lerobot.py``)
using a manual forward/backward loop where bit-identical results are
achievable.  In the Trainer-based integration tests, Lightning's internal
bookkeeping introduces small float32 rounding differences (~1e-6/step) that
accumulate across steps, making exact weight comparison unreliable.  Loss
trajectories, however, remain within tight tolerances (rtol/atol=1e-5) and
are the correct integration-level signal for wrapper correctness.

Tests are structured in three tiers:
    1. Fast-dev-run: Single step, verifies loss matches native (all policies, CI).
    2. Multi-step (10 steps): Per-step loss trajectories match (all policies, CI).
    3. Regression (50 steps): Loss decreases consistently, trajectories match
       (all policies, nightly/@slow).

All target policies run in CI for tiers 1-2. Only tier 3 is @slow (nightly).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from lightning.pytorch.callbacks import Callback

from physicalai.data import LeRobotDataModule
from physicalai.data.lerobot import get_delta_timestamps_from_policy
from physicalai.policies.lerobot import LeRobotPolicy
from physicalai.train import Trainer

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("lerobot", reason="LeRobot not installed"),
    reason="Requires lerobot",
)

# All target policies — run in CI for fast-dev-run and multi-step tests.
# VLA policies (pi0, pi05, pi0_fast, groot) require flash_attn and are
# auto-skipped when dependencies are missing.
ALL_POLICIES = [
    "act",
    "diffusion",
    "vqbet",
    "tdmpc",
    "sac",
    "pi0",
    "pi05",
    "pi0_fast",
    "groot",
]

DATASET_REPO_ID = "lerobot/aloha_sim_insertion_human"

# VLA policies that need smaller batch/episode counts for GPU memory
_VLA_POLICIES = {"pi0", "pi05", "pi0_fast", "groot"}

# Policies excluded from wrapper-vs-native equivalence tests:
#   groot: hardcodes flash_attention_2 in eagle2_hg_model (upstream lerobot limitation)
#   tdmpc: encoder expects [B,T,C,H,W] images + requires square images
#   sac: MultiAdamConfig returns dict[str, Optimizer], not a single Optimizer
#   pi05: QUANTILES normalization requires q01/q99 stats not present in aloha dataset
#         (upstream lerobot dataset limitation, not a wrapper issue)
_EQUIVALENCE_SKIP_POLICIES = {"groot", "tdmpc", "sac", "pi05"}


class LossBatchCaptureCallback(Callback):
    def __init__(self) -> None:
        self.step_data: list[tuple[dict[str, Any], float]] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LeRobotPolicy,  # noqa: ARG002
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        loss_value = _to_loss_value(outputs)
        self.step_data.append((_clone_batch(batch), loss_value))


class SeedPerStepCallback(Callback):
    def __init__(self, base_seed: int = 42) -> None:
        self.base_seed = base_seed
        self._step = 0

    def on_train_batch_start(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LeRobotPolicy,  # noqa: ARG002
        batch: dict[str, Any],  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        torch.manual_seed(self.base_seed + self._step)
        self._step += 1


@pytest.fixture(scope="module")
def aloha_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(DATASET_REPO_ID)


def _skip_if_unsupported(policy_name: str) -> None:
    if policy_name in _EQUIVALENCE_SKIP_POLICIES:
        pytest.skip(f"{policy_name} excluded from equivalence tests")


def _to_loss_value(outputs: Any) -> float:
    if isinstance(outputs, dict):
        loss = outputs["loss"]
        return float(loss.item() if isinstance(loss, torch.Tensor) else loss)
    if isinstance(outputs, torch.Tensor):
        return float(outputs.item())
    return float(outputs)


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().clone()
        elif isinstance(value, dict):
            out[key] = _clone_batch(value)
        elif isinstance(value, (list, tuple)):
            out[key] = type(value)(
                item.detach().clone() if isinstance(item, torch.Tensor) else copy.deepcopy(item) for item in value
            )
        else:
            out[key] = copy.deepcopy(value)
    return out


def _make_datamodule(policy_name: str) -> LeRobotDataModule:
    delta_timestamps = get_delta_timestamps_from_policy(policy_name)
    batch_size = 1 if policy_name in _VLA_POLICIES else 8
    episodes = list(range(2)) if policy_name in _VLA_POLICIES else list(range(5))
    return LeRobotDataModule(
        repo_id=DATASET_REPO_ID,
        train_batch_size=batch_size,
        episodes=episodes,
        data_format="lerobot",
        delta_timestamps=delta_timestamps or None,
    )


def _extract_grad_clip_norm(wrapper: LeRobotPolicy) -> float | None:
    config = wrapper._config
    if config is None:
        return None

    preset = config.get_optimizer_preset()
    if hasattr(preset, "grad_clip_norm") and preset.grad_clip_norm is not None:
        value = float(preset.grad_clip_norm)
        return value if value > 0 else None

    for attr_name in ("optimizer_grad_clip_norm", "grad_clip_norm"):
        if hasattr(config, attr_name) and getattr(config, attr_name) is not None:
            value = float(getattr(config, attr_name))
            return value if value > 0 else None

    return None


def _get_policy_kwargs(policy_name: str) -> dict[str, Any]:
    if policy_name == "diffusion":
        return {"num_train_timesteps": 10, "num_inference_steps": 5}
    if policy_name == "groot":
        return {
            "tune_llm": False,
            "tune_visual": False,
            "tune_projector": True,
            "tune_diffusion_model": False,
        }
    # VLA models require bfloat16 to fit in 48GB GPU memory during training
    if policy_name in _VLA_POLICIES:
        return {"dtype": "bfloat16"}
    return {}


def _make_wrapper_and_native(
    policy_name: str,
    dataset: Any,
) -> tuple[LeRobotPolicy, torch.nn.Module, Any, float | None, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_kwargs = _get_policy_kwargs(policy_name)

    wrapper = LeRobotPolicy.from_dataset(policy_name, dataset, **policy_kwargs)
    native = copy.deepcopy(wrapper.lerobot_policy)
    native = native.to(device)
    native.train()

    optim_params = native.get_optim_params()
    native_optimizer = wrapper._config.get_optimizer_preset().build(optim_params)
    grad_clip_norm = _extract_grad_clip_norm(wrapper)
    return wrapper, native, native_optimizer, grad_clip_norm, device


def _make_native_from_state_dict(
    policy_name: str,
    dataset: Any,
    initial_state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    """Create a fresh native LeRobot policy loaded with *initial_state_dict*.

    This avoids ``copy.deepcopy`` on large VLA policies (4B+ params) which
    either OOMs (two full-size copies on GPU) or fails outright for some
    policy classes (e.g. PI05's lazy-init modules).
    """
    policy_kwargs = _get_policy_kwargs(policy_name)
    tmp_wrapper = LeRobotPolicy.from_dataset(policy_name, dataset, **policy_kwargs)
    native = tmp_wrapper.lerobot_policy
    native.load_state_dict(initial_state_dict)
    del tmp_wrapper
    native = native.to(device)
    native.train()
    return native


def _build_trainer(
    *,
    base_seed: int,
    grad_clip_norm: float | None,
    fast_dev_run: int | bool = False,
    max_steps: int | None = None,
    precision: str = "32-true",
) -> tuple[Trainer, LossBatchCaptureCallback]:
    capture_callback = LossBatchCaptureCallback()
    seed_callback = SeedPerStepCallback(base_seed=base_seed)

    trainer_kwargs: dict[str, Any] = {
        "enable_checkpointing": False,
        "logger": False,
        "enable_progress_bar": False,
        "devices": 1,
        "deterministic": True,
        "precision": precision,
        "callbacks": [seed_callback, capture_callback],
    }
    if grad_clip_norm is not None:
        trainer_kwargs["gradient_clip_val"] = grad_clip_norm
        trainer_kwargs["gradient_clip_algorithm"] = "norm"

    if fast_dev_run:
        trainer = Trainer(fast_dev_run=fast_dev_run, **trainer_kwargs)
    else:
        trainer = Trainer(max_steps=max_steps, **trainer_kwargs)

    return trainer, capture_callback


def _replay_batches_on_native(
    native: torch.nn.Module,
    native_optimizer: Any,
    captured_steps: list[tuple[dict[str, Any], float]],
    preprocessor: Any,
    *,
    base_seed: int,
    grad_clip_norm: float | None,
    use_autocast: bool = False,
) -> list[float]:
    native_losses: list[float] = []
    device_type = "cuda" if next(native.parameters()).is_cuda else "cpu"
    for step_idx, (captured_batch, _) in enumerate(captured_steps):
        native_optimizer.zero_grad()
        torch.manual_seed(base_seed + step_idx)
        preprocessed = preprocessor(_clone_batch(captured_batch))
        if use_autocast:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                native_output = native(preprocessed)
        else:
            native_output = native(preprocessed)
        native_loss = native_output[0] if isinstance(native_output, tuple) else native_output
        native_loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(native.parameters(), grad_clip_norm)
        native_optimizer.step()
        native_losses.append(float(native_loss.item()))
    return native_losses


def _run_vla_equivalence(
    policy_name: str,
    dataset: Any,
    *,
    base_seed: int,
    fast_dev_run: int | bool = False,
    max_steps: int | None = None,
) -> tuple[list[float], list[float], LeRobotPolicy, torch.nn.Module]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Create wrapper and snapshot initial weights on CPU.
    policy_kwargs = _get_policy_kwargs(policy_name)
    wrapper = LeRobotPolicy.from_dataset(policy_name, dataset, **policy_kwargs)
    initial_state_dict = {k: v.cpu().clone() for k, v in wrapper.lerobot_policy.state_dict().items()}
    grad_clip_norm = _extract_grad_clip_norm(wrapper)
    config = wrapper._config

    # 2. Train wrapper via Lightning Trainer and capture batches.
    trainer, capture_callback = _build_trainer(
        base_seed=base_seed,
        grad_clip_norm=grad_clip_norm,
        fast_dev_run=fast_dev_run,
        max_steps=max_steps,
        precision="bf16-mixed",
    )
    torch.manual_seed(base_seed)
    trainer.fit(wrapper, datamodule=_make_datamodule(policy_name))
    wrapper_losses = [loss for _, loss in capture_callback.step_data]
    captured_steps = list(capture_callback.step_data)
    preprocessor = wrapper._preprocessor

    # Free GPU before loading native model.
    wrapper.cpu()
    del trainer
    torch.cuda.empty_cache()

    # 3. Build native model from saved initial weights (on GPU).
    native = _make_native_from_state_dict(policy_name, dataset, initial_state_dict, device)
    del initial_state_dict
    torch.cuda.empty_cache()

    optim_params = native.get_optim_params()
    native_optimizer = config.get_optimizer_preset().build(optim_params)

    # 4. Replay captured batches on native model.
    native_losses = _replay_batches_on_native(
        native,
        native_optimizer,
        captured_steps,
        preprocessor,
        base_seed=base_seed,
        grad_clip_norm=grad_clip_norm,
        use_autocast=True,
    )
    return wrapper_losses, native_losses, wrapper, native


def _run_standard_equivalence(
    policy_name: str,
    dataset: Any,
    *,
    base_seed: int,
    fast_dev_run: int | bool = False,
    max_steps: int | None = None,
) -> tuple[list[float], list[float], LeRobotPolicy, torch.nn.Module]:
    wrapper, native, native_optimizer, grad_clip_norm, _device = _make_wrapper_and_native(policy_name, dataset)

    trainer, capture_callback = _build_trainer(
        base_seed=base_seed,
        grad_clip_norm=grad_clip_norm,
        fast_dev_run=fast_dev_run,
        max_steps=max_steps,
    )
    torch.manual_seed(base_seed)
    trainer.fit(wrapper, datamodule=_make_datamodule(policy_name))

    wrapper_losses = [loss for _, loss in capture_callback.step_data]
    native_losses = _replay_batches_on_native(
        native,
        native_optimizer,
        list(capture_callback.step_data),
        wrapper._preprocessor,
        base_seed=base_seed,
        grad_clip_norm=grad_clip_norm,
    )
    return wrapper_losses, native_losses, wrapper, native


def _run_trainer_capture_and_native_replay(
    policy_name: str,
    dataset: Any,
    *,
    base_seed: int,
    fast_dev_run: int | bool = False,
    max_steps: int | None = None,
) -> tuple[list[float], list[float], LeRobotPolicy, torch.nn.Module]:
    if policy_name in _VLA_POLICIES:
        return _run_vla_equivalence(
            policy_name,
            dataset,
            base_seed=base_seed,
            fast_dev_run=fast_dev_run,
            max_steps=max_steps,
        )
    return _run_standard_equivalence(
        policy_name,
        dataset,
        base_seed=base_seed,
        fast_dev_run=fast_dev_run,
        max_steps=max_steps,
    )


def _assert_trajectories_match(
    wrapper_losses: list[float],
    native_losses: list[float],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    assert len(wrapper_losses) == len(native_losses)

    torch.testing.assert_close(
        torch.tensor(wrapper_losses),
        torch.tensor(native_losses),
        rtol=rtol,
        atol=atol,
    )


def _assert_loss_decreases(losses: list[float], policy_name: str, label: str) -> None:
    mid = len(losses) // 2
    assert mid > 0
    first_half_mean = sum(losses[:mid]) / mid
    second_half_mean = sum(losses[mid:]) / (len(losses) - mid)
    assert first_half_mean > second_half_mean, (
        f"{label} ({policy_name}) did not decrease: "
        f"first_half_mean={first_half_mean:.6f}, second_half_mean={second_half_mean:.6f}, "
        f"trajectory={[f'{x:.6f}' for x in losses]}"
    )


# ---------------------------------------------------------------------------
# Tier 1: Fast-dev-run — single step, all policies, CI
# ---------------------------------------------------------------------------


class TestFastDevRunEquivalence:
    @pytest.fixture(params=ALL_POLICIES)
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        name = str(request.param)
        _skip_if_unsupported(name)
        return name

    def test_single_step_loss_matches(self, policy_name: str, aloha_dataset: Any) -> None:
        wrapper_losses, native_losses, _, _ = _run_trainer_capture_and_native_replay(
            policy_name,
            aloha_dataset,
            base_seed=101,
            fast_dev_run=1,
        )

        assert len(wrapper_losses) == 1
        assert len(native_losses) == 1
        _assert_trajectories_match(wrapper_losses, native_losses)


# ---------------------------------------------------------------------------
# Tier 2: Multi-step (10 steps) — all policies, CI
# ---------------------------------------------------------------------------


class TestMultiStepTrainerEquivalence:
    NUM_STEPS = 10

    @pytest.fixture(params=ALL_POLICIES)
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        name = str(request.param)
        _skip_if_unsupported(name)
        return name

    def test_loss_trajectories_match(self, policy_name: str, aloha_dataset: Any) -> None:
        wrapper_losses, native_losses, _, _ = _run_trainer_capture_and_native_replay(
            policy_name,
            aloha_dataset,
            base_seed=202,
            max_steps=self.NUM_STEPS,
        )

        assert len(wrapper_losses) == self.NUM_STEPS
        assert len(native_losses) == self.NUM_STEPS
        _assert_trajectories_match(wrapper_losses, native_losses)


# ---------------------------------------------------------------------------
# Tier 3: Regression (50 steps) — all policies, nightly/@slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRegressionTraining:
    NUM_STEPS = 50

    @pytest.fixture(params=ALL_POLICIES)
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        name = str(request.param)
        _skip_if_unsupported(name)
        return name

    def test_loss_decreases(self, policy_name: str, aloha_dataset: Any) -> None:
        wrapper_losses, native_losses, _, _ = _run_trainer_capture_and_native_replay(
            policy_name,
            aloha_dataset,
            base_seed=404,
            max_steps=self.NUM_STEPS,
        )

        assert len(wrapper_losses) == self.NUM_STEPS
        _assert_loss_decreases(wrapper_losses, policy_name, label="wrapper")
        _assert_loss_decreases(native_losses, policy_name, label="native")

    def test_long_run_trajectories_match(self, policy_name: str, aloha_dataset: Any) -> None:
        wrapper_losses, native_losses, _, _ = _run_trainer_capture_and_native_replay(
            policy_name,
            aloha_dataset,
            base_seed=505,
            max_steps=self.NUM_STEPS,
        )

        assert len(wrapper_losses) == self.NUM_STEPS
        _assert_trajectories_match(wrapper_losses, native_losses, rtol=1e-3, atol=1e-3)
