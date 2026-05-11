# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for LeRobot wrapper numerical equivalence.

Validates that training through physicalai's LeRobotPolicy wrapper via the
Lightning Trainer produces equivalent loss trajectories, gradients, and final
weights compared to running native LeRobot training with the same data and
config.

Lightning's internal bookkeeping introduces small float32 rounding differences
(~1e-6/step) that accumulate across steps; tier tolerances are calibrated from
measured drift rather than set to bit-exactness. See per-test docstrings for
the empirical justification of each tolerance.

Tests are structured in four tiers:
    1. Fast-dev-run: Single step, verifies loss matches native (all policies).
    2. Multi-step (10 steps): Per-step loss trajectories match (rtol=1e-5).
    3. Gradient parity: First-step parameter gradients match (rtol=1e-5).
    4. Weight parity: Post-training (10 steps) weights match (rtol=5e-5).
    5. Regression (50 steps): Loss decreases consistently (@slow / nightly).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from lightning.pytorch.callbacks import Callback

from physicalai.data import LeRobotDataModule
from physicalai.data.lerobot import get_delta_timestamps_from_policy
from physicalai.devices import get_available_device, get_device
from physicalai.policies.lerobot import SUPPORTED_POLICIES, VALIDATED_EQUIVALENCE_POLICIES, LeRobotPolicy
from physicalai.train import Trainer

pytest.importorskip("lerobot", reason="LeRobot not installed")

DATASET_REPO_ID = "lerobot/aloha_sim_insertion_human"

# VLA policies that need smaller batch/episode counts for GPU memory
_VLA_POLICIES = {"pi0", "pi05", "pi0_fast", "groot"}


def _empty_accelerator_cache(device: torch.device) -> None:
    """Free accelerator memory between large policy loads (CUDA + XPU)."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()

# Per-policy reasons why wrapper-vs-native equivalence cannot be validated.
# Named (in SUPPORTED_POLICIES) but not yet validated end-to-end: registered as
# xfail so test output stays honest and any future fix raises XPASS.
_EQUIVALENCE_XFAIL_REASONS: dict[str, str] = {
    "groot": "hardcodes flash_attention_2 in eagle2_hg_model (upstream lerobot)",
    "xvla": "requires explicit `vision_config` kwarg, not derivable from dataset",
}


def _policy_param(policy_name: str):
    reason = _EQUIVALENCE_XFAIL_REASONS.get(policy_name)
    if reason is not None:
        return pytest.param(policy_name, marks=pytest.mark.xfail(strict=False, reason=reason))
    return policy_name


ALL_POLICIES_PARAMS = [_policy_param(p) for p in SUPPORTED_POLICIES]


def _vla_cpu_skip(policy_name: str) -> None:
    """Skip VLA policies on hosts without an accelerator.

    VLA policies (pi0/pi05/pi0_fast/groot) require a GPU + bf16-mixed precision
    because the underlying transformer (PaliGemma / SmolVLM-2 / Eagle2) is
    too large for CPU inference. Without this skip the test would either OOM
    or hang for tens of minutes per parametrized invocation. Both CUDA and
    Intel XPU are accepted; CPU-only hosts skip.
    """
    if policy_name not in _VLA_POLICIES:
        return
    if get_available_device() == "cpu":
        pytest.skip(f"{policy_name} requires CUDA or XPU (bf16 transformer too large for CPU)")


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


def _ensure_quantile_stats(dataset: Any) -> None:
    """Add approximate q01/q99 stats derived from min/max when missing.

    PI05 uses QUANTILES normalization which requires q01 and q99 statistics.
    Older datasets (e.g. aloha_sim_insertion_human) predate this feature.
    For equivalence testing the exact quantile values don't matter — only
    that wrapper and native receive identical normalization — so min/max
    are a safe stand-in.
    """
    if all("q01" in s for s in dataset.meta.stats.values()):
        return
    for feature_stats in dataset.meta.stats.values():
        if "min" in feature_stats and "q01" not in feature_stats:
            min_val = feature_stats["min"]
            max_val = feature_stats["max"]
            feature_stats["q01"] = min_val.copy() if hasattr(min_val, "copy") else min_val
            feature_stats["q99"] = max_val.copy() if hasattr(max_val, "copy") else max_val


@pytest.fixture(scope="module")
def aloha_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(DATASET_REPO_ID)
    _ensure_quantile_stats(ds)
    return ds


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
    device = get_device()
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
    device_type = next(native.parameters()).device.type
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
    device = get_device()

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

    # Free accelerator memory before loading native model.
    wrapper.cpu()
    del trainer
    _empty_accelerator_cache(device)

    # 3. Build native model from saved initial weights (on accelerator).
    native = _make_native_from_state_dict(policy_name, dataset, initial_state_dict, device)
    del initial_state_dict
    _empty_accelerator_cache(device)

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


# ---------------------------------------------------------------------------- #
# Tier 1: Fast-dev-run — single step, all policies, CI
# ---------------------------------------------------------------------------- #
class TestFastDevRunEquivalence:
    @pytest.fixture(params=ALL_POLICIES_PARAMS)
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        name = str(request.param)
        _vla_cpu_skip(name)
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


# ---------------------------------------------------------------------------- #
# Tier 2: Multi-step (10 steps) — all policies, CI
# ---------------------------------------------------------------------------- #
class TestMultiStepTrainerEquivalence:
    NUM_STEPS = 10

    @pytest.fixture(params=ALL_POLICIES_PARAMS)
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        name = str(request.param)
        _vla_cpu_skip(name)
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


# ---------------------------------------------------------------------------- #
# Tier 2b: Gradient equivalence — first-step grad check (all policies, CI).
# Catches optimizer-independent wrapper bugs that loss-only checks miss.
# ---------------------------------------------------------------------------- #
class TestGradientEquivalence:
    @pytest.fixture(params=ALL_POLICIES_PARAMS)
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        name = str(request.param)
        _vla_cpu_skip(name)
        return name

    def test_first_step_gradients_match(self, policy_name: str, aloha_dataset: Any) -> None:
        wrapper, native, _native_optimizer, _grad_clip, device = _make_wrapper_and_native(policy_name, aloha_dataset)
        dm = _make_datamodule(policy_name)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        wrapper = wrapper.to(device)
        wrapper.train()

        torch.manual_seed(0)
        wrapper_loss, _ = wrapper(_clone_batch(batch))
        wrapper_loss.backward()
        wrapper_grads = {
            n: p.grad.detach().cpu().clone()
            for n, p in wrapper.lerobot_policy.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        torch.manual_seed(0)
        preprocessed = wrapper._preprocessor(_clone_batch(batch))
        native_out = native(preprocessed)
        native_loss = native_out[0] if isinstance(native_out, tuple) else native_out
        native_loss.backward()
        native_grads = {
            n: p.grad.detach().cpu().clone()
            for n, p in native.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        assert set(wrapper_grads) == set(native_grads), (
            f"Gradient parameter sets differ: "
            f"wrapper-only={set(wrapper_grads) - set(native_grads)}, "
            f"native-only={set(native_grads) - set(wrapper_grads)}"
        )
        for name in wrapper_grads:
            torch.testing.assert_close(
                wrapper_grads[name],
                native_grads[name],
                rtol=1e-5,
                atol=1e-5,
                msg=lambda m, n=name: f"Gradient mismatch on {n}: {m}",
            )


# ---------------------------------------------------------------------------- #
# Tier 2c: Weight equivalence after N optimizer steps (all policies, CI).
# Catches optimizer-state bugs invisible to loss-trajectory comparisons.
# ---------------------------------------------------------------------------- #
class TestWeightEquivalenceAfterTraining:
    NUM_STEPS = 10

    @pytest.fixture(params=ALL_POLICIES_PARAMS)
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        name = str(request.param)
        _vla_cpu_skip(name)
        return name

    def test_final_weights_match(self, policy_name: str, aloha_dataset: Any) -> None:
        """Per-parameter weight match after 10 trainer steps.

        Tolerance: ``rtol=atol=5e-5``. Justification: per-step fp32 rounding
        in Lightning's optimizer/scaler path is ~1.5e-6; 10 steps accumulates
        to ~1.5e-5 worst-case absolute drift on dense ResNet weights (measured
        on ACT). ``5e-5`` gives ~3x headroom over the empirical max while
        still catching any *new* drift source (e.g. an optimizer-state bug
        that diverges after the first step). Tier-1/2 loss checks use ``1e-5``
        because direct forward avoids accumulator paths; this tier exercises
        the full optimizer cycle and warrants the slightly looser bound.
        """
        _, _, wrapper, native = _run_trainer_capture_and_native_replay(
            policy_name,
            aloha_dataset,
            base_seed=303,
            max_steps=self.NUM_STEPS,
        )

        wrapper_params = {n: p.detach().cpu() for n, p in wrapper.lerobot_policy.named_parameters()}
        native_params = {n: p.detach().cpu() for n, p in native.named_parameters()}
        assert set(wrapper_params) == set(native_params)

        for name in wrapper_params:
            torch.testing.assert_close(
                wrapper_params[name],
                native_params[name],
                rtol=5e-5,
                atol=5e-5,
                msg=lambda m, n=name: f"Weight drift on {n} after {self.NUM_STEPS} steps: {m}",
            )


# ---------------------------------------------------------------------------- #
# Tier 3: Regression (50 steps) — all policies, nightly/@slow
# ---------------------------------------------------------------------------- #
@pytest.mark.slow
class TestRegressionTraining:
    NUM_STEPS = 50

    @pytest.fixture(params=ALL_POLICIES_PARAMS)
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        name = str(request.param)
        _vla_cpu_skip(name)
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
        """50-step trainer-driven loss trajectory match.

        Tolerance: ``rtol=atol=1e-3`` (vs ``1e-5`` at tiers 1/2). Justification:
        Lightning Trainer drives forward/backward through autograd graphs that
        accumulate fp32 rounding (~1e-6 per step on RTX A6000). VLA policies
        run under bf16-mixed precision where per-step drift is closer to ~1e-4.
        Over 50 steps the worst-case bound is ~5e-3; ``1e-3`` keeps that bound
        as the failure floor, surfacing any *new* drift source (numeric bug,
        kernel divergence, dtype regression) while tolerating the ambient
        non-determinism Lightning + bf16 introduce. Tightening below 5e-4
        previously caused intermittent failures in pre-merge runs.
        """
        wrapper_losses, native_losses, _, _ = _run_trainer_capture_and_native_replay(
            policy_name,
            aloha_dataset,
            base_seed=505,
            max_steps=self.NUM_STEPS,
        )

        assert len(wrapper_losses) == self.NUM_STEPS
        _assert_trajectories_match(wrapper_losses, native_losses, rtol=1e-3, atol=1e-3)
