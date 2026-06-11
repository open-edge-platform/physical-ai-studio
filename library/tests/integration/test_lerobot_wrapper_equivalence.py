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
    "pi05": "model repo is gated",
    "pi0_fast": "model repo is gated",
    "pi0": "model repo is gated",
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


# ---------------------------------------------------------------------------- #
# Inference equivalence — wrapper vs direct LeRobot calls (synthetic batches).
# Moved from the unit suite: these exercise select_action / predict_action_chunk
# against a real ACT policy built from a real dataset, so they belong here.
# ---------------------------------------------------------------------------- #
def _make_training_batch(config: Any, device: torch.device) -> dict[str, torch.Tensor]:
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


@pytest.fixture(scope="module")
def pusht_dataset() -> Any:
    """Load pusht dataset once per module for inference-equivalence tests."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset("lerobot/pusht")


class TestLeRobotPolicyNumericalEquivalence:
    """Tests verifying wrapper produces identical results to direct LeRobot calls.

    These tests ensure our wrapper's predict_action_chunk and select_action
    produce numerically equivalent outputs to calling the underlying LeRobot
    policy methods directly.

    Uses synthetic mock data to avoid FFmpeg/torchcodec dependency during inference.
    """

    @pytest.fixture
    def policy_and_batch(self, pusht_dataset: Any) -> tuple[LeRobotPolicy, dict[str, torch.Tensor]]:
        """Create policy and matching synthetic batch on same device.

        Uses ``_make_training_batch`` so the shapes are correct for both
        inference *and* training (ACT's VAE encoder requires temporal dims).
        """
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

    def test_multiple_select_action_uses_cached_chunk(self, pusht_dataset: Any) -> None:
        """Verify select_action uses internal queue correctly."""
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


_TRAINING_POLICY_NAMES = ["act", "diffusion", "smolvla"]
"""Policies validated end-to-end on the accelerator with the pusht dataset fixture.

Excluded with reason:
- ``vlas`` (pi0/pi05/pi0_fast/groot): require GPU + bf16; covered by the Trainer tiers above.
- ``xvla``: requires explicit ``vision_config`` kwarg, not derivable from the dataset.
"""


class TestTrainingForwardContract:
    """Contract checks for a single wrapper.forward on the accelerator with synthetic batches.

    Complements the Trainer-driven equivalence tiers above. These verify the
    structure of the wrapper's forward output (loss/loss_dict shape, gradient
    flow, optimizer effect, preprocessing) without spinning up a Lightning
    Trainer. Synthetic batches keep them FFmpeg/torchcodec-free.
    """

    @pytest.fixture(params=_TRAINING_POLICY_NAMES)
    def training_policy_and_batch(self, request, pusht_dataset):
        policy_name = request.param

        device = get_device()

        # Let LeRobot autodetect the accelerator for the config, then move the
        # wrapper onto it so weights and the synthetic batch share one device.
        policy = LeRobotPolicy.from_dataset(policy_name, pusht_dataset)
        policy = policy.to(device)
        policy.train()

        batch = _make_training_batch(policy._config, device)

        return policy, batch, policy_name

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
