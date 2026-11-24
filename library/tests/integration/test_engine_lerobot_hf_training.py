# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in configurable training runs against public LeRobot datasets."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tests.integration import _device

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lerobot.configs.train import TrainPipelineConfig

_DEFAULT_STEPS = 1
_STEP_ENV = "PHYSICALAI_LEROBOT_TEST_STEPS"
_DEVICE_ENV = "PHYSICALAI_LEROBOT_DEVICE"
_ALLOWED_DEVICES = {"auto", "cpu", "cuda", "xpu"}
_NATIVE_TRAINER = Path(__file__).with_name("_native_lerobot_train.py")

_DATASETS = [
    pytest.param(
        "act",
        "lerobot/pusht",
        "b1c3ecbae7f244acc039a3dbc255a00dad1372b9",
        id="act-pusht",
    ),
    pytest.param(
        "diffusion",
        "lerobot/pusht",
        "b1c3ecbae7f244acc039a3dbc255a00dad1372b9",
        id="diffusion-pusht",
    ),
    pytest.param(
        "act",
        "lerobot/aloha_sim_insertion_human",
        "a3565744bd671cfe9acfb456de0ea4da40b6f55e",
        id="act-aloha-insertion",
    ),
]


def _training_options() -> tuple[int, str]:
    """Read and validate adjustable training-test settings from the environment.

    Returns:
        Requested optimization steps and device.

    Raises:
        RuntimeError: If the selected GPU backend is unusable by host PyTorch.
        ValueError: If a test option is invalid.
    """
    raw_steps = os.environ.get(_STEP_ENV, str(_DEFAULT_STEPS))
    try:
        steps = int(raw_steps)
    except ValueError as error:
        msg = f"{_STEP_ENV} must be a positive integer, got {raw_steps!r}."
        raise ValueError(msg) from error
    if steps < 1:
        msg = f"{_STEP_ENV} must be a positive integer, got {steps}."
        raise ValueError(msg)

    requested_device = os.environ.get(_DEVICE_ENV, "auto").strip().lower()
    if requested_device not in _ALLOWED_DEVICES:
        msg = f"{_DEVICE_ENV} must be one of {sorted(_ALLOWED_DEVICES)}, got {requested_device!r}."
        raise ValueError(msg)
    return steps, _device.select_lerobot_test_device(requested_device)


def _make_training_config(
    policy_type: str,
    repo_id: str,
    revision: str,
    output_dir: Path,
    *,
    steps: int,
    device: str,
    cudnn_deterministic: bool = False,
) -> TrainPipelineConfig:
    """Build a small native LeRobot config for one pinned public dataset.

    Returns:
        A native train config with the requested step count and device.
    """
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    if policy_type == "act":
        policy = ACTConfig(
            push_to_hub=False,
            device=device,
            pretrained_backbone_weights=None,
            chunk_size=8,
            n_action_steps=8,
            dim_model=64,
            n_heads=4,
            dim_feedforward=128,
            n_encoder_layers=1,
            n_decoder_layers=1,
            use_vae=False,
        )
    else:
        policy = DiffusionConfig(
            push_to_hub=False,
            device=device,
            pretrained_backbone_weights=None,
            horizon=16,
            n_action_steps=8,
            down_dims=(32, 64, 128),
            diffusion_step_embed_dim=32,
            num_train_timesteps=10,
            num_inference_steps=2,
        )

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=repo_id,
            revision=revision,
            episodes=[0],
            eval_split=0.0,
        ),
        policy=policy,
        output_dir=output_dir,
        seed=42,
        steps=steps,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        env_eval_freq=0,
        eval_steps=0,
        log_freq=steps,
        cudnn_deterministic=cudnn_deterministic,
        save_checkpoint=True,
        save_freq=steps,
    )


class _RecordingHooks:
    """Collect per-step progress from a native LeRobot training run."""

    def __init__(self) -> None:
        self.steps: list[tuple[int, int | None, dict[str, float]]] = []

    def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None:
        self.steps.append((step, total, dict(metrics)))

    @staticmethod
    def should_stop() -> bool:
        return False


def _assert_training_trajectory(
    hooks: _RecordingHooks,
    requested_steps: int,
    *,
    record_property: Callable[[str, object], None],
    output_path: Path,
) -> list[float]:
    """Assert update count and finite metrics, then persist a readable trajectory.

    Returns:
        Per-step scalar losses in update order.
    """
    assert [step for step, _, _ in hooks.steps] == list(range(1, requested_steps + 1))
    assert all(total == requested_steps for _, total, _ in hooks.steps)
    assert all("loss" in metrics for _, _, metrics in hooks.steps)
    assert all(math.isfinite(value) for _, _, metrics in hooks.steps for value in metrics.values())

    summary = [{"step": step, "total": total, **metrics} for step, total, metrics in hooks.steps]
    trajectory = json.dumps(summary, sort_keys=True)
    output_path.write_text(trajectory + "\n", encoding="utf-8")
    record_property("training_trajectory", trajectory)
    print(f"Training trajectory ({output_path}): {trajectory}")
    return [metrics["loss"] for _, _, metrics in hooks.steps]


def test_native_parity_config_can_enable_cudnn_determinism(tmp_path: Path) -> None:
    """Use LeRobot's native cuDNN determinism setting for CUDA parity runs."""
    config = _make_training_config(
        "act",
        "lerobot/pusht",
        "b1c3ecbae7f244acc039a3dbc255a00dad1372b9",
        tmp_path / "run",
        steps=1,
        device="cuda",
        cudnn_deterministic=True,
    )

    assert config.cudnn_deterministic is True


def _mock_host_accelerators(
    monkeypatch: pytest.MonkeyPatch,
    *,
    auto_accelerator: str,
    cuda: bool,
    xpu: bool,
) -> None:
    def resolve_accelerator(requested: str) -> str:
        return auto_accelerator if requested == "auto" else {"cpu": "cpu", "cuda": "cuda", "xpu": "xpu"}[requested]

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        xpu=SimpleNamespace(is_available=lambda: xpu),
    )
    monkeypatch.setattr(_device, "resolve_accelerator", resolve_accelerator)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_training_options_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow developers to extend the smoke run and select its training device."""
    monkeypatch.setenv(_STEP_ENV, "25")
    monkeypatch.setenv(_DEVICE_ENV, "cuda")
    _mock_host_accelerators(monkeypatch, auto_accelerator="cuda", cuda=True, xpu=False)

    assert _training_options() == (25, "cuda")


def test_training_options_honor_explicit_cpu_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow developers to keep a GPU-equipped host on CPU intentionally."""
    monkeypatch.delenv(_STEP_ENV, raising=False)
    monkeypatch.setenv(_DEVICE_ENV, "cpu")
    _mock_host_accelerators(monkeypatch, auto_accelerator="cuda", cuda=False, xpu=False)

    assert _training_options() == (1, "cpu")


def test_training_options_auto_prefers_usable_xpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto mode chooses usable XPU before CUDA and doesn't default to CPU."""
    monkeypatch.delenv(_STEP_ENV, raising=False)
    monkeypatch.delenv(_DEVICE_ENV, raising=False)
    _mock_host_accelerators(monkeypatch, auto_accelerator="xpu", cuda=True, xpu=True)

    assert _training_options() == (1, "xpu")


def test_training_parity_is_exact_on_cpu_and_cuda_and_bounded_by_measured_noise_on_xpu() -> None:
    assert _training_parity_tolerances("cpu") == _training_parity_tolerances("cuda") == ((1e-6, 1e-7), (1e-6, 1e-7))
    loss_tolerances, weight_tolerances = _training_parity_tolerances("xpu")
    # Measured XPU run-to-run noise at 20 steps: about 2e-3 in loss (at losses near 1) and 1.1e-4 in weights.
    assert loss_tolerances == (1e-2, 1e-3)
    assert weight_tolerances == (1e-3, 5e-4)
    with pytest.raises(ValueError, match="measure its run-to-run noise"):
        _training_parity_tolerances("mps")


@pytest.mark.parametrize(("steps", "device"), [("0", "cpu"), ("many", "cpu"), ("1", "mps")])
def test_training_options_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    steps: str,
    device: str,
) -> None:
    """Reject nonpositive steps and unsupported test devices."""
    monkeypatch.setenv(_STEP_ENV, steps)
    monkeypatch.setenv(_DEVICE_ENV, device)

    with pytest.raises(ValueError, match="PHYSICALAI_LEROBOT_TEST_STEPS|PHYSICALAI_LEROBOT_DEVICE"):
        _training_options()


@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_unavailable_accelerator_request_fails_with_setup_guidance(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Fail clearly instead of silently selecting CPU when GPU use is unavailable."""
    monkeypatch.setenv(_DEVICE_ENV, device)
    _mock_host_accelerators(
        monkeypatch,
        auto_accelerator="xpu" if device == "xpu" else "cuda",
        cuda=False,
        xpu=False,
    )

    with pytest.raises(RuntimeError, match="explicitly requested.*device.*torch"):
        _training_options()


@pytest.mark.integration
@pytest.mark.requires_download
@pytest.mark.parametrize(("policy_type", "repo_id", "revision"), _DATASETS)
def test_engine_trains_public_lerobot_dataset_and_packages_checkpoint(
    policy_type: str,
    repo_id: str,
    revision: str,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    """Train a configurable seeded run from a public dataset and package its checkpoint.

    Set ``PHYSICALAI_RUN_LEROBOT_INTEGRATION=1`` to run the cases. The default
    budget is one update, and ``PHYSICALAI_LEROBOT_DEVICE=auto`` selects XPU,
    then CUDA, then CPU using core hardware detection. Set
    ``PHYSICALAI_LEROBOT_TEST_STEPS`` for a longer run and override the device
    explicitly only when intended. A detected but unusable GPU fails with setup
    guidance instead of falling back to CPU. The pinned dataset may be downloaded.
    """
    if os.environ.get("PHYSICALAI_RUN_LEROBOT_INTEGRATION") != "1":
        pytest.skip("Set PHYSICALAI_RUN_LEROBOT_INTEGRATION=1 to run public-dataset training tests.")

    from physicalai.inference.manifest import Manifest
    from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

    requested_steps, device = _training_options()
    output_dir = tmp_path / f"{policy_type}-{repo_id.rsplit('/', maxsplit=1)[-1]}"
    config = _make_training_config(
        policy_type,
        repo_id,
        revision,
        output_dir,
        steps=requested_steps,
        device=device,
    )
    hooks = _RecordingHooks()
    run = LeRobotEngine().train(config, hooks=hooks)

    assert run.output_dir == output_dir
    assert run.checkpoint is not None
    assert (run.checkpoint / "config.json").is_file()
    assert (run.checkpoint / "train_config.json").is_file()
    assert (run.checkpoint / "model.safetensors").is_file()
    _assert_training_trajectory(
        hooks,
        requested_steps,
        record_property=record_property,
        output_path=tmp_path / "training-trajectory.json",
    )

    package = LeRobotEngine.export(run, output_path=tmp_path / "export")
    artifact = package / f"{policy_type}.lerobot" / "pretrained_model"
    assert artifact.is_dir()
    assert (artifact / "train_config.json").is_file()
    assert Manifest.load(package).model.artifacts == {"lerobot": f"{policy_type}.lerobot"}


@pytest.mark.integration
@pytest.mark.requires_download
def test_native_lerobot_training_matches_engine_on_pinned_pusht(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    """Compare native and Engine training traces/configs/weights on the same data.

    Both runs use the LeRobot worker environment, so the comparison checks the Engine path
    (config serialization, worker protocol, progress events, cancellation listener) rather than
    differences between the host's and the worker's PyTorch builds. The native run starts
    LeRobot's trainer directly in that environment, the way the worker does.
    """
    if os.environ.get("PHYSICALAI_RUN_LEROBOT_INTEGRATION") != "1":
        pytest.skip("Set PHYSICALAI_RUN_LEROBOT_INTEGRATION=1 to run training parity tests.")

    import numpy as np
    import torch
    from physicalai import isolate
    from physicalai.engine import prepare_environment, resolve_accelerator
    from safetensors.torch import load_file
    from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

    requested_steps, device = _training_options()
    dataset_repo = "lerobot/pusht"
    dataset_revision = "b1c3ecbae7f244acc039a3dbc255a00dad1372b9"
    native_output = tmp_path / "native-output"
    engine_output = tmp_path / "engine-output"
    # Separate processes can select different cuDNN benchmark kernels unless this native flag is set.
    deterministic_cuda = device == "cuda"
    native_config = _make_training_config(
        "act",
        dataset_repo,
        dataset_revision,
        native_output,
        steps=requested_steps,
        device=device,
        cudnn_deterministic=deterministic_cuda,
    )
    engine_config = _make_training_config(
        "act",
        dataset_repo,
        dataset_revision,
        engine_output,
        steps=requested_steps,
        device=device,
        cudnn_deterministic=deterministic_cuda,
    )
    assert native_config is not engine_config
    assert native_output != engine_output

    environment = prepare_environment(LeRobotEngine.environment_for_policy("act"), resolve_accelerator(device))
    native_config_dir = tmp_path / "native-config"
    native_config.save_pretrained(native_config_dir)
    native_losses_path = tmp_path / "native-losses.json"
    native = isolate.run(
        environment.isolate,
        _NATIVE_TRAINER,
        [str(native_config_dir / "train_config.json"), str(native_losses_path)],
        check=False,
        capture_output=True,
    )
    assert native.returncode == 0, native.stderr.decode(errors="replace")[-4000:]
    native_losses = [float(loss) for loss in json.loads(native_losses_path.read_text(encoding="utf-8"))]

    hooks = _RecordingHooks()
    run = LeRobotEngine().train(engine_config, hooks=hooks)

    native_checkpoint = native_output / "checkpoints" / "last" / "pretrained_model"
    assert run.output_dir == engine_output
    assert run.checkpoint == engine_output / "checkpoints" / "last" / "pretrained_model"
    assert len(native_losses) == requested_steps
    engine_losses = _assert_training_trajectory(
        hooks,
        requested_steps,
        record_property=record_property,
        output_path=tmp_path / "engine-training-trajectory.json",
    )
    assert all(math.isfinite(loss) for loss in native_losses)

    native_saved_config = json.loads((native_checkpoint / "train_config.json").read_text(encoding="utf-8"))
    engine_saved_config = json.loads((run.checkpoint / "train_config.json").read_text(encoding="utf-8"))
    native_saved_config["output_dir"] = "<run-output>"
    engine_saved_config["output_dir"] = "<run-output>"
    assert engine_saved_config == native_saved_config

    (loss_rtol, loss_atol), (weight_rtol, weight_atol) = _training_parity_tolerances(device)
    native_weights = load_file(str(native_checkpoint / "model.safetensors"), device="cpu")
    engine_weights = load_file(str(run.checkpoint / "model.safetensors"), device="cpu")
    assert engine_weights.keys() == native_weights.keys()
    weight_differences = {
        name: float((engine_weights[name] - native_weight).abs().max())
        for name, native_weight in native_weights.items()
    }
    weight_mismatches = [
        name
        for name, native_weight in native_weights.items()
        if not torch.allclose(engine_weights[name], native_weight, rtol=weight_rtol, atol=weight_atol)
    ]
    loss_max_abs_difference = float(np.max(np.abs(np.asarray(engine_losses) - np.asarray(native_losses))))
    losses_match = bool(np.allclose(engine_losses, native_losses, rtol=loss_rtol, atol=loss_atol))
    # The first update sees the same weights and batch on both sides, so its loss must match exactly
    # on every device; only later steps may drift on a nondeterministic backend.
    first_loss_matches = math.isclose(engine_losses[0], native_losses[0], rel_tol=1e-6, abs_tol=1e-7)

    summary = {
        "steps": requested_steps,
        "device": device,
        "cudnn_deterministic": deterministic_cuda,
        "losses": {"native": native_losses, "engine": engine_losses},
        "first_loss_matches": first_loss_matches,
        "loss_max_abs_difference": loss_max_abs_difference,
        "weight_max_abs_difference": max(weight_differences.values()),
        "weight_mismatch_count": len(weight_mismatches),
        "loss_tolerances": {"rtol": loss_rtol, "atol": loss_atol},
        "weight_tolerances": {"rtol": weight_rtol, "atol": weight_atol},
    }
    trajectory = json.dumps(summary, sort_keys=True)
    record_property("native_engine_training_parity", trajectory)
    assert first_loss_matches and losses_match and not weight_mismatches, trajectory
    (tmp_path / "native-engine-training-parity.json").write_text(trajectory + "\n", encoding="utf-8")
    print(f"Native/Engine training parity ({requested_steps} steps, {device}): {trajectory}")


def _training_parity_tolerances(device: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return loss and weight tolerances for repeated LeRobot optimization on one backend.

    CPU and CUDA with deterministic cuDNN repeat a run exactly. XPU has no deterministic mode:
    two native ACT/PushT runs of 20 steps on one Intel GPU differed by up to 2e-3 in loss and
    1.1e-4 in weights, as much as the Engine run differs from a native one. Its tolerances sit
    above that run-to-run noise and were calibrated at 20 steps; drift grows with more steps.
    The first update's loss is compared exactly on every device.

    Returns:
        ``(rtol, atol)`` for the per-step losses, then for the final weights.

    Raises:
        ValueError: If no parity tolerances are defined for *device*.
    """
    if device in {"cpu", "cuda"}:
        return (1e-6, 1e-7), (1e-6, 1e-7)
    if device == "xpu":
        return (1e-2, 1e-3), (1e-3, 5e-4)
    msg = f"No training parity tolerances for device {device!r}; measure its run-to-run noise first."
    raise ValueError(msg)
