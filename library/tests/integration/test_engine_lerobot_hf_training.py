# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in configurable training runs against public LeRobot datasets."""

from __future__ import annotations

import json
import math
import os
import random
import sys
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

_DEFAULT_STEPS = 1
_STEP_ENV = "PHYSICALAI_LEROBOT_TEST_STEPS"
_DEVICE_ENV = "PHYSICALAI_LEROBOT_DEVICE"
_ALLOWED_DEVICES = {"cpu", "cuda", "xpu"}

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

    device = os.environ.get(_DEVICE_ENV, "cpu").strip().lower()
    if device not in _ALLOWED_DEVICES:
        msg = f"{_DEVICE_ENV} must be one of {sorted(_ALLOWED_DEVICES)}, got {device!r}."
        raise ValueError(msg)
    return steps, device


def _require_training_device(device: str) -> None:
    """Skip an explicitly requested accelerator when PyTorch cannot use it."""
    if device == "cpu":
        return

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(f"{_DEVICE_ENV}=cuda was requested, but CUDA is unavailable in this PyTorch environment.")
    xpu_backend = getattr(torch, "xpu", None)
    if device == "xpu" and (
        xpu_backend is None
        or not callable(getattr(xpu_backend, "is_available", None))
        or not xpu_backend.is_available()
    ):
        pytest.skip(f"{_DEVICE_ENV}=xpu was requested, but Intel XPU is unavailable in this PyTorch environment.")


def _make_training_config(
    policy_type: str,
    repo_id: str,
    revision: str,
    output_dir: Path,
    *,
    steps: int,
    device: str,
) -> object:
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


def test_training_options_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow developers to extend the smoke run and select its training device."""
    monkeypatch.setenv(_STEP_ENV, "25")
    monkeypatch.setenv(_DEVICE_ENV, "cuda")

    assert _training_options() == (25, "cuda")


def test_training_options_default_to_one_cpu_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ordinary opt-in runs small and hardware-independent."""
    monkeypatch.delenv(_STEP_ENV, raising=False)
    monkeypatch.delenv(_DEVICE_ENV, raising=False)

    assert _training_options() == (1, "cpu")


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
def test_unavailable_accelerator_request_skips_with_reason(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Explain why an unavailable accelerator cannot run this test."""
    import sys
    from types import SimpleNamespace

    backend = SimpleNamespace(is_available=lambda: False)
    fake_torch = SimpleNamespace(cuda=backend, xpu=backend)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(pytest.skip.Exception, match="PHYSICALAI_LEROBOT_DEVICE=.*unavailable"):
        _require_training_device(device)


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

    Set ``PHYSICALAI_RUN_LEROBOT_INTEGRATION=1`` to run the cases. The defaults
    are one CPU update; set ``PHYSICALAI_LEROBOT_TEST_STEPS`` and
    ``PHYSICALAI_LEROBOT_DEVICE`` for longer or accelerator-backed runs. The
    pinned public dataset may be downloaded on first use.
    """
    if os.environ.get("PHYSICALAI_RUN_LEROBOT_INTEGRATION") != "1":
        pytest.skip("Set PHYSICALAI_RUN_LEROBOT_INTEGRATION=1 to run public-dataset training tests.")

    from physicalai.engine import Engine
    from physicalai.inference.manifest import Manifest

    requested_steps, device = _training_options()
    _require_training_device(device)
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
    run = Engine().train(config, hooks=hooks)

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

    package = Engine().export(run, output_dir=tmp_path / "export")
    artifact = package / f"{policy_type}.lerobot" / "pretrained_model"
    assert artifact.is_dir()
    assert (artifact / "train_config.json").is_file()
    assert Manifest.load(package).model.artifacts == {"lerobot": f"{policy_type}.lerobot"}


@pytest.mark.integration
@pytest.mark.requires_download
def test_native_lerobot_training_matches_engine_on_pinned_pusht(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    """Compare native and Engine training traces/configs/weights on the same data."""
    if os.environ.get("PHYSICALAI_RUN_LEROBOT_INTEGRATION") != "1":
        pytest.skip("Set PHYSICALAI_RUN_LEROBOT_INTEGRATION=1 to run training parity tests.")

    import numpy as np
    import torch
    from lerobot.scripts import lerobot_train
    from lerobot.utils.import_utils import register_third_party_plugins
    from safetensors.torch import load_file
    from physicalai.engine import Engine

    requested_steps, device = _training_options()
    _require_training_device(device)
    dataset_repo = "lerobot/pusht"
    dataset_revision = "b1c3ecbae7f244acc039a3dbc255a00dad1372b9"
    native_output = tmp_path / "native-output"
    engine_output = tmp_path / "engine-output"
    native_config = _make_training_config(
        "act", dataset_repo, dataset_revision, native_output, steps=requested_steps, device=device
    )
    engine_config = _make_training_config(
        "act", dataset_repo, dataset_revision, engine_output, steps=requested_steps, device=device
    )
    assert native_config is not engine_config
    assert native_output != engine_output

    native_losses: list[float] = []
    phase = "native"
    original_update = cast("Callable[..., tuple[object, object]]", lerobot_train.update_policy)

    def record_native_update(*args: object, **kwargs: object) -> tuple[object, object]:
        result = original_update(*args, **kwargs)
        if phase == "native":
            tracker = result[0]
            to_dict = getattr(tracker, "to_dict")
            native_losses.append(float(to_dict(use_avg=False)["loss"]))
        return result

    monkeypatch.setattr(lerobot_train, "update_policy", record_native_update)
    _reset_training_seed(42, device)
    original_argv = sys.argv
    try:
        sys.argv = ["lerobot-train"]
        register_third_party_plugins()
        lerobot_train.train(native_config)
    finally:
        sys.argv = original_argv

    phase = "engine"
    hooks = _RecordingHooks()
    _reset_training_seed(42, device)
    run = Engine().train(engine_config, hooks=hooks)

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

    rtol, atol = _training_parity_tolerances(device)
    np.testing.assert_allclose(engine_losses, native_losses, rtol=rtol, atol=atol)
    native_weights = load_file(str(native_checkpoint / "model.safetensors"), device="cpu")
    engine_weights = load_file(str(run.checkpoint / "model.safetensors"), device="cpu")
    assert engine_weights.keys() == native_weights.keys()
    for name, native_weight in native_weights.items():
        torch.testing.assert_close(engine_weights[name], native_weight, rtol=rtol, atol=atol, msg=name)

    summary = {
        "steps": requested_steps,
        "device": device,
        "losses": {"native": native_losses, "engine": engine_losses},
        "rtol": rtol,
        "atol": atol,
    }
    trajectory = json.dumps(summary, sort_keys=True)
    (tmp_path / "native-engine-training-parity.json").write_text(trajectory + "\n", encoding="utf-8")
    record_property("native_engine_training_parity", trajectory)
    print(f"Native/Engine training parity ({requested_steps} steps, {device}): {trajectory}")


def _reset_training_seed(seed: int, device: str) -> None:
    """Reset host and requested accelerator RNGs before a comparison run."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device == "xpu":
        torch.xpu.manual_seed_all(seed)


def _training_parity_tolerances(device: str) -> tuple[float, float]:
    """Return tolerances for repeated LeRobot optimization on one backend.

    Returns:
        Relative and absolute tolerances. CPU reductions are nearly exact;
        accelerator kernels allow small float32 implementation differences.
    """
    return (1e-6, 1e-7) if device == "cpu" else (5e-4, 5e-5)
