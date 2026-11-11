# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, SLF001, PLR2004

"""SnapFlow one-step distillation for VLA-JEPA."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightning
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from physicalai.data import Observation
from physicalai.inference import InferenceModel
from physicalai.policies.vla_jepa import VLAJEPA, VLAJEPAConfig, VLAJEPAModel


def test_enable_snapflow_starts_from_teacher_and_samples_in_one_step(
    jepa_config: VLAJEPAConfig,
    video_observation: Observation,
) -> None:
    """Phase 2 freezes Qwen and JEPA, keeps the teacher's velocity, and takes one Euler step."""
    policy = VLAJEPA.from_config(jepa_config)
    policy.enable_snapflow(alpha=0.5, lambda_=0.1, num_inference_steps=1)
    model = policy.model
    assert policy.config.snapflow_enabled
    assert policy.hparams["snapflow_enabled"]
    assert not any(parameter.requires_grad for parameter in model.qwen.parameters())
    assert not any(parameter.requires_grad for parameter in model.video_predictor.parameters())
    assert all(parameter.requires_grad for parameter in model.action_model.parameters())

    policy.train()
    loss, metrics = policy(video_observation)
    assert metrics["world_model_loss"] == 0
    loss.backward()
    assert model.action_model.model.target_time_encoder.timestep_embedder.linear_2.weight.grad is not None

    head = model.action_model.eval()
    context = {"tokens": torch.rand(2, 2, 32), "state": torch.rand(2, 1, 2)}
    noise = torch.randn(2, 3, 3)
    zero = torch.zeros(2)
    teacher = head.denoise(noise, zero, context)
    torch.testing.assert_close(head.denoise(noise, zero, {**context, "target_time": torch.ones(2)}), teacher)
    torch.testing.assert_close(model.sample_actions(context, noise), noise + teacher)


def test_snapflow_translates_the_shared_time_convention(tiny_config: VLAJEPAConfig) -> None:
    """The FM branch equals plain training, and the shortcut targets noise-to-data in this head's time."""
    model = VLAJEPAModel.from_config(tiny_config).eval()  # no dropout, so RNG order does not matter
    head = model.action_model
    context = {"tokens": torch.rand(4, 2, 32), "state": torch.rand(4, 1, 2)}
    actions = torch.rand(4, 3, 3)
    torch.manual_seed(0)
    expected = head.compute_loss(actions, context)
    model._snapflow_alpha = 1.0
    torch.manual_seed(0)
    losses, cd_idx = model._snapflow_loss(actions, context)
    assert cd_idx.numel() == 0
    torch.testing.assert_close(losses, expected)

    model._snapflow_alpha = 0.0
    calls = []
    denoise = head.denoise

    def record(x_t: torch.Tensor, t: torch.Tensor, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        calls.append((t[0].item(), cond["target_time"][0].item()))
        return denoise(x_t, t, cond)

    with patch.object(head, "denoise", side_effect=record):
        _, cd_idx = model._snapflow_loss(actions, context)
    assert cd_idx.numel() == 4
    assert calls == [(0.0, 0.0), (0.5, 0.5), (0.0, 1.0)]


def test_phase_one_checkpoint_distills_after_reload(
    tiny_config: VLAJEPAConfig,
    observation: Observation,
    tmp_path: Path,
) -> None:
    """A checkpoint without the target-time embedding reloads straight into SnapFlow mode."""
    policy = VLAJEPA.from_config(tiny_config, gradient_checkpointing=False)
    trainer = lightning.Trainer(
        accelerator="cpu",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    with patch.object(policy, "setup"):
        trainer.fit(policy, train_dataloaders=DataLoader([observation], batch_size=None))
    path = tmp_path / "phase1.ckpt"
    trainer.save_checkpoint(path, weights_only=True)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["state_dict"] = {
        name: value for name, value in checkpoint["state_dict"].items() if ".target_time_encoder." not in name
    }
    torch.save(checkpoint, path)

    restored = VLAJEPA.load_from_checkpoint(path, weights_only=False, snapflow_enabled=True)
    assert restored.config == replace(policy.config, snapflow_enabled=True)
    assert restored.model._snapflow_enabled
    assert not any(parameter.requires_grad for parameter in restored.model.qwen.parameters())
    policy.eval()
    restored.eval()
    noise = torch.zeros(2, 3, 3)
    batch = policy._prepare_batch(observation)
    with torch.no_grad():
        teacher = noise + policy.model.action_model.denoise(noise, torch.zeros(2), policy.model._context(batch))
    torch.testing.assert_close(restored.model.predict_action_chunk(batch, noise=noise), teacher, atol=1e-5, rtol=1e-5)


def test_torch_export_keeps_one_step_sampler(
    tiny_config: VLAJEPAConfig,
    observation: Observation,
    tmp_path: Path,
) -> None:
    """Runtime reproduces the distilled one-step, target-time-conditioned actions."""
    policy = VLAJEPA.from_config(replace(tiny_config, snapflow_enabled=True)).eval()
    with torch.no_grad():
        for parameter in policy.model.action_model.model.target_time_encoder.parameters():
            parameter.normal_(std=0.1)
    torch.manual_seed(5)
    expected = policy.predict_action_chunk(observation)
    policy.export(tmp_path, backend="torch")
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained", side_effect=AssertionError("network")):
        runtime = InferenceModel(tmp_path, backend="torch", device="cpu")
    inputs = {
        "images.front": observation.images["front"].numpy(),
        "state": observation.state.numpy(),
        "task": observation.task,
    }
    torch.manual_seed(5)
    np.testing.assert_allclose(runtime(inputs)["action"], expected[:, :2].numpy(), rtol=1e-5, atol=1e-5)


def test_phase_two_schedule_spans_the_remaining_steps(tiny_config: VLAJEPAConfig) -> None:
    """Rebuilding at the boundary warms up again and reaches the head's floor at the end of the run."""
    policy = VLAJEPA.from_config(tiny_config, scheduler_warmup_steps=4)
    policy.enable_snapflow()
    policy._trainer = SimpleNamespace(estimated_stepping_batches=60, global_step=36)
    optimizers = policy.configure_optimizers()
    optimizer, scheduler = optimizers["optimizer"], optimizers["lr_scheduler"]["scheduler"]
    assert len(optimizer.param_groups) == 1
    lrs = []
    for _ in range(24):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[4] == pytest.approx(1e-4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 * 1e-6 / 1e-5)
