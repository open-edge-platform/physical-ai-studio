# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LIBERO training recipe wiring without datasets, weights, or a simulator."""

from collections.abc import Callable
from pathlib import Path

import torch
import yaml
from jsonargparse import ArgumentParser


def test_libero_recipe_augmentation_and_optimizer_step_clock() -> None:
    path = Path(__file__).parents[4] / "configs/physicalai/molmoact2/libero.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = config["data"]["init_args"]
    trainer = config["trainer"]
    assert data["train_batch_size"] * trainer["accumulate_grad_batches"] == 256
    assert config["model"]["init_args"]["preserve_pretrained_normalization_in_training"]
    assert "val_gym" not in data
    assert trainer["limit_val_batches"] == 0
    model = config["model"]["init_args"]
    assert model["optimizer_lr"] == 1e-5
    assert model["optimizer_vit_lr"] == model["optimizer_connector_lr"] == 5e-6
    assert model["optimizer_action_expert_lr"] == 5e-5
    assert model["scheduler_warmup_steps"] == 200
    assert model["scheduler_decay_steps"] == 24_000
    assert model["scheduler_decay_lr"] == 1e-6
    checkpoint = trainer["callbacks"][0]["init_args"]
    assert checkpoint["every_n_train_steps"] == 2_000
    assert "monitor" not in checkpoint

    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--image_transforms", type=Callable)
    parsed = parser.parse_object({"image_transforms": data["image_transforms"]})
    transforms = parser.instantiate(parsed).image_transforms
    image = transforms(torch.rand(3, 256, 256))
    assert image.shape == (3, 256, 256)
    assert image.dtype == torch.float32
