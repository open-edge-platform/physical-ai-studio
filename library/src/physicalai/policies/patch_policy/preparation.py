# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""One-shot PATCH training preparation within a normal Lightning fit."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder
from lightning.pytorch.strategies import SingleDeviceStrategy
from tqdm.auto import tqdm

from .components.heads import BehaviourTransformerHead

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .policy import PatchPolicy

logger = logging.getLogger(__name__)


def fit_tokenizer(policy: PatchPolicy) -> None:
    """Fit a fresh VQ tokenizer from training actions; leave diffusion/resumed models alone.

    Raises:
        ValueError: If the trainer, precision, dataset, or action windows are unsupported.
        RuntimeError: If tokenizer fitting produces a non-finite loss.
    """
    head = policy.model.action_head
    if not isinstance(head, BehaviourTransformerHead) or bool(head.quantizer.fitted):
        return
    trainer = policy.trainer
    if not isinstance(trainer.strategy, SingleDeviceStrategy) or trainer.world_size != 1:
        msg = "PATCH tokenizer fitting currently supports single-device training only."
        raise ValueError(msg)
    if any(isinstance(callback, (BatchSizeFinder, LearningRateFinder)) for callback in trainer.callbacks):
        msg = "PATCH tokenizer fitting does not support batch-size or learning-rate tuning."
        raise ValueError(msg)
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in head.quantizer.parameters()
    ):
        msg = "PATCH tokenizer fitting requires FP32 parameters; use 32-true or mixed precision."
        raise ValueError(msg)
    data = trainer.datamodule
    if data is None or getattr(data, "train_dataset", None) is None:
        msg = "PATCH tokenizer fitting requires a datamodule with a train_dataset."
        raise ValueError(msg)
    dataset = data.train_dataset
    config = policy.config
    # Fit a copy so a failed preparation cannot partially update the registered tokenizer.
    tokenizer = copy.deepcopy(head.quantizer).float().requires_grad_(requires_grad=True).train()
    actions = []
    for target in tqdm(
        _action_targets(dataset), total=len(dataset), desc="Collecting tokenizer actions", unit="sample",
    ):
        expected = (config.n_obs_steps + config.chunk_size - 1, config.action_dim)
        if not isinstance(target, torch.Tensor) or target.shape != expected:
            msg = f"Tokenizer preparation requires action targets shaped {expected}."
            raise ValueError(msg)
        actions.append(target.detach().cpu().unfold(0, config.chunk_size, 1).transpose(-1, -2).flatten(1))
    if not actions:
        msg = "Cannot fit the action tokenizer on an empty training dataset."
        raise ValueError(msg)
    # Upstream fits unique overlapping windows, including episode-boundary padding.
    windows = torch.cat(actions).unique(dim=0).reshape(-1, config.chunk_size, config.action_dim)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3, weight_decay=1e-4)
    logger.info("Fitting PATCH tokenizer: %d unique windows, %d full passes", len(windows), config.vqvae_iters)
    with (
        torch.enable_grad(),
        torch.autocast(device_type=policy.device.type, enabled=False),
        tqdm(total=config.vqvae_iters, desc="Fitting action tokenizer", unit="pass") as progress,
    ):
        for _ in range(config.vqvae_iters):
            order = torch.randperm(len(windows))
            for indices in order.split(config.vqvae_batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = tokenizer.compute_loss(windows[indices].to(policy.device, dtype=torch.float32))
                if not torch.isfinite(loss):
                    msg = "PATCH tokenizer fitting produced a non-finite loss."
                    raise RuntimeError(msg)
                loss.backward()
                optimizer.step()
            progress.set_postfix(loss=float(loss.detach()), refresh=False)
            progress.update()
    tokenizer.fitted.fill_(value=True)
    head.quantizer.load_state_dict(tokenizer.state_dict())
    head.quantizer.requires_grad_(requires_grad=False).eval()


def _action_targets(dataset: Any) -> Iterator[torch.Tensor]:  # noqa: ANN401
    """Read LeRobot action windows without decoding any visual observations.

    Yields:
        Unnormalized action targets with the configured temporal offsets.

    Raises:
        ValueError: If an action window leaves the training split.
    """
    source = getattr(dataset, "_lerobot_dataset", None)
    if source is None:
        for index in range(len(dataset)):
            yield dataset[index].action
        return
    keys = getattr(dataset, "_action_columns", None) or ["action"]
    table = source.hf_dataset.select_columns(["index", "episode_index", *keys]).with_format("torch")
    absolute = np.asarray(list(table["index"]), dtype=np.int64)
    for row in table:
        episode = source.meta.episodes[int(row["episode_index"])]
        values = {}
        for key in keys:
            targets = np.clip(
                int(row["index"]) + np.asarray(dataset.delta_indices[key]),
                episode["dataset_from_index"],
                episode["dataset_to_index"] - 1,
            )
            positions = np.searchsorted(absolute, targets)
            if np.any(positions >= len(absolute)) or np.any(absolute[positions] != targets):
                msg = "Tokenizer action windows reference frames outside the training split."
                raise ValueError(msg)
            values[key] = table[positions.tolist()][key]
        yield dataset._combine_split_columns(values)["action"]  # noqa: SLF001
