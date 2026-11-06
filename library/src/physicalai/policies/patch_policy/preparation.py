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

from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC2701

from .components.heads import BehaviourTransformerHead

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .components.heads.behavior_transformer import ResidualActionQuantizer
    from .config import PatchPolicyConfig
    from .policy import PatchPolicy

logger = logging.getLogger(__name__)


def fit_tokenizer(policy: PatchPolicy) -> None:
    """Fit a fresh VQ tokenizer from training actions; leave diffusion/resumed models alone.

    Raises:
        ValueError: If the model is not initialized.
    """
    if policy.model is None:
        msg = "PATCH tokenizer fitting requires an initialized model."
        raise ValueError(msg)
    head = policy.model.action_head
    if not isinstance(head, BehaviourTransformerHead):
        return
    if bool(head.quantizer.fitted):
        return

    dataset = _validate_preparation(policy, head.quantizer)

    # Only commit fitted weights after every preparation step succeeds.
    tokenizer = copy.deepcopy(head.quantizer).float().requires_grad_(requires_grad=True).train()
    windows = _collect_action_windows(dataset, policy.config)
    _train_tokenizer(
        tokenizer,
        windows,
        device=policy.device,
        num_passes=policy.vqvae_iters,
        batch_size=policy.vqvae_batch_size,
    )

    tokenizer.fitted.fill_(value=True)
    head.quantizer.load_state_dict(tokenizer.state_dict())
    head.quantizer.requires_grad_(requires_grad=False).eval()


def _validate_preparation(policy: PatchPolicy, tokenizer: ResidualActionQuantizer) -> Any:  # noqa: ANN401
    """Validate fitting requirements and return the training dataset.

    Returns:
        The trainer's action-window dataset.

    Raises:
        ValueError: If the trainer, precision, or dataset is unsupported.
    """
    trainer = policy.trainer
    if not isinstance(trainer.strategy, SingleDeviceStrategy) or trainer.world_size != 1:
        msg = "PATCH tokenizer fitting currently supports single-device training only."
        raise ValueError(msg)
    if any(isinstance(callback, (BatchSizeFinder, LearningRateFinder)) for callback in trainer.callbacks):
        msg = "PATCH tokenizer fitting does not support batch-size or learning-rate tuning."
        raise ValueError(msg)
    for parameter in tokenizer.parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            msg = "PATCH tokenizer fitting requires FP32 parameters; use 32-true or mixed precision."
            raise ValueError(msg)
    data = trainer.datamodule
    if data is None or getattr(data, "train_dataset", None) is None:
        msg = "PATCH tokenizer fitting requires a datamodule with a train_dataset."
        raise ValueError(msg)
    return data.train_dataset


def _collect_action_windows(dataset: Any, config: PatchPolicyConfig) -> torch.Tensor:  # noqa: ANN401
    """Collect unique overlapping action windows, including episode-boundary padding.

    Returns:
        CPU action windows shaped [windows, chunk_size, action_dim].

    Raises:
        ValueError: If action shapes are invalid or the dataset is empty.
    """
    expected = (config.n_obs_steps + config.chunk_size - 1, config.action_dim)
    action_windows = []
    for target in tqdm(
        _lerobot_action_targets(dataset),
        total=len(dataset),
        desc="Collecting tokenizer actions",
        unit="sample",
    ):
        if not isinstance(target, torch.Tensor) or target.shape != expected:
            msg = f"Tokenizer preparation requires action targets shaped {expected}."
            raise ValueError(msg)
        windows = target.detach().cpu().unfold(0, config.chunk_size, 1)
        action_windows.append(windows.transpose(-1, -2).flatten(1))

    if not action_windows:
        msg = "Cannot fit the action tokenizer on an empty training dataset."
        raise ValueError(msg)
    unique_windows = torch.cat(action_windows).unique(dim=0)
    return unique_windows.reshape(-1, config.chunk_size, config.action_dim)


def _train_tokenizer(
    tokenizer: ResidualActionQuantizer,
    windows: torch.Tensor,
    *,
    device: torch.device,
    num_passes: int,
    batch_size: int,
) -> None:
    """Fit the working tokenizer copy in FP32 over shuffled full passes.

    Raises:
        RuntimeError: If fitting produces a non-finite loss.
    """
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3, weight_decay=1e-4)
    logger.info("Fitting PATCH tokenizer: %d unique windows, %d full passes", len(windows), num_passes)
    with (
        torch.enable_grad(),
        torch.autocast(device_type=device.type, enabled=False),
        tqdm(total=num_passes, desc="Fitting action tokenizer", unit="pass") as progress,
    ):
        for _ in range(num_passes):
            order = torch.randperm(len(windows))
            for indices in order.split(batch_size):
                actions = windows[indices].to(device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = tokenizer.compute_loss(actions)
                if not torch.isfinite(loss):
                    msg = "PATCH tokenizer fitting produced a non-finite loss."
                    raise RuntimeError(msg)
                loss.backward()
                optimizer.step()
            progress.set_postfix(loss=float(loss.detach()), refresh=False)
            progress.update()


def _lerobot_action_targets(dataset: Any) -> Iterator[torch.Tensor]:  # noqa: ANN401
    """Read action-only LeRobot rows using the configured temporal offsets.

    Yields:
        Action targets concatenated in the dataset's configured feature order.

    Raises:
        TypeError: If the dataset is not a LeRobot dataset adapter.
        ValueError: If an action window leaves the training split.
    """
    if not isinstance(dataset, _LeRobotDatasetAdapter):
        msg = "PATCH tokenizer preparation only supports LeRobot datasets."
        raise TypeError(msg)
    source = dataset._lerobot_dataset  # noqa: SLF001
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
            if np.any(positions >= len(absolute)):
                msg = "Tokenizer action windows reference frames outside the training split."
                raise ValueError(msg)
            if np.any(absolute[positions] != targets):
                msg = "Tokenizer action windows reference frames outside the training split."
                raise ValueError(msg)
            values[key] = table[positions.tolist()][key]
        yield dataset._combine_split_columns(values)["action"]  # noqa: SLF001
