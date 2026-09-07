# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""In-process training backend using torch/Lightning.

This is an adapter, not a training implementation: it maps a `TrainingContext`
onto a `training.TrainingJobSpec` and hands it to `training.run_training_job`,
the same runner the trainer service uses. Keeping the training logic in one
place is what stops the local and remote paths from drifting apart.

Imports of `physicalai`, torch, and Lightning are deferred to call time so this
module can be imported in environments without the `[train]` extra installed.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import SecretStr

from schemas.job import _DEFAULT_MAX_EPOCHS
from services.training_backends._log_format import render_progress_log
from settings import get_settings

if TYPE_CHECKING:
    from physicalai.train.callbacks import ReportFn

    from services.training_backends.base import TrainingContext
    from training import TrainingJobSpec


class LocalTrainingBackend:
    """Train in the worker process with Lightning."""

    async def train(self, context: TrainingContext) -> None:
        """Run Lightning training, save, and export into the model directory."""
        from training import RunOptions, run_training_job

        if context.snapshot is None:
            raise ValueError("Local training requires a dataset snapshot")

        spec = build_spec(context)
        hf_token = get_settings().huggingface.hf_token
        # Fallback to Environment Variable based hf token if settings hasn't been set
        if (hf_token is None or not hf_token.get_secret_value()) and (
            legacy_hf_token := os.environ.get("HF_TOKEN", "")
        ):
            hf_token = SecretStr(legacy_hf_token)
        spec.run_options = RunOptions(
            resume_from=_resume_checkpoint(context),
            hf_token=hf_token,
        )
        await asyncio.to_thread(
            run_training_job,
            spec,
            dataset_root=context.snapshot.path,
            output_dir=context.output_dir,
            cache_dir=context.cache_dir,
            report=self._reporter(context),
            should_stop=context.should_stop,
        )

    @staticmethod
    def _reporter(context: TrainingContext) -> ReportFn:
        """Wrap the job reporter for `run_training_job`'s telemetry sink.

        Mirrors loggable telemetry to the job log using the shared renderer, so
        local runs produce the same log lines as remote ones, and caps running
        progress at 99 because the worker writes 100 on completion.
        """
        reporter = context.progress

        def report(progress: int, message: str | None, extra_info: dict) -> None:
            line = render_progress_log(extra_info)
            if line is not None:
                logger.info(line)
            reporter(min(99, progress), message=message, extra_info=extra_info)

        return report


def build_spec(context: TrainingContext) -> TrainingJobSpec:
    """Translate a job's payload into the shared training spec.

    Shared with the remote backend, which sends the same spec over the wire, so
    both paths train from one set of defaults instead of two.

    Args:
        context: The training job to describe.

    Returns:
        The spec describing what to train.
    """
    from training import TrainingJobSpec

    payload = context.payload
    device = payload.device
    return TrainingJobSpec(
        # A resumed run's architecture is dictated by the base model's checkpoint.
        policy=(context.base_model or context.model).policy,
        max_epochs=payload.max_epochs if payload.max_epochs is not None else _DEFAULT_MAX_EPOCHS,
        batch_size=payload.batch_size,
        num_workers=payload.num_workers,
        val_split=payload.val_split,
        precision=str(payload.precision),
        compile_model=payload.compile_model,
        auto_scale_batch_size=payload.auto_scale_batch_size,
        device_type=str(device.type) if device else None,
        device_index=device.index if device else None,
    )


def _resume_checkpoint(context: TrainingContext) -> Path | None:
    """Return the base model's checkpoint to resume from, if the job has one."""
    from training.job import CHECKPOINT_NAME

    if context.base_model is None:
        return None
    return Path(context.base_model.path) / CHECKPOINT_NAME
