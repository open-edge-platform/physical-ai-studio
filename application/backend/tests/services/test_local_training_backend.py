# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from schemas.dataset import Snapshot
from schemas.job import TrainingPrecision, TrainJobPayload
from schemas.model import Model
from services.training_backends.base import TrainingContext
from services.training_backends.local import LocalTrainingBackend

if TYPE_CHECKING:
    from pathlib import Path

LOCAL = "services.training_backends.local"


def _payload(*, compile_model: bool, precision: TrainingPrecision) -> TrainJobPayload:
    return TrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="m",
        max_steps=100,
        batch_size=8,
        num_workers=0,
        auto_scale_batch_size=False,
        compile_model=compile_model,
        precision=precision,
    )


def _context(tmp_path: Path, payload: TrainJobPayload) -> TrainingContext:
    model_dir = tmp_path / "models" / str(uuid4())
    snap_dir = tmp_path / "snap"
    snap_dir.mkdir(parents=True)
    cache_dir = tmp_path / "cache" / "job"
    cache_dir.mkdir(parents=True)
    model = Model(
        id=uuid4(),
        project_id=uuid4(),
        dataset_id=uuid4(),
        path=str(model_dir),
        name="m",
        snapshot_id=uuid4(),
        policy="act",
        properties={},
        train_job_id=uuid4(),
        version=1,
        created_at=None,
    )
    return TrainingContext(
        job=MagicMock(),
        model=model,
        snapshot=Snapshot(id=uuid4(), dataset_id=uuid4(), path=str(snap_dir)),
        payload=payload,
        base_model=None,
        output_dir=model_dir,
        cache_dir=cache_dir,
        progress=MagicMock(),
        should_stop=lambda: False,
    )


def _patches(tmp_path: Path):
    trainer = MagicMock()
    trainer.fit = MagicMock()
    trainer.save_checkpoint = MagicMock()
    return trainer, (
        patch("physicalai.train.Trainer", return_value=trainer),
        patch("physicalai.data.LeRobotDataModule"),
        patch("lightning.pytorch.callbacks.ModelCheckpoint"),
        patch("lightning.pytorch.loggers.CSVLogger"),
        patch(f"{LOCAL}.setup_policy", return_value=MagicMock()),
        patch(f"{LOCAL}.load_policy", return_value=MagicMock()),
        patch(f"{LOCAL}.get_torch_device", return_value="cpu"),
        patch(f"{LOCAL}.get_lightning_strategy", return_value="auto"),
        patch(f"{LOCAL}.shutil.move"),
    )


class TestLocalTrainingBackend:
    @pytest.mark.anyio
    async def test_precision_fp32_passed_to_trainer(self, tmp_path):
        payload = _payload(compile_model=False, precision=TrainingPrecision.FP32)
        context = _context(tmp_path, payload)
        trainer, patches = _patches(tmp_path)

        with (
            patches[0] as MockTrainer,
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            await LocalTrainingBackend().train(context)

        trainer.fit.assert_called_once()
        assert MockTrainer.call_args.kwargs["precision"] == "32-true"

    @pytest.mark.anyio
    async def test_precision_bf16_passed_to_trainer(self, tmp_path):
        payload = _payload(compile_model=False, precision=TrainingPrecision.BF16_MIXED)
        context = _context(tmp_path, payload)
        trainer, patches = _patches(tmp_path)

        with (
            patches[0] as MockTrainer,
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            await LocalTrainingBackend().train(context)

        assert MockTrainer.call_args.kwargs["precision"] == "bf16-mixed"

    @pytest.mark.anyio
    async def test_compile_reloads_non_compiled_policy_for_export(self, tmp_path):
        payload = _payload(compile_model=True, precision=TrainingPrecision.BF16_MIXED)
        context = _context(tmp_path, payload)
        trainer, patches = _patches(tmp_path)

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4] as mock_setup,
            patches[5] as mock_load,
            patches[6],
            patches[7],
            patches[8],
        ):
            await LocalTrainingBackend().train(context)

        # compile_model=True with an 'act' policy triggers a non-compiled reload for export.
        assert mock_setup.call_args.kwargs["compile_model"] is True
        mock_load.assert_called_once_with(context.model, compile_model=False)
