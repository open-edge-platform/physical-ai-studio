# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local training backend's spec adapter.

The backend is an adapter over ``training.run_training_job``: what it must get
right is the spec it builds and the paths it passes, so the runner itself is
patched out here. Training behaviour is covered by ``training``'s own tests
instead of being re-asserted through a mock Trainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from schemas.dataset import Snapshot
from schemas.job import _DEFAULT_MAX_EPOCHS, TrainingDevice, TrainingPrecision, TrainJobPayload
from schemas.model import Model
from services.training_backends.base import TrainingContext
from services.training_backends.local import LocalTrainingBackend, build_spec
from training import TrainingJobSpec
from training.job import CHECKPOINT_NAME

if TYPE_CHECKING:
    from pathlib import Path

LOCAL = "services.training_backends.local"


def _payload(**overrides) -> TrainJobPayload:
    return TrainJobPayload.model_validate(
        {
            "project_id": uuid4(),
            "dataset_id": uuid4(),
            "policy": "act",
            "model_name": "m",
            "max_steps": 100,
            "batch_size": 8,
            "num_workers": 0,
            "auto_scale_batch_size": False,
            "compile_model": False,
            "precision": TrainingPrecision.BF16_MIXED,
        }
        | overrides
    )


def _model(path: Path, *, policy: str = "act") -> Model:
    return Model(
        id=uuid4(),
        project_id=uuid4(),
        dataset_id=uuid4(),
        path=str(path),
        name="m",
        snapshot_id=uuid4(),
        policy=policy,
        properties={},
        train_job_id=uuid4(),
        version=1,
        created_at=None,
    )


def _context(tmp_path: Path, payload: TrainJobPayload, *, base_model: Model | None = None) -> TrainingContext:
    model_dir = tmp_path / "models" / str(uuid4())
    snap_dir = tmp_path / "snap"
    snap_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache" / "job"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return TrainingContext(
        job=MagicMock(),
        model=_model(model_dir),
        snapshot=Snapshot(id=uuid4(), dataset_id=uuid4(), path=str(snap_dir)),
        payload=payload,
        base_model=base_model,
        output_dir=model_dir,
        cache_dir=cache_dir,
        progress=MagicMock(),
        should_stop=lambda: False,
    )


class TestBuildSpec:
    """The payload -> spec translation shared by the local and remote backends."""

    def test_training_parameters_are_forwarded(self, tmp_path):
        payload = _payload(max_steps=500, batch_size=16, num_workers="auto", val_split=0.2)
        spec = build_spec(_context(tmp_path, payload))

        assert spec == TrainingJobSpec(
            policy="act",
            max_epochs=_DEFAULT_MAX_EPOCHS,
            batch_size=16,
            num_workers="auto",
            val_split=0.2,
            precision="bf16-mixed",
        )

    @pytest.mark.parametrize(
        ("precision", "expected"),
        [(TrainingPrecision.FP32, "32-true"), (TrainingPrecision.BF16_MIXED, "bf16-mixed")],
    )
    def test_precision_is_passed_as_a_lightning_string(self, tmp_path, precision, expected):
        spec = build_spec(_context(tmp_path, _payload(precision=precision)))

        assert spec.precision == expected

    def test_device_selection_is_forwarded(self, tmp_path):
        payload = _payload(device=TrainingDevice(type="cuda", index=1))
        spec = build_spec(_context(tmp_path, payload))

        assert (spec.device_type, spec.device_index) == ("cuda", 1)

    def test_device_is_left_unset_when_the_job_does_not_choose_one(self, tmp_path):
        spec = build_spec(_context(tmp_path, _payload()))

        assert (spec.device_type, spec.device_index) == (None, None)

    def test_resumed_run_trains_the_base_model_policy(self, tmp_path):
        """A resumed run's architecture comes from the checkpoint, not the request."""
        base_model = _model(tmp_path / "base", policy="pi0")
        context = _context(tmp_path, _payload(base_model_id=base_model.id), base_model=base_model)

        assert build_spec(context).policy == "pi0"


class TestLocalTrainingBackend:
    @pytest.mark.anyio
    async def test_train_runs_the_shared_job_with_the_jobs_paths(self, tmp_path):
        context = _context(tmp_path, _payload())

        with patch("training.run_training_job") as mock_run:
            await LocalTrainingBackend().train(context)

        mock_run.assert_called_once()
        spec, kwargs = mock_run.call_args.args[0], mock_run.call_args.kwargs
        assert spec == build_spec(context)
        assert kwargs["dataset_root"] == context.snapshot.path
        assert kwargs["output_dir"] == context.output_dir
        assert kwargs["cache_dir"] == context.cache_dir
        assert kwargs["resume_from"] is None

    @pytest.mark.anyio
    async def test_train_resumes_from_the_base_models_checkpoint(self, tmp_path):
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        base_model = _model(base_dir, policy="act")
        context = _context(tmp_path, _payload(base_model_id=base_model.id), base_model=base_model)

        with patch("training.run_training_job") as mock_run:
            await LocalTrainingBackend().train(context)

        assert mock_run.call_args.kwargs["resume_from"] == base_dir / CHECKPOINT_NAME

    @pytest.mark.anyio
    async def test_train_without_a_snapshot_is_rejected(self, tmp_path):
        context = _context(tmp_path, _payload())
        context.snapshot = None

        with pytest.raises(ValueError, match="snapshot"):
            await LocalTrainingBackend().train(context)

    @pytest.mark.anyio
    async def test_progress_is_capped_below_completion(self, tmp_path):
        """The worker owns the final 100, so a running report must stay under it."""
        context = _context(tmp_path, _payload())

        with patch("training.run_training_job") as mock_run:
            await LocalTrainingBackend().train(context)
        mock_run.call_args.kwargs["report"](100, "Exporting", {})

        context.progress.assert_called_once_with(99, message="Exporting", extra_info={})
