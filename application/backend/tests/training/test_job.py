# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared training job spec and runner.

``run_training_job`` is the single training path used both in-process by the
studio and by the standalone trainer service, so what is asserted here is the
contract both depend on: how a spec becomes a policy and a ``Trainer``, where
artifacts land, and what a canceled run leaves behind. Lightning's own fit loop
is mocked out; it is not under test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from physicalai.export import ExportablePolicyMixin, ExportBackend
from pydantic import ValidationError

from training import TrainingJobSpec
from training.job import CHECKPOINT_NAME, EXPORTS_DIRNAME, PRETRAINED_BASE_CHECKPOINTS, build_policy, run_training_job

if TYPE_CHECKING:
    from pathlib import Path

JOB = "training.job"


class TestTrainingJobSpec:
    """The spec is a wire format as well as a local config, so both matter."""

    def test_only_the_policy_is_required(self) -> None:
        spec = TrainingJobSpec(policy="act")

        assert (spec.policy_source, spec.max_steps, spec.batch_size) == ("physicalai", 100, 8)
        assert (spec.num_workers, spec.val_split, spec.precision) == ("auto", 0.1, "bf16-mixed")
        assert (spec.compile_model, spec.auto_scale_batch_size) == (False, False)
        assert (spec.device_type, spec.device_index) == (None, None)

    def test_unknown_field_is_rejected(self) -> None:
        """A stray field over the wire is a version mismatch, not a value to drop."""
        with pytest.raises(ValidationError):
            TrainingJobSpec(policy="act", learning_rate=0.1)

    @pytest.mark.parametrize(
        "invalid",
        [
            {"max_steps": 0},
            {"batch_size": 0},
            {"val_split": 1.0},
            {"val_split": -0.1},
            {"device_index": -1},
            {"policy_source": "elsewhere"},
        ],
    )
    def test_out_of_range_values_are_rejected(self, invalid: dict) -> None:
        with pytest.raises(ValidationError):
            TrainingJobSpec(policy="act", **invalid)

    def test_spec_round_trips_through_json(self) -> None:
        """Remote submission sends the spec as JSON; it must survive the trip."""
        spec = TrainingJobSpec(policy="pi0", max_steps=500, num_workers=4, device_type="xpu", device_index=1)

        assert TrainingJobSpec.model_validate_json(spec.model_dump_json()) == spec


class TestBuildPolicy:
    def test_fresh_policy_is_built_from_the_spec(self) -> None:
        with patch("physicalai.policies.get_policy") as get_policy:
            policy = build_policy(TrainingJobSpec(policy="act", compile_model=True))

        assert policy is get_policy.return_value
        get_policy.assert_called_once_with("act", source="physicalai", compile_model=True)

    @pytest.mark.parametrize("policy_name", sorted(PRETRAINED_BASE_CHECKPOINTS))
    def test_finetune_only_policies_start_from_pretrained_weights(self, policy_name: str) -> None:
        """These policies have no from-scratch initialization worth training."""
        with patch("physicalai.policies.get_policy") as get_policy:
            build_policy(TrainingJobSpec(policy=policy_name))

        assert get_policy.call_args.kwargs["pretrained_name_or_path"] == PRETRAINED_BASE_CHECKPOINTS[policy_name]

    def test_lerobot_policies_are_left_to_lerobots_own_defaults(self) -> None:
        with patch("physicalai.policies.get_policy") as get_policy:
            build_policy(TrainingJobSpec(policy="smolvla", policy_source="lerobot"))

        assert "pretrained_name_or_path" not in get_policy.call_args.kwargs

    def test_resume_loads_the_checkpoint_instead_of_a_new_policy(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / CHECKPOINT_NAME
        policy_class = MagicMock()

        with patch("physicalai.policies.get_physicalai_policy_class", return_value=policy_class):
            policy = build_policy(TrainingJobSpec(policy="act"), resume_from=checkpoint)

        assert policy is policy_class.load_from_checkpoint.return_value
        policy_class.load_from_checkpoint.assert_called_once_with(str(checkpoint))

    def test_pi0_is_resumed_weights_only(self, tmp_path: Path) -> None:
        """Pi0 checkpoints hold objects Lightning will not unpickle by default."""
        policy_class = MagicMock()

        with patch("physicalai.policies.get_physicalai_policy_class", return_value=policy_class):
            build_policy(TrainingJobSpec(policy="pi0"), resume_from=tmp_path / CHECKPOINT_NAME)

        assert policy_class.load_from_checkpoint.call_args.kwargs == {"weights_only": True}

    def test_lerobot_policies_are_resumed_through_the_wrapper(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / CHECKPOINT_NAME

        with patch("physicalai.policies.lerobot.LeRobotPolicy") as wrapper:
            policy = build_policy(TrainingJobSpec(policy="act", policy_source="lerobot"), resume_from=checkpoint)

        assert policy is wrapper.load_from_checkpoint.return_value
        wrapper.load_from_checkpoint.assert_called_once_with(checkpoint)


class _ExportablePolicy(ExportablePolicyMixin):
    """A policy that records the export calls the runner makes."""

    def __init__(self, backends: list[ExportBackend], *, failing: set[ExportBackend] | None = None) -> None:
        self._backends = backends
        self._failing = failing or set()
        self.exported: list[tuple[str, ExportBackend]] = []

    def get_supported_export_backends(self) -> list[ExportBackend]:  # type: ignore[override]
        return self._backends

    def export(self, output_path, backend, input_sample=None, **export_kwargs) -> None:  # type: ignore[override]
        if backend in self._failing:
            msg = f"{backend} unavailable"
            raise RuntimeError(msg)
        self.exported.append((str(output_path), backend))


def _run(
    spec: TrainingJobSpec,
    tmp_path: Path,
    *,
    policy: object | None = None,
    should_stop: bool = False,
    report: MagicMock | None = None,
) -> MagicMock:
    """Run a job with the datamodule and Lightning trainer mocked out.

    Returns:
        The patched ``Trainer`` class, so tests can assert how it was configured.
    """
    with (
        patch("physicalai.data.LeRobotDataModule"),
        patch(f"{JOB}.build_policy", return_value=policy if policy is not None else MagicMock()),
        patch("physicalai.train.trainer.Trainer") as trainer_class,
    ):
        run_training_job(
            spec,
            dataset_root=tmp_path / "snapshot",
            output_dir=tmp_path / "model",
            cache_dir=tmp_path / "cache" / "job",
            report=report or MagicMock(),
            should_stop=lambda: should_stop,
        )
    return trainer_class


class TestRunTrainingJob:
    def test_trainer_is_configured_from_the_spec(self, tmp_path: Path) -> None:
        spec = TrainingJobSpec(
            policy="act",
            max_steps=500,
            precision="32-true",
            auto_scale_batch_size=True,
            device_type="cpu",
            device_index=1,
        )

        trainer_class = _run(spec, tmp_path)

        kwargs = trainer_class.call_args.kwargs
        assert kwargs["max_steps"] == 500
        assert kwargs["precision"] == "32-true"
        assert kwargs["auto_scale_batch_size"] is True
        assert (kwargs["accelerator"], kwargs["strategy"], kwargs["devices"]) == ("cpu", "auto", [1])

    def test_xpu_gets_its_single_device_strategy(self, tmp_path: Path) -> None:
        """Device resolution is shared with the trainer service; assert it is used."""
        trainer_class = _run(TrainingJobSpec(policy="act", device_type="xpu"), tmp_path)

        kwargs = trainer_class.call_args.kwargs
        assert (kwargs["accelerator"], kwargs["strategy"], kwargs["devices"]) == ("xpu", "xpu_single", 1)

    def test_dataset_is_loaded_from_the_local_root(self, tmp_path: Path) -> None:
        spec = TrainingJobSpec(policy="act", batch_size=16, num_workers=2, val_split=0.25)

        with (
            patch("physicalai.data.LeRobotDataModule") as datamodule,
            patch(f"{JOB}.build_policy"),
            patch("physicalai.train.trainer.Trainer"),
        ):
            run_training_job(
                spec,
                dataset_root=tmp_path / "snapshot",
                output_dir=tmp_path / "model",
                cache_dir=tmp_path / "cache" / "job",
                report=MagicMock(),
                should_stop=lambda: False,
            )

        kwargs = datamodule.call_args.kwargs
        assert kwargs["root"] == str(tmp_path / "snapshot")
        assert (kwargs["train_batch_size"], kwargs["num_workers"], kwargs["val_split"]) == (16, 2, 0.25)

    def test_completed_run_publishes_the_cache_as_the_model_directory(self, tmp_path: Path) -> None:
        trainer_class = _run(TrainingJobSpec(policy="act"), tmp_path)

        trainer = trainer_class.return_value
        trainer.save_checkpoint.assert_called_once_with(tmp_path / "cache" / "job" / CHECKPOINT_NAME)
        assert (tmp_path / "model").is_dir()
        assert not (tmp_path / "cache" / "job").exists()

    def test_completed_run_replaces_an_existing_model_directory(self, tmp_path: Path) -> None:
        """Retraining into the same directory must not merge with the old model."""
        output_dir = tmp_path / "model"
        output_dir.mkdir()
        (output_dir / "stale.txt").write_text("old")

        _run(TrainingJobSpec(policy="act"), tmp_path)

        assert not (output_dir / "stale.txt").exists()

    def test_canceled_run_leaves_no_model_behind(self, tmp_path: Path) -> None:
        """A partially trained policy is not an artifact worth keeping."""
        trainer_class = _run(TrainingJobSpec(policy="act"), tmp_path, should_stop=True)

        trainer_class.return_value.save_checkpoint.assert_not_called()
        assert not (tmp_path / "model").exists()

    def test_training_start_is_reported(self, tmp_path: Path) -> None:
        report = MagicMock()

        _run(TrainingJobSpec(policy="act"), tmp_path, report=report)

        assert report.call_args_list[0].args == (0, "Training model", {})

    def test_policy_is_exported_to_every_supported_backend(self, tmp_path: Path) -> None:
        policy = _ExportablePolicy([ExportBackend.TORCH, ExportBackend.OPENVINO])
        report = MagicMock()

        _run(TrainingJobSpec(policy="act"), tmp_path, policy=policy, report=report)

        exports = tmp_path / "model" / EXPORTS_DIRNAME
        assert policy.exported == [
            (str(exports / "torch"), ExportBackend.TORCH),
            (str(exports / "openvino"), ExportBackend.OPENVINO),
        ]
        assert (99, "Exporting to torch format", {}) in [call.args for call in report.call_args_list]

    def test_a_failing_export_backend_does_not_fail_the_job(self, tmp_path: Path) -> None:
        """Weights are already saved by then; one bad backend must not lose them."""
        policy = _ExportablePolicy(
            [ExportBackend.TORCH, ExportBackend.OPENVINO],
            failing={ExportBackend.TORCH},
        )

        _run(TrainingJobSpec(policy="act"), tmp_path, policy=policy)

        assert [backend for _, backend in policy.exported] == [ExportBackend.OPENVINO]
