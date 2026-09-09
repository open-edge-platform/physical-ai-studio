# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `TrainJobPayload` discriminated union.

Local training, the direct-URL remote trainer registry, and SSH-provisioned
servers are mutually exclusive targets, each modeled as its own payload class
(`LocalTrainJobPayload`, `RemoteTrainJobPayload`, `SshTrainJobPayload`). A
payload can never express two targets at once: a target's fields simply don't
exist on another target's class, and `extra="forbid"` rejects any attempt to
pass them in anyway.
"""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.job import LocalTrainJobPayload, RemoteTrainJobPayload, SshTrainJobPayload, TrainingTarget


def _base_kwargs() -> dict:
    return {
        "project_id": uuid4(),
        "dataset_id": uuid4(),
        "policy": "act",
        "model_name": "model",
    }


class TestLocalTarget:
    def test_local_job_accepts_no_remote_fields(self) -> None:
        payload = LocalTrainJobPayload(**_base_kwargs())
        assert payload.training_target is TrainingTarget.LOCAL

    @pytest.mark.parametrize(
        "extra",
        [
            {"remote_trainer_id": uuid4()},
            {"remote_trainer_url": "https://trainer.test"},
            {"remote_trainer_name": "trainer"},
            {"remote_server_id": uuid4()},
        ],
    )
    def test_local_job_rejects_any_remote_field(self, extra: dict) -> None:
        with pytest.raises(ValidationError):
            LocalTrainJobPayload(**_base_kwargs(), **extra)


class TestRemoteTarget:
    def test_remote_job_requires_remote_trainer_id(self) -> None:
        with pytest.raises(ValidationError):
            RemoteTrainJobPayload(**_base_kwargs())

    def test_remote_job_accepts_direct_url_fields(self) -> None:
        payload = RemoteTrainJobPayload(
            **_base_kwargs(),
            remote_trainer_id=uuid4(),
            remote_trainer_url="https://trainer.test",
        )
        assert payload.training_target is TrainingTarget.REMOTE

    def test_remote_job_rejects_remote_server_id(self) -> None:
        with pytest.raises(ValidationError):
            RemoteTrainJobPayload.model_validate(
                {**_base_kwargs(), "remote_trainer_id": uuid4(), "remote_server_id": uuid4()}
            )


class TestSshTarget:
    def test_ssh_job_requires_remote_server_id(self) -> None:
        with pytest.raises(ValidationError):
            SshTrainJobPayload(**_base_kwargs())

    def test_ssh_job_accepts_remote_server_id(self) -> None:
        payload = SshTrainJobPayload(
            **_base_kwargs(),
            remote_server_id=uuid4(),
        )
        assert payload.training_target is TrainingTarget.SSH
        assert payload.remote_server_id is not None

    @pytest.mark.parametrize(
        "extra",
        [
            {"remote_trainer_id": uuid4()},
            {"remote_trainer_url": "https://trainer.test"},
            {"remote_trainer_name": "trainer"},
        ],
    )
    def test_ssh_job_rejects_direct_url_fields(self, extra: dict) -> None:
        with pytest.raises(ValidationError):
            SshTrainJobPayload(
                **_base_kwargs(),
                remote_server_id=uuid4(),
                **extra,
            )


class TestSnapFlowDistillation:
    """The payload expresses the distillation budget; the runner needs a boundary.

    Distillation epochs are additive on top of ``max_epochs`` (the teacher
    phase always runs the full budget, then distillation extends the run),
    so `snapflow_start_epoch` is always just `max_epochs`.
    """

    def test_training_is_flow_matching_by_default(self) -> None:
        payload = LocalTrainJobPayload(**_base_kwargs())

        assert payload.snapflow_enabled is False
        assert payload.snapflow_start_epoch is None

    @pytest.mark.parametrize("policy", ["pi05", "smolvla", "Pi05", "SmolVLA"])
    def test_flow_matching_policies_can_be_distilled(self, policy: str) -> None:
        payload = LocalTrainJobPayload(
            **{**_base_kwargs(), "policy": policy},
            max_epochs=8,
            snapflow_enabled=True,
            snapflow_distill_epochs=3,
        )

        assert payload.snapflow_start_epoch == 8
        assert payload.total_epochs == 11

    @pytest.mark.parametrize("policy", ["act", "pi0", "groot"])
    def test_other_policies_are_rejected_rather_than_silently_trained_without_it(self, policy: str) -> None:
        with pytest.raises(ValidationError, match="not available for policy"):
            LocalTrainJobPayload(**{**_base_kwargs(), "policy": policy}, snapflow_enabled=True)

    def test_the_distillation_budget_is_not_bounded_by_max_epochs(self) -> None:
        """Distillation is additive, so it may exceed max_epochs without error."""
        payload = LocalTrainJobPayload(
            **{**_base_kwargs(), "policy": "pi05"},
            max_epochs=3,
            snapflow_enabled=True,
            snapflow_distill_epochs=3,
        )

        assert payload.snapflow_start_epoch == 3
        assert payload.total_epochs == 6

    def test_the_boundary_is_measured_against_the_default_epoch_count_when_unset(self) -> None:
        """`max_epochs` is filled in by `resolve_training_limit`, which runs first."""
        payload = LocalTrainJobPayload(
            **{**_base_kwargs(), "policy": "pi05"},
            snapflow_enabled=True,
            snapflow_distill_epochs=2,
        )

        assert payload.snapflow_start_epoch == payload.max_epochs
        assert payload.total_epochs == payload.max_epochs + 2

    def test_a_distillation_budget_on_an_unsupported_policy_is_fine_while_disabled(self) -> None:
        """The field carries a default, so it must not gate an ordinary ACT run."""
        payload = LocalTrainJobPayload(**_base_kwargs(), snapflow_distill_epochs=99)

        assert payload.snapflow_start_epoch is None

    def test_every_target_can_distil(self) -> None:
        remote = RemoteTrainJobPayload(
            **{**_base_kwargs(), "policy": "pi05"},
            remote_trainer_id=uuid4(),
            snapflow_enabled=True,
        )
        ssh = SshTrainJobPayload(
            **{**_base_kwargs(), "policy": "pi05"},
            remote_server_id=uuid4(),
            snapflow_enabled=True,
        )

        assert (remote.snapflow_start_epoch, ssh.snapflow_start_epoch) == (5, 5)
