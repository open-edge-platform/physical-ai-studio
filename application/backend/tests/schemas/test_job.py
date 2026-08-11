# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for TrainJobPayload's execution-target validator.

Local training, the direct-URL remote trainer registry, and SSH-provisioned
servers are mutually exclusive targets. Each must reject fields that belong to
a different target so a payload can never express two targets at once.
"""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.job import TrainingTarget, TrainJobPayload


def _base_kwargs() -> dict:
    return {
        "project_id": uuid4(),
        "dataset_id": uuid4(),
        "policy": "act",
        "model_name": "model",
    }


class TestLocalTarget:
    def test_local_job_accepts_no_remote_fields(self) -> None:
        payload = TrainJobPayload(**_base_kwargs(), training_target=TrainingTarget.LOCAL)
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
            TrainJobPayload(**_base_kwargs(), training_target=TrainingTarget.LOCAL, **extra)


class TestRemoteTarget:
    def test_remote_job_requires_remote_trainer_id(self) -> None:
        with pytest.raises(ValidationError):
            TrainJobPayload(**_base_kwargs(), training_target=TrainingTarget.REMOTE)

    def test_remote_job_accepts_direct_url_fields(self) -> None:
        payload = TrainJobPayload(
            **_base_kwargs(),
            training_target=TrainingTarget.REMOTE,
            remote_trainer_id=uuid4(),
            remote_trainer_url="https://trainer.test",
        )
        assert payload.training_target is TrainingTarget.REMOTE

    def test_remote_job_rejects_remote_server_id(self) -> None:
        with pytest.raises(ValidationError):
            TrainJobPayload(
                **_base_kwargs(),
                training_target=TrainingTarget.REMOTE,
                remote_trainer_id=uuid4(),
                remote_server_id=uuid4(),
            )


class TestSshTarget:
    def test_ssh_job_requires_remote_server_id(self) -> None:
        with pytest.raises(ValidationError):
            TrainJobPayload(**_base_kwargs(), training_target=TrainingTarget.SSH)

    def test_ssh_job_accepts_remote_server_id(self) -> None:
        payload = TrainJobPayload(
            **_base_kwargs(),
            training_target=TrainingTarget.SSH,
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
            TrainJobPayload(
                **_base_kwargs(),
                training_target=TrainingTarget.SSH,
                remote_server_id=uuid4(),
                **extra,
            )
