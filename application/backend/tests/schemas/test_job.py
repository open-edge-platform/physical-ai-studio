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
