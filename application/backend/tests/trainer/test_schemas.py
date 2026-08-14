# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for trainer request validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trainer.schemas import DatasetTransfer, SubmitJobRequest
from training import TrainingJobSpec


def test_http_request_defaults() -> None:
    request = SubmitJobRequest(spec=TrainingJobSpec(policy="act"))
    assert request.dataset_transfer == DatasetTransfer.HTTP


def test_spec_defaults_come_from_the_library() -> None:
    """The wire format must not restate the library's training defaults."""
    request = SubmitJobRequest.model_validate({"spec": {"policy": "act"}})
    assert request.spec == TrainingJobSpec(policy="act")


@pytest.mark.parametrize("policy", ["unknown", "gpt", ""])
def test_unsupported_policy_rejected(policy: str) -> None:
    with pytest.raises(ValidationError):
        SubmitJobRequest(spec=TrainingJobSpec(policy=policy))


def test_unknown_spec_field_rejected() -> None:
    """A stray field is a client/server mismatch, not something to ignore."""
    with pytest.raises(ValidationError):
        SubmitJobRequest.model_validate({"spec": {"policy": "act", "learning_rate": 0.1}})


def test_protocol_1_request_rejected() -> None:
    """A protocol-1 body (untyped payload + policy) is no longer accepted."""
    with pytest.raises(ValidationError):
        SubmitJobRequest.model_validate({"payload": {"max_steps": 100}, "policy": "act"})
