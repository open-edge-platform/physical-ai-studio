# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for trainer request validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trainer.schemas import DatasetTransfer, SubmitJobRequest


def test_http_request_defaults() -> None:
    request = SubmitJobRequest(payload={}, policy="act")
    assert request.dataset_transfer == DatasetTransfer.HTTP


@pytest.mark.parametrize("policy", ["unknown", "gpt", ""])
def test_unsupported_policy_rejected(policy: str) -> None:
    with pytest.raises(ValidationError):
        SubmitJobRequest(payload={}, policy=policy)
