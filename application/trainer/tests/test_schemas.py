# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for trainer request validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trainer.schemas import DatasetTransfer, SubmitJobRequest

_SHA = "a" * 40


def test_valid_hf_request_accepted() -> None:
    request = SubmitJobRequest(
        payload={},
        dataset_transfer="hf",
        repo_id="acme/snap-1",
        revision=_SHA,
        policy="smolvla",
    )
    assert request.policy == "smolvla"
    assert request.revision == _SHA


def test_http_request_defaults_and_omits_repo_fields() -> None:
    request = SubmitJobRequest(payload={}, policy="act")
    assert request.dataset_transfer == DatasetTransfer.HTTP
    assert request.repo_id is None
    assert request.revision is None


def test_http_request_rejects_repo_fields() -> None:
    """An http job must not carry HF repo coordinates."""
    with pytest.raises(ValidationError):
        SubmitJobRequest(payload={}, dataset_transfer="http", repo_id="acme/snap", revision=_SHA, policy="act")


def test_hf_request_requires_repo_and_revision() -> None:
    with pytest.raises(ValidationError):
        SubmitJobRequest(payload={}, dataset_transfer="hf", policy="act")


@pytest.mark.parametrize("repo_id", ["../etc/passwd", "bad/../escape", "a/b/c", "", "with space"])
def test_invalid_repo_id_rejected(repo_id: str) -> None:
    with pytest.raises(ValidationError):
        SubmitJobRequest(payload={}, dataset_transfer="hf", repo_id=repo_id, revision=_SHA, policy="act")


@pytest.mark.parametrize("revision", ["main", "v1.0", "a" * 39, "a" * 41, "g" * 40, ""])
def test_non_sha_revision_rejected(revision: str) -> None:
    """Branch names and non-hex/short SHAs must be rejected; only pinned SHAs allowed."""
    with pytest.raises(ValidationError):
        SubmitJobRequest(payload={}, dataset_transfer="hf", repo_id="acme/snap", revision=revision, policy="act")


@pytest.mark.parametrize("policy", ["unknown", "gpt", ""])
def test_unsupported_policy_rejected(policy: str) -> None:
    with pytest.raises(ValidationError):
        SubmitJobRequest(payload={}, dataset_transfer="hf", repo_id="acme/snap", revision=_SHA, policy=policy)
