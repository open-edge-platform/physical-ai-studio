# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for trainer request validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trainer.schemas import SubmitJobRequest

_SHA = "a" * 40


def test_valid_request_accepted() -> None:
    request = SubmitJobRequest(payload={}, repo_id="acme/snap-1", revision=_SHA, policy="smolvla")
    assert request.policy == "smolvla"
    assert request.revision == _SHA


@pytest.mark.parametrize("repo_id", ["../etc/passwd", "bad/../escape", "a/b/c", "", "with space"])
def test_invalid_repo_id_rejected(repo_id: str) -> None:
    with pytest.raises(ValidationError):
        SubmitJobRequest(payload={}, repo_id=repo_id, revision=_SHA, policy="act")


@pytest.mark.parametrize("revision", ["main", "v1.0", "a" * 39, "a" * 41, "g" * 40, ""])
def test_non_sha_revision_rejected(revision: str) -> None:
    """Branch names and non-hex/short SHAs must be rejected; only pinned SHAs allowed."""
    with pytest.raises(ValidationError):
        SubmitJobRequest(payload={}, repo_id="acme/snap", revision=revision, policy="act")


@pytest.mark.parametrize("policy", ["unknown", "gpt", ""])
def test_unsupported_policy_rejected(policy: str) -> None:
    with pytest.raises(ValidationError):
        SubmitJobRequest(payload={}, repo_id="acme/snap", revision=_SHA, policy=policy)
