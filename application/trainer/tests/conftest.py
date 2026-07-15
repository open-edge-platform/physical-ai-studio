# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for trainer service tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from trainer.schemas import SubmitJobRequest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_request() -> SubmitJobRequest:
    """A valid http-transfer job submission request."""
    return SubmitJobRequest(
        payload={"max_steps": 100, "batch_size": 8, "precision": "bf16-mixed"},
        policy="act",
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path to a throwaway SQLite database."""
    return tmp_path / "trainer.db"
