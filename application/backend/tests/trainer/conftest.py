# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for trainer service tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from trainer.schemas import SubmitJobRequest
from training import TrainingJobSpec

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_request() -> SubmitJobRequest:
    """A valid http-transfer job submission request."""
    return SubmitJobRequest(
        spec=TrainingJobSpec(policy="act", max_epochs=5, batch_size=8, precision="bf16-mixed"),
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path to a throwaway SQLite database."""
    return tmp_path / "trainer.db"
