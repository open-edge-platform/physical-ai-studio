# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for trainer health metadata."""

from __future__ import annotations

import asyncio

from trainer.main import health


def test_health_reports_image_compatibility_metadata(monkeypatch) -> None:
    """Health reports the build attributes that provisioning validates."""
    monkeypatch.setenv("TRAINER_API_PROTOCOL_VERSION", "3")
    monkeypatch.setenv("TRAINER_DEVICE_TYPE", "cuda")
    monkeypatch.setenv("TRAINER_BUILD_REVISION", "a" * 40)
    monkeypatch.setenv("TRAINER_BUILD_DATE", "2026-07-14T08:00:00Z")
    monkeypatch.setenv("TRAINER_APPLICATION_VERSION", "0.1.0")

    result = asyncio.run(health())

    assert result.model_dump() == {
        "status": "healthy",
        "protocol_version": 3,
        "device_type": "cuda",
        "build_revision": "a" * 40,
        "build_date": "2026-07-14T08:00:00Z",
        "application_version": "0.1.0",
    }
