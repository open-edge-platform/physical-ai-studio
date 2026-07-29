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


def test_health_falls_back_on_invalid_protocol_version(monkeypatch) -> None:
    """A malformed protocol version env var doesn't crash the health check."""
    monkeypatch.setenv("TRAINER_API_PROTOCOL_VERSION", "not-a-number")

    result = asyncio.run(health())

    assert result.protocol_version == 1


def test_health_endpoint_serializes_by_field_name_not_env_alias(monkeypatch) -> None:
    """The HTTP /health response must use field names, not env-var aliases.

    ``HealthInfo`` fields carry env-var aliases (e.g. ``TRAINER_API_PROTOCOL_VERSION``)
    used only to source values from the environment. FastAPI serializes
    ``response_model`` fields by alias by default, so a regression here would
    leak those env-var names into the JSON body instead of documented field
    names like ``protocol_version``.
    """
    from fastapi.testclient import TestClient

    from trainer import main

    monkeypatch.setenv("TRAINER_API_PROTOCOL_VERSION", "1")
    monkeypatch.setenv("TRAINER_DEVICE_TYPE", "xpu")
    monkeypatch.setenv("TRAINER_BUILD_REVISION", "a" * 40)
    monkeypatch.setenv("TRAINER_BUILD_DATE", "2026-07-14T08:00:00Z")
    monkeypatch.setenv("TRAINER_APPLICATION_VERSION", "0.1.0")

    # No context manager: the /health route needs no app lifespan/queue manager.
    client = TestClient(main.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "protocol_version": 1,
        "device_type": "xpu",
        "build_revision": "a" * 40,
        "build_date": "2026-07-14T08:00:00Z",
        "application_version": "0.1.0",
    }
