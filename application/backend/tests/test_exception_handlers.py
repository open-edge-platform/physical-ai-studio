# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the app-wide FastAPI exception handlers in `exception_handlers.py`."""

from fastapi.testclient import TestClient

from main import app


def test_validation_exception_handler_reports_clean_field_names() -> None:
    """Validation responses report the plain field name."""
    response = TestClient(app).post(
        "/api/remote-servers/aliases",
        json={"alias": "A100 GPU", "hostname": "10.0.0.9"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "bad_request"
    assert "alias" in body["message"]
    assert all(isinstance(value, list) for value in body["message"].values())


def test_validation_exception_handler_reports_nested_field_names() -> None:
    """A nested body field is reported with real dot-notation, not a stringified tuple."""
    response = TestClient(app).patch(
        "/api/settings",
        json={"ssh": {"connect_timeout_s": "not-a-number"}},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "bad_request"
    assert "ssh.connect_timeout_s" in body["message"]
