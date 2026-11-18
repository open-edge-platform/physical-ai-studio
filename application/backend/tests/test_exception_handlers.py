# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the app-wide FastAPI exception handlers in `exception_handlers.py`."""

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from exception_handlers import _log_handled_exception, _rewrite_validation_error_status_code
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


@pytest.mark.parametrize("status_code", [400, 500])
@pytest.mark.parametrize("chained", [False, True])
def test_handled_exception_logging_keeps_context_and_tracebacks(mocker, status_code: int, chained: bool) -> None:
    logger = mocker.patch("exception_handlers.logger")
    request = Request({"type": "http", "method": "POST", "path": "/api/robots", "headers": []})
    exception = ValueError("sanitized failure")
    cause = RuntimeError("device unavailable") if chained else None
    exception.__cause__ = cause

    _log_handled_exception(request, status_code, "failed", exception)

    if status_code >= 500:
        logger.opt.assert_called_once_with(exception=exception)
        logger.opt.return_value.error.assert_called_once_with("POST /api/robots -> 500 failed")
    elif cause is not None:
        logger.opt.assert_called_once_with(exception=cause)
        logger.opt.return_value.warning.assert_called_once_with(
            "POST /api/robots -> 400 failed (cause: RuntimeError('device unavailable'))"
        )
    else:
        logger.opt.assert_not_called()
        logger.warning.assert_called_once_with("POST /api/robots -> 400 failed")


def test_openapi_preserves_explicit_request_errors_alongside_validation() -> None:
    schema = {
        "paths": {
            "/test": {
                "post": {
                    "responses": {
                        "400": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/OperationalIssue"}}
                            }
                        },
                        "422": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
                            }
                        },
                    }
                }
            }
        }
    }
    rewritten = _rewrite_validation_error_status_code(schema)
    responses = rewritten["paths"]["/test"]["post"]["responses"]
    assert "422" not in responses
    assert responses["400"]["content"]["application/json"]["schema"]["anyOf"] == [
        {"$ref": "#/components/schemas/OperationalIssue"},
        {"$ref": "#/components/schemas/HTTPValidationError"},
    ]
    assert _rewrite_validation_error_status_code(rewritten) == rewritten
