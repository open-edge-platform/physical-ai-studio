# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from uuid import uuid4

from db.schema import RemoteServerDB
from repositories.mappers.remote_server_mapper import RemoteServerMapper
from schemas.hardware import DeviceType
from schemas.remote_server import RemoteServer
from schemas.ssh_preflight import CheckKey, CheckOutcome, PreflightCheck, PreflightTier


def _check() -> PreflightCheck:
    return PreflightCheck(
        key=CheckKey.IMAGE_RESOLVED,
        tier=PreflightTier.TIER_2,
        outcome=CheckOutcome.PASSED,
        blocking=True,
        checked_at=datetime.now(UTC),
        detail="pinned image resolved",
        method="docker",
    )


def test_to_schema_serializes_last_check_checks_for_json_storage() -> None:
    """`last_check_checks` must round-trip through a JSON-serializable form.

    Regression guard: before this field existed, the detailed Tier 2 checks
    from `/check` were discarded after the response, so the "Image pull &
    verification" card always reset to "Not verified yet" on a page refresh
    even though the server had just been verified.
    """
    check = _check()
    remote_server = RemoteServer(
        id=uuid4(),
        name="server",
        ssh_host_alias="gpu-box",
        device_type=DeviceType.CUDA,
        last_check_status="healthy",
        last_check_checks=[check],
    )

    db_row = RemoteServerMapper.to_schema(remote_server)

    assert db_row.last_check_checks == [check.model_dump(mode="json")]


def test_from_schema_restores_last_check_checks_as_preflight_checks() -> None:
    """A persisted row's stored dicts must come back as `PreflightCheck` models."""
    check = _check()
    db_row = RemoteServerDB(
        id=str(uuid4()),
        name="server",
        ssh_host_alias="gpu-box",
        device_type=DeviceType.CUDA.value,
        last_check_status="healthy",
        last_check_checks=[check.model_dump(mode="json")],
    )

    remote_server = RemoteServerMapper.from_schema(db_row)

    assert remote_server.last_check_checks == [check]


def test_from_schema_defaults_last_check_checks_to_empty_list_when_null() -> None:
    """A never-checked server (column still ``NULL``) reads back as an empty list."""
    db_row = RemoteServerDB(
        id=str(uuid4()),
        name="server",
        ssh_host_alias="gpu-box",
        device_type=DeviceType.CUDA.value,
        last_check_status="unknown",
        last_check_checks=None,
    )

    remote_server = RemoteServerMapper.from_schema(db_row)

    assert remote_server.last_check_checks == []
