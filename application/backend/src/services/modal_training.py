# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pinned Modal resources and connectivity checks (credentials are SDK-managed)."""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlsplit

from loguru import logger

from schemas.remote_trainer import ModalConnection, RemoteTrainerHealth
from training.policy_catalog import PolicyCatalog, declared_policy_catalog

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    import modal


def modal_trainer_url(connection: ModalConnection) -> str:
    """Pin all non-secret resources so edits cannot redirect an existing job."""
    return f"modal://trainer?{urlencode({'connection': connection.model_dump_json()})}"


def modal_connection_from_url(url: str) -> ModalConnection:
    """Validate the resources of a previously submitted Modal job."""
    location = urlsplit(url)
    if location.scheme != "modal":
        raise ValueError("Expected a modal trainer URL")
    try:
        return ModalConnection.model_validate_json(parse_qs(location.query, strict_parsing=True)["connection"][0])
    except KeyError as error:
        raise ValueError("Modal trainer URL is missing connection resources") from error


def modal_policy_catalog() -> PolicyCatalog:
    """Offer all Studio policies without checking the deployed image or GPU."""
    return declared_policy_catalog(
        {
            "act": ["torch", "openvino", "onnx", "executorch"],
            "molmoact2": ["torch", "openvino"],
            "pi05": ["torch", "openvino"],
            "rldx1": ["torch", "onnx", "openvino"],
            "smolvla": ["torch", "openvino"],
            "xr0": ["torch", "openvino"],
        }
    )


def modal_resources(connection: ModalConnection) -> tuple[modal.Function, modal.Volume]:
    """Look up resources without creating missing deployments or storing credentials."""
    import modal

    return (
        modal.Function.from_name(
            connection.app_name, connection.function_name, environment_name=connection.environment_name
        ),
        modal.Volume.from_name(connection.volume_name, environment_name=connection.environment_name),
    )


def read_volume_file(volume: modal.Volume, path: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes | None:
    """Read a bounded status/metrics snapshot; an unpublished file is not an error."""
    import modal

    contents = bytearray()
    try:
        for chunk in volume.read_file(path):
            contents.extend(chunk)
            if len(contents) > max_bytes:
                raise ValueError(f"Modal snapshot exceeds {max_bytes} bytes")
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    return bytes(contents)


async def probe_modal(
    remote_trainer_id: UUID, connection: ModalConnection, checked_at: datetime
) -> RemoteTrainerHealth:
    """Check function and Volume access without starting billable GPU compute."""
    import modal

    started = perf_counter()
    status, reason = "healthy", None

    def probe() -> None:
        function, volume = modal_resources(connection)
        function.hydrate()
        volume.hydrate()

    try:
        await asyncio.wait_for(asyncio.to_thread(probe), timeout=10)
    except TimeoutError:
        status, reason = "unreachable", "timeout"
    except (modal.exception.AuthError, modal.exception.PermissionDeniedError):
        status, reason = "degraded", "authentication_required"
    except modal.exception.NotFoundError:
        status, reason = "degraded", "resource_missing"
    except Exception:
        logger.opt(exception=True).debug("Modal connectivity check failed")
        status, reason = "unreachable", "connection_failed"
    return RemoteTrainerHealth(
        remote_trainer_id=remote_trainer_id,
        status=status,
        checked_at=checked_at,
        latency_ms=round((perf_counter() - started) * 1000),
        devices=[],
        reason_code=reason,
        issues=[] if status == "healthy" else None,
    )
