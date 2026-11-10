# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""User-confirmed deployments of the bundled Modal trainer, tracked through health."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from loguru import logger

from schemas.operational_issue import OperationalIssue
from schemas.remote_trainer import ModalConnection, ModalDeploymentRequest, RemoteTrainerHealth


@dataclass
class _Deployment:
    message: str = "Checking Modal credentials and resource names"
    failed: bool = False
    task: asyncio.Task[None] | None = None


# Like managed SSH setup, progress is process-local; resources and saved targets survive restarts.
_deployments: dict[UUID, _Deployment] = {}


def new_modal_connection(environment_name: str) -> ModalConnection:
    """Use unpredictable resource names instead of accepting an app to overwrite."""
    suffix = uuid4().hex
    return ModalConnection(
        app_name=f"physicalai-{suffix}",
        volume_name=f"physicalai-training-{suffix}",
        environment_name=environment_name,
    )


def modal_deployment_health(trainer_id: UUID) -> RemoteTrainerHealth | None:
    """Report setup progress or a sanitized error before probing the deployed function."""
    deployment = _deployments.get(trainer_id)
    if deployment is None:
        return None
    return RemoteTrainerHealth(
        remote_trainer_id=trainer_id,
        status="degraded" if deployment.failed else "starting",
        checked_at=datetime.now(UTC),
        reason_code=deployment.message,
        issues=[
            OperationalIssue(id="modal-deployment-failed", title="Modal deployment failed", detail=deployment.message)
        ]
        if deployment.failed
        else [],
    )


def forget_modal_deployment(trainer_id: UUID) -> None:
    """Forget deployment progress and cancel any local deployment task."""
    deployment = _deployments.pop(trainer_id, None)
    if deployment and deployment.task and not deployment.task.done():
        deployment.task.cancel()


async def stop_modal_deployment(trainer_id: UUID) -> None:
    """Wait for the local deployment command to stop before cleaning up its resources."""
    deployment = _deployments.get(trainer_id)
    task = deployment.task if deployment else None
    forget_modal_deployment(trainer_id)
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)


def start_modal_deployment(trainer_id: UUID, connection: ModalConnection, request: ModalDeploymentRequest) -> None:
    """Launch explicitly confirmed setup without holding a database session open."""
    deployment = _Deployment()
    _deployments[trainer_id] = deployment
    deployment.task = asyncio.create_task(_deploy(trainer_id, connection, request, deployment))


async def _check_resources(connection: ModalConnection, hf_secret_name: str | None) -> None:
    import modal

    # Fail closed on authentication/transport errors, not just missing resources.
    try:
        await modal.App.lookup.aio(connection.app_name, environment_name=connection.environment_name)
    except modal.exception.NotFoundError:
        pass
    else:
        raise FileExistsError("The generated Modal app already exists. Nothing was overwritten.")
    try:
        await modal.Volume.from_name(connection.volume_name, environment_name=connection.environment_name).hydrate.aio()
    except modal.exception.NotFoundError:
        pass
    else:
        raise FileExistsError("The generated Modal Volume already exists. Nothing was overwritten.")
    if hf_secret_name:
        await modal.Secret.from_name(hf_secret_name, environment_name=connection.environment_name).hydrate.aio()


async def _deploy(
    trainer_id: UUID, connection: ModalConnection, request: ModalDeploymentRequest, deployment: _Deployment
) -> None:
    import modal

    process: asyncio.subprocess.Process | None = None
    try:
        await _check_resources(connection, request.hf_secret_name)
        deployment.message = "Building and deploying the NVIDIA trainer. The first deployment can take several minutes."
        environment = {
            **os.environ,
            "PHYSICALAI_MODAL_APP": connection.app_name,
            "PHYSICALAI_MODAL_VOLUME": connection.volume_name,
            "PHYSICALAI_MODAL_SECRET": request.hf_secret_name or "",
            "PHYSICALAI_MODAL_IMAGE": "ghcr.io/open-edge-platform/physicalai-trainer-cuda:main",
        }
        environment.pop("PHYSICALAI_MODAL_GPU", None)
        script = Path(__file__).resolve().parents[1] / "trainer" / "modal_app.py"
        # Per-process environment avoids changing another deployment's app or secret.
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "modal",
            "deploy",
            "--env",
            connection.environment_name,
            str(script),
            env=environment,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        async with asyncio.timeout(30 * 60):
            code = await process.wait()
        if code:
            raise RuntimeError("The Modal deployment command failed")
        function = modal.Function.from_name(
            connection.app_name, connection.function_name, environment_name=connection.environment_name
        )
        await function.hydrate.aio()
        _deployments.pop(trainer_id, None)
    except asyncio.CancelledError:
        # A stopped client cannot roll back a build already accepted by Modal.
        deployment.failed = True
        deployment.message = "Deployment was interrupted. Check Modal for remaining resources before deploying again."
        raise
    except (modal.exception.AuthError, modal.exception.PermissionDeniedError):
        deployment.failed = True
        deployment.message = (
            "Modal authentication failed. Configure the Studio backend's Modal credentials and permissions."
        )
    except modal.exception.NotFoundError:
        deployment.failed = True
        deployment.message = "Modal environment or Hugging Face secret was not found. Check the names and workspace."
    except FileExistsError as error:
        deployment.failed = True
        deployment.message = str(error)
    except Exception:
        deployment.failed = True
        deployment.message = (
            "Deployment failed or timed out. Check the Modal dashboard for build errors and remaining resources. "
            "The saved target is retained; no existing app was intentionally replaced."
        )
        logger.exception("Modal deployment failed for trainer {}", trainer_id)
    finally:
        if process and process.returncode is None:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass
        deployment.task = None
