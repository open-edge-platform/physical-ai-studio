import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, status
from fastapi.responses import JSONResponse

from api.aws_auth import router as aws_auth_router
from api.dependencies import get_remote_trainer_service
from schemas.operational_issue import OperationalIssue
from schemas.remote_trainer import (
    AwsBatchConnection,
    AwsBatchProviderConfiguration,
    CloudProviderField,
    CloudProviderSchema,
    RemoteTrainer,
    RemoteTrainerCreate,
    RemoteTrainerHealth,
    RemoteTrainerUpdate,
)
from services.aws_auth_service import operational_aws_error
from services.aws_batch import resolve_aws_batch_configuration
from services.remote_trainer_service import RemoteTrainerService

router = APIRouter(prefix="/api/remote-trainers", tags=["Remote trainers"])
router.include_router(aws_auth_router)


@router.get("/providers")
async def list_cloud_provider_schemas() -> list[CloudProviderSchema]:
    """Return the backend-owned configuration fields for supported providers."""
    schema = AwsBatchProviderConfiguration.model_json_schema()
    properties = schema["properties"]
    return [
        CloudProviderSchema(
            fields=[
                CloudProviderField.model_validate({"name": name, **field})
                for name, field in properties.items()
                if name == "configuration_uri"
            ]
        )
    ]


@router.post(
    "/providers/aws/configuration", response_model=AwsBatchConnection, responses={400: {"model": OperationalIssue}}
)
async def resolve_cloud_provider_configuration(
    configuration: AwsBatchProviderConfiguration,
) -> AwsBatchConnection | JSONResponse:
    """Resolve provisioned resources before saving an AWS provider training target."""
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        return await asyncio.wait_for(asyncio.to_thread(resolve_aws_batch_configuration, configuration), timeout=30)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content=OperationalIssue(
                id="provider-configuration-invalid",
                title="Invalid AWS configuration",
                detail="The S3 configuration document is invalid or exceeds 64 KiB.",
            ).model_dump(),
        )
    except (BotoCoreError, ClientError, TimeoutError) as error:
        return JSONResponse(
            status_code=400, content=operational_aws_error(error, configuration.profile_name).model_dump()
        )


@router.get("")
async def list_remote_trainers(
    remote_trainer_service: Annotated[RemoteTrainerService, Depends(get_remote_trainer_service)],
) -> list[RemoteTrainer]:
    """Return configured remote trainers."""
    return await remote_trainer_service.list_remote_trainers()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_remote_trainer(
    config: RemoteTrainerCreate,
    remote_trainer_service: Annotated[RemoteTrainerService, Depends(get_remote_trainer_service)],
    accepted_host_key_fingerprint: Annotated[str | None, Header()] = None,
    install_prerequisites: bool = False,
) -> RemoteTrainer:
    """Persist a remote trainer endpoint, optionally installing SSH prerequisites first."""
    return await remote_trainer_service.create_remote_trainer(
        config, accepted_host_key_fingerprint, install_prerequisites
    )


@router.post("/{remote_trainer_id}/install-prerequisites", status_code=status.HTTP_202_ACCEPTED)
async def install_remote_trainer_prerequisites(
    remote_trainer_id: UUID,
    remote_trainer_service: Annotated[RemoteTrainerService, Depends(get_remote_trainer_service)],
) -> None:
    """Explicitly install SSH host prerequisites; poll trainer health for the result."""
    await remote_trainer_service.install_remote_trainer(remote_trainer_id)


@router.post("/{remote_trainer_id}/reboot-after-install", status_code=status.HTTP_202_ACCEPTED)
async def reboot_remote_trainer_after_install(
    remote_trainer_id: UUID,
    remote_trainer_service: Annotated[RemoteTrainerService, Depends(get_remote_trainer_service)],
) -> None:
    """Confirm the reboot required by a completed host installation."""
    await remote_trainer_service.reboot_installed_host(remote_trainer_id)


@router.get("/{remote_trainer_id}/health")
async def check_remote_trainer(
    remote_trainer_id: UUID,
    remote_trainer_service: Annotated[RemoteTrainerService, Depends(get_remote_trainer_service)],
) -> RemoteTrainerHealth:
    """Check a configured trainer's health and compute-device report."""
    return await remote_trainer_service.check_remote_trainer(remote_trainer_id)


@router.patch("/{remote_trainer_id}")
async def update_remote_trainer(
    remote_trainer_id: UUID,
    update: RemoteTrainerUpdate,
    remote_trainer_service: Annotated[RemoteTrainerService, Depends(get_remote_trainer_service)],
    accepted_host_key_fingerprint: Annotated[str | None, Header()] = None,
) -> RemoteTrainer:
    """Update a configured remote trainer endpoint."""
    return await remote_trainer_service.update_remote_trainer(remote_trainer_id, update, accepted_host_key_fingerprint)


@router.delete("/{remote_trainer_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_remote_trainer(
    remote_trainer_id: UUID,
    remote_trainer_service: Annotated[RemoteTrainerService, Depends(get_remote_trainer_service)],
    forget_locally: bool = False,
) -> None:
    """Delete a trainer with no queued or running managed SSH jobs.

    Forget locally skips remote container and volume cleanup when the SSH host is gone.
    """
    await remote_trainer_service.delete_remote_trainer(remote_trainer_id, forget_locally=forget_locally)
