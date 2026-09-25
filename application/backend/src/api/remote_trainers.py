from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, status

from api.dependencies import get_remote_trainer_service
from schemas.remote_trainer import RemoteTrainer, RemoteTrainerCreate, RemoteTrainerHealth, RemoteTrainerUpdate
from services.remote_trainer_service import RemoteTrainerService

router = APIRouter(prefix="/api/remote-trainers", tags=["Remote trainers"])


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
) -> RemoteTrainer:
    """Persist a remote trainer endpoint."""
    return await remote_trainer_service.create_remote_trainer(config, accepted_host_key_fingerprint)


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
) -> None:
    """Delete a trainer with no queued or running managed SSH jobs."""
    await remote_trainer_service.delete_remote_trainer(remote_trainer_id)
