from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from api.dependencies import CameraClaimRegistryDep, get_camera_id, get_camera_service, get_project_id
from exceptions import RecordingLockError
from schemas.project_camera import Camera
from services import ProjectCameraService
from services.camera_claims import CameraClaimRegistry

router = APIRouter(prefix="/api/projects/{project_id}/cameras", tags=["Project Cameras"])

ProjectID = Annotated[UUID, Depends(get_project_id)]


def _ensure_not_claimed(fingerprint: str, claims: CameraClaimRegistry) -> None:
    holder = claims.holder_of(fingerprint)
    if holder is not None:
        raise RecordingLockError(f"Camera is in use by project {holder.project_name!r}.")


@router.get("")
async def list_project_cameras(
    project_id: ProjectID,
    camera_service: Annotated[ProjectCameraService, Depends(get_camera_service)],
) -> list[Camera]:
    """Fetch all cameras."""
    return await camera_service.get_camera_list(project_id)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project_camera(
    project_id: ProjectID,
    camera: Camera,
    camera_service: Annotated[ProjectCameraService, Depends(get_camera_service)],
) -> Camera:
    """Create a new camera."""
    return await camera_service.create_camera(project_id, camera)


@router.get("/{camera_id}")
async def get_project_camera(
    project_id: ProjectID,
    camera_id: Annotated[UUID, Depends(get_camera_id)],
    camera_service: Annotated[ProjectCameraService, Depends(get_camera_service)],
) -> Camera:
    """Get camera by id."""
    return await camera_service.get_camera_by_id(project_id, camera_id)


@router.put("/{camera_id}")
async def update_project_camera(
    project_id: ProjectID,
    camera_id: Annotated[UUID, Depends(get_camera_id)],
    camera_service: Annotated[ProjectCameraService, Depends(get_camera_service)],
    camera: Camera,
    claims: CameraClaimRegistryDep,
) -> Camera:
    """Set camera."""
    existing = await camera_service.get_camera_by_id(project_id, camera_id)
    _ensure_not_claimed(existing.fingerprint, claims)
    _ensure_not_claimed(camera.fingerprint, claims)
    camera_with_id = camera.model_copy(update={"id": camera_id})

    return await camera_service.update_camera(
        project_id,
        camera_with_id,
    )


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_camera(
    project_id: ProjectID,
    camera_id: Annotated[UUID, Depends(get_camera_id)],
    camera_service: Annotated[ProjectCameraService, Depends(get_camera_service)],
    claims: CameraClaimRegistryDep,
) -> None:
    """Delete a camera."""
    existing = await camera_service.get_camera_by_id(project_id, camera_id)
    _ensure_not_claimed(existing.fingerprint, claims)
    await camera_service.delete_camera(project_id, camera_id)
