from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import CameraClaimRegistryDep, RobotServiceDep, SimulationManagerDep, get_camera_service
from exceptions import RecordingLockError, ResourceNotFoundError
from schemas.project_camera import Camera, IPCamera
from services import ProjectCameraService, RobotService
from services.camera_claims import CameraClaimRegistry

router = APIRouter(prefix="/api/projects/{project_id}/cameras", tags=["Project Cameras"])

ProjectID = UUID


def _require_fingerprint(fingerprint: dict[str, Any] | None) -> dict[str, Any]:
    if not fingerprint:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Camera must be reselected")
    return fingerprint


def _ensure_not_claimed(fingerprint: dict[str, Any] | None, claims: CameraClaimRegistry) -> None:
    if fingerprint is None:
        return
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
    """Create a new camera.

    An IP camera's simulated robot (``payload.simulation_robot_id``) is not taken from the request:
    only adding a simulation's cameras sets it.
    """
    _require_fingerprint(camera.fingerprint)
    if isinstance(camera, IPCamera):
        camera = camera.model_copy(update={"payload": camera.payload.model_copy(update={"simulation_robot_id": None})})
    return await camera_service.create_camera(project_id, camera)


@router.get("/{camera_id}")
async def get_project_camera(
    project_id: ProjectID,
    camera_id: UUID,
    camera_service: Annotated[ProjectCameraService, Depends(get_camera_service)],
) -> Camera:
    """Get camera by id."""
    return await camera_service.get_camera_by_id(project_id, camera_id)


@router.put("/{camera_id}")
async def update_project_camera(
    project_id: ProjectID,
    camera_id: UUID,
    camera_service: Annotated[ProjectCameraService, Depends(get_camera_service)],
    camera: Camera,
    claims: CameraClaimRegistryDep,
    robot_service: RobotServiceDep,
    simulations: SimulationManagerDep,
) -> Camera:
    """Set camera.

    An IP camera added from a simulation keeps its simulated robot (``payload.simulation_robot_id``,
    which Studio sets, not the request). While Studio starts, runs or stops that robot's simulation,
    the camera also keeps the stream URL Studio stored: a form opened before the simulation moved
    still carries the old one. Otherwise a URL edited to point away from the robot's simulation drops
    the robot, so that restarting its simulation leaves the camera be.
    """
    existing = await camera_service.get_camera_by_id(project_id, camera_id)
    _ensure_not_claimed(existing.fingerprint, claims)
    _ensure_not_claimed(_require_fingerprint(camera.fingerprint), claims)
    camera_with_id = camera.model_copy(update={"id": camera_id})
    if not isinstance(camera_with_id, IPCamera):
        return await camera_service.update_camera(project_id, camera_with_id)

    # The lock Studio moves and adopts cameras under: decided on the camera as stored then.
    async with simulations.holding_addresses() as simulated_by_studio:
        camera_service.session.expire_all()  # re-read below, not from this session's cache
        existing = await camera_service.get_camera_by_id(project_id, camera_id)
        owner = existing.payload.simulation_robot_id if isinstance(existing, IPCamera) else None
        if isinstance(existing, IPCamera) and owner is not None and simulated_by_studio(project_id, owner):
            payload = camera_with_id.payload.model_copy(
                update={"url": existing.payload.url, "simulation_robot_id": owner}
            )
            camera_with_id = camera_with_id.model_copy(update={"payload": payload, "fingerprint": existing.fingerprint})
        else:
            owner = await _simulation_owner(robot_service, project_id, existing, camera_with_id)
            payload = camera_with_id.payload.model_copy(update={"simulation_robot_id": owner})
            camera_with_id = camera_with_id.model_copy(update={"payload": payload})
        return await camera_service.update_camera(project_id, camera_with_id)


async def _simulation_owner(
    robot_service: RobotService, project_id: UUID, existing: Camera, camera: IPCamera
) -> UUID | None:
    """Return the simulated robot an edited IP camera keeps: its own, while it streams from it."""
    if not isinstance(existing, IPCamera) or existing.payload.simulation_robot_id is None:
        return None
    owner = existing.payload.simulation_robot_id
    if camera.payload.url == existing.payload.url:
        return owner
    try:
        robot = await robot_service.get_robot_by_id(project_id, owner)
    except ResourceNotFoundError:
        return None
    http_url = getattr(robot.payload, "http_url", None)
    if isinstance(http_url, str) and camera.payload.url.startswith(f"{http_url}/cameras/"):
        return owner
    return None


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_camera(
    project_id: ProjectID,
    camera_id: UUID,
    camera_service: Annotated[ProjectCameraService, Depends(get_camera_service)],
    claims: CameraClaimRegistryDep,
) -> None:
    """Delete a camera."""
    existing = await camera_service.get_camera_by_id(project_id, camera_id)
    _ensure_not_claimed(existing.fingerprint, claims)
    await camera_service.delete_camera(project_id, camera_id)
