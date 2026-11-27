import asyncio
import functools
import hashlib
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import FileResponse, Response
from physicalai_studio_plugin import RobotCatalogDefinition
from pydantic import BaseModel, Field

from api.dependencies import RobotCatalogServiceDep, RobotConnectionManagerDep
from exceptions import ResourceNotFoundError, ResourceType, RobotIdentifyError
from robots.catalog.assets import resolve_robot_relative_asset_path, resolve_robot_urdf_path
from schemas import SerialPortInfo


class RobotZeroCalibrationResponse(BaseModel):
    instructions: str = Field(..., description="Description of the zero pose shown in the calibration wizard")
    zero_tolerance_deg: float = Field(..., description="Largest joint reading, in degrees, that counts as zero")


class RobotSimulationSceneResponse(BaseModel):
    id: str = Field(..., description="Scene id to pass when starting the simulation")
    display_name: str = Field(..., description="Human-readable scene name")
    description: str = Field("", description="One-line description of the scene")


class RobotSimulationResponse(BaseModel):
    scenes: list[RobotSimulationSceneResponse] = Field(..., description="Scenes to choose from, in display order")
    default_scene: str = Field(..., description="Id of the scene to preselect")
    max_seed: int | None = Field(None, description="Largest fixed seed, or null when the simulation takes no seed")
    labels: list[str] = Field(default_factory=list, description="Read-only facts shown in the start dialog")


class RobotCatalogDefinitionResponse(BaseModel):
    type: str = Field(..., description="Stable backend robot type identifier")
    display_name: str = Field(..., description="Human-readable robot type label")
    role: Literal["follower", "leader"] = Field(..., description="Default robot role")
    urdf_path: str | None = Field(
        default=None,
        description="URDF URL used by the UI model loader, or null when the robot type has no 3D model",
    )
    package_map: dict[str, str] = Field(default_factory=dict, description="URDF package name to URL prefix map")
    joint_map: dict[str, list[str]] = Field(
        description="Observation joint name to URDF joint(s) mapping",
    )
    zero_calibration: RobotZeroCalibrationResponse | None = Field(
        default=None,
        description="Zero-pose calibration offered when adding this robot type, if any",
    )
    simulation: RobotSimulationResponse | None = Field(
        default=None,
        description="How Studio can start this robot type's simulation, if it is simulated",
    )


def _to_response(definition: RobotCatalogDefinition) -> RobotCatalogDefinitionResponse:
    catalog_root = f"/api/robots/catalog/{definition.type}"

    urdf_path = None
    package_map = {}
    joint_map = {}
    if definition.asset is not None:
        urdf_path = f"{catalog_root}/urdf"
        package_map = dict.fromkeys(definition.asset.packages, f"{catalog_root}")
        joint_map = definition.asset.joint_map

    return RobotCatalogDefinitionResponse(
        type=definition.type,
        display_name=definition.display_name,
        role=definition.role,
        urdf_path=urdf_path,
        package_map=package_map,
        joint_map=joint_map,
        zero_calibration=None
        if definition.zero_calibration is None
        else RobotZeroCalibrationResponse(
            instructions=definition.zero_calibration.instructions,
            zero_tolerance_deg=definition.zero_calibration.zero_tolerance_deg,
        ),
        simulation=_simulation_response(getattr(definition, "simulation", None)),
    )


def _simulation_response(launch: Any) -> RobotSimulationResponse | None:
    # `simulation` is missing on physicalai-studio-plugin releases without SimulationLaunch.
    if launch is None:
        return None
    return RobotSimulationResponse(
        scenes=[
            RobotSimulationSceneResponse(id=scene.id, display_name=scene.display_name, description=scene.description)
            for scene in launch.scenes
        ],
        default_scene=launch.default_scene,
        max_seed=launch.max_seed,
        labels=list(launch.labels),
    )


def _get_robot_definition(
    robot_type: str,
    catalog_service: RobotCatalogServiceDep,
) -> RobotCatalogDefinition:
    return catalog_service.get_definition(robot_type)


def _validate_probe_payload(
    payload: Annotated[dict[str, Any], Body(...)],
    definition: Annotated[RobotCatalogDefinition, Depends(_get_robot_definition)],
) -> BaseModel | dict[str, Any]:
    if definition.robot_payload is None:
        return payload
    return definition.robot_payload.model_validate(payload)


RobotDefinitionDep = Annotated[RobotCatalogDefinition, Depends(_get_robot_definition)]
ProbePayloadDep = Annotated[BaseModel | dict[str, Any], Depends(_validate_probe_payload)]


router = APIRouter(prefix="/api/robots/catalog", tags=["Robot Catalog"])


def _is_so101_overload_protection_error(error: RuntimeError) -> bool:
    """Return whether a plugin error represents the SO101 identify recovery action."""
    return "stopped responding during motion" in str(error) and "overload protection" in str(error)


@router.get("")
async def list_robot_catalog(catalog_service: RobotCatalogServiceDep) -> list[RobotCatalogDefinitionResponse]:
    """List robot catalog definitions exposed to the UI."""
    return [_to_response(definition) for definition in catalog_service.list_entries()]


@router.get("/{robot_type}/discover")  # noqa: FAST003 - consumed by RobotDefinitionDep via _get_robot_definition
async def discover_robots(
    definition: RobotDefinitionDep,
    robot_manager: RobotConnectionManagerDep,
) -> list[SerialPortInfo]:
    """Discover connected devices for a robot type."""
    if definition.probe is None:
        return []
    return await definition.probe.discover(robot_manager)


@router.post("/{robot_type}/identify")  # noqa: FAST003 - consumed by RobotDefinitionDep via _get_robot_definition
async def identify_robot(
    definition: RobotDefinitionDep,
    robot_manager: RobotConnectionManagerDep,
    payload: ProbePayloadDep,
    joint: str | None = None,
) -> None:
    """Visually identify a robot by moving a joint or gripper."""
    if definition.probe is None:
        raise ResourceNotFoundError(
            resource_type=ResourceType.ROBOT,
            resource_id=definition.type,
            message=f"Robot type {definition.type} does not support identification.",
        )
    try:
        await definition.probe.identify(payload, robot_manager, joint)
    except RuntimeError as error:
        if _is_so101_overload_protection_error(error):
            raise RobotIdentifyError(str(error)) from error
        raise


@router.post("/{robot_type}/is-online")  # noqa: FAST003 - consumed by RobotDefinitionDep via _get_robot_definition
async def check_robot_online(
    definition: RobotDefinitionDep,
    payload: ProbePayloadDep,
) -> bool:
    """Check if a robot is currently online/reachable."""
    if definition.probe is None:
        return False
    return await definition.probe.is_online(payload)


@functools.lru_cache(maxsize=1024)
def _content_etag(path: str, inode: int, size: int, mtime_ns: int, ctime_ns: int) -> str:  # noqa: ARG001 - cache key
    """A strong ETag from the file's bytes, computed once per version of the file.

    The stat fields are only the cache key. ctime is in it because on Linux and macOS a rewrite always changes it,
    even when the mtime is set back and the size is the same, which would fool an mtime-and-size ETag.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return f'"{digest.hexdigest()[:32]}"'


def _stat_and_etag(path: Path) -> tuple[os.stat_result, str]:
    """Stat ``path`` and return its content ETag, from the cache where the stat fields can key it.

    On Windows ``st_ctime_ns`` is the creation time, which an in-place rewrite keeps, so a same-size rewrite with its
    mtime set back would keep the cache key and the old ETag. There the file is hashed on every request.
    """
    stat = path.stat()
    key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if sys.platform == "win32":
        return stat, _content_etag.__wrapped__(*key)
    return stat, _content_etag(*key)


def _matches_if_none_match(if_none_match: str, etag: str) -> bool:
    """Whether ``If-None-Match`` names ``etag``, by the weak comparison GET uses (RFC 9110 13.1.2).

    ``*`` matches any file that exists, and a ``W/`` prefix is ignored on either side.
    """
    tags = if_none_match.strip()
    if tags == "*":
        return True
    opaque = etag.removeprefix("W/")
    return any(tag.strip().removeprefix("W/") == opaque for tag in tags.split(","))


async def _revalidated_file(request: Request, path: Path) -> Response:
    """Serve a URDF or mesh that the browser must check before reusing: ``no-cache`` plus a content ETag.

    Without ``Cache-Control`` browsers keep these for hours by their own heuristic, so a mesh that changed on disk
    (a plugin update, or an LFS pointer replaced by the real file) kept showing the old one. A 304 answers a matching
    ``If-None-Match``, so the check costs a round trip, not the file. ``If-Modified-Since`` alone is answered with
    the file: an mtime can be set back, the content ETag cannot.
    """
    stat, etag = await asyncio.to_thread(_stat_and_etag, path)
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if_none_match = request.headers.get("if-none-match")
    if if_none_match is not None and _matches_if_none_match(if_none_match, etag):
        return Response(status_code=304, headers=headers)
    return FileResponse(path, headers=headers, stat_result=stat)


@router.get("/{robot_type}/urdf")  # noqa: FAST003 - consumed by RobotDefinitionDep via _get_robot_definition
async def get_robot_catalog_urdf(
    request: Request,
    definition: RobotDefinitionDep,
) -> Response:
    """Return the URDF file for a catalog robot type."""
    return await _revalidated_file(request, resolve_robot_urdf_path(definition))


@router.get("/{robot_type}/schema")  # noqa: FAST003 - consumed by RobotDefinitionDep via _get_robot_definition
async def get_robot_catalog_schema(
    definition: RobotDefinitionDep,
) -> dict[str, Any]:
    """Return the Pydantic JSON Schema for a catalog robot payload."""
    payload = definition.robot_payload
    if payload is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Robot type {definition.type} has no payload schema")
    return payload.model_json_schema()


@router.get("/{robot_type}/{asset_path:path}")  # noqa: FAST003 - consumed by RobotDefinitionDep via _get_robot_definition
async def get_robot_catalog_asset(
    request: Request,
    definition: RobotDefinitionDep,
    asset_path: Path,
) -> Response:
    """Return an asset referenced by a relative URDF URI."""
    return await _revalidated_file(request, resolve_robot_relative_asset_path(definition, asset_path=asset_path))
