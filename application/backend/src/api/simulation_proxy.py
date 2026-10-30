# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A robot's simulation viewer and camera frames, passed on through Studio's own address.

See ``services.simulation_proxy``: in Docker the browser reaches only Studio's port, not the
simulation's own. Natively the same paths work the same way.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, WebSocket, status
from fastapi.responses import RedirectResponse
from loguru import logger
from starlette.responses import Response

from api.dependencies import RobotCatalogServiceDep, SimulationManagerDep
from db import get_async_db_session_ctx
from exceptions import SimulationNotServingError
from robots.catalog.registry import RobotCatalogRegistry
from services.robot_simulation_service import RobotSimulationService
from services.simulation_manager import SimulationManager
from services.simulation_proxy import is_proxied_http_path, is_viewer_path, proxy_get, proxy_websocket

router = APIRouter(prefix="/api/projects/{project_id}/robots/{robot_id}/simulation", tags=["Simulation"])

VIEWER_READ_TIMEOUT_S = 30.0
SNAPSHOT_READ_TIMEOUT_S = 10.0


async def _addresses(
    registry: RobotCatalogRegistry, manager: SimulationManager, project_id: UUID, robot_id: UUID
) -> tuple[str | None, str | None]:
    # A session of its own, closed before passing anything on: a viewer stays connected for long.
    async with get_async_db_session_ctx() as session:
        return await RobotSimulationService(session, registry, manager).proxy_addresses(project_id, robot_id)


@router.get("/viewer", include_in_schema=False)
async def simulation_viewer_root(request: Request) -> RedirectResponse:
    """Redirect to the viewer's page: it connects its WebSocket to the page's URL without the slash."""
    return RedirectResponse(f"{request.url.path}/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get("/viewer/{path:path}", include_in_schema=False)
async def simulation_viewer(
    project_id: UUID,
    robot_id: UUID,
    path: str,
    catalog: RobotCatalogServiceDep,
    manager: SimulationManagerDep,
) -> Response:
    """The page and files of the robot's simulation viewer."""
    if not is_viewer_path(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    _, viewer_url = await _addresses(catalog.registry, manager, project_id, robot_id)
    if viewer_url is None:
        raise SimulationNotServingError("The simulation serves no viewer on this machine.")
    return await proxy_get(f"{viewer_url}/{path}", read_timeout_s=VIEWER_READ_TIMEOUT_S)


@router.websocket("/viewer")
async def simulation_viewer_socket(
    websocket: WebSocket,
    project_id: UUID,
    robot_id: UUID,
    catalog: RobotCatalogServiceDep,
    manager: SimulationManagerDep,
) -> None:
    """The WebSocket of the robot's simulation viewer."""
    try:
        _, viewer_url = await _addresses(catalog.registry, manager, project_id, robot_id)
    except Exception as error:  # an unknown robot, or no simulation: the same refusal for the browser
        logger.debug("Not passing on the viewer of robot {}: {}", robot_id, error)
        await websocket.close(code=1008)
        return
    if viewer_url is None:
        await websocket.close(code=1008)
        return
    await proxy_websocket(websocket, "ws" + viewer_url.removeprefix("http"))


@router.get("/http/{path:path}", include_in_schema=False)
async def simulation_http(
    project_id: UUID,
    robot_id: UUID,
    path: str,
    catalog: RobotCatalogServiceDep,
    manager: SimulationManagerDep,
) -> Response:
    """A camera frame or stream of the robot's simulation; nothing else of its HTTP API."""
    api_path = f"/{path}"
    if not is_proxied_http_path(api_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    http_url, _ = await _addresses(catalog.registry, manager, project_id, robot_id)
    if http_url is None:
        raise SimulationNotServingError("The simulation serves no HTTP API on this machine.")
    streaming = api_path.endswith("/mjpeg")
    return await proxy_get(f"{http_url}{api_path}", read_timeout_s=None if streaming else SNAPSHOT_READ_TIMEOUT_S)
