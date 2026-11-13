# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A simulated robot's simulation: what it serves, and adding its cameras to an environment."""

from uuid import UUID

from fastapi import APIRouter, status
from fastapi.requests import HTTPConnection
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RobotSimulationServiceDep, get_robot_catalog_service, get_simulation_manager
from schemas.project_camera import Camera, SimulationCamera
from schemas.simulation import AddSimulationCamerasRequest, ProjectRobotSimulation, RobotSimulation
from services.robot_simulation_service import RobotSimulationService

router = APIRouter(prefix="/api/projects/{project_id}", tags=["Simulation"])

ProjectID = UUID


@router.get("/robots/{robot_id}/simulation")
async def get_robot_simulation(
    project_id: ProjectID,
    robot_id: UUID,
    service: RobotSimulationServiceDep,
) -> RobotSimulation:
    """Whether the robot's simulation runs, who started it, and its viewer, speed and cameras."""
    return await service.get(project_id, robot_id)


@router.get("/simulations")
async def list_project_simulations(
    project_id: ProjectID, service: RobotSimulationServiceDep
) -> list[ProjectRobotSimulation]:
    """The running simulations of the project's simulated robots, started by Studio or elsewhere."""
    return await service.list_running(project_id)


@router.post("/environments/{environment_id}/simulation-cameras", status_code=status.HTTP_201_CREATED)
async def add_simulation_cameras(
    project_id: ProjectID,
    environment_id: UUID,
    request: AddSimulationCamerasRequest,
    service: RobotSimulationServiceDep,
) -> list[SimulationCamera]:
    """Add cameras of a running simulation to an environment; they stream while it runs."""
    return await service.add_cameras(project_id, environment_id, request.robot_id, request.cameras, request.names)


async def resolve_export_cameras(
    request: HTTPConnection, session: AsyncSession, project_id: UUID, cameras: list[Camera]
) -> list[Camera]:
    """Return *cameras* for a runtime export, simulation cameras on the streams they have now.

    The simulation manager is looked up only if there is a simulation camera.
    """
    if not any(isinstance(camera, SimulationCamera) for camera in cameras):
        return cameras
    service = RobotSimulationService(session, get_robot_catalog_service().registry, get_simulation_manager(request))
    return await service.resolve_cameras(project_id, cameras)
