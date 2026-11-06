# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The simulation Studio starts and supervises for a simulated robot (at most one at a time)."""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, status
from sse_starlette import EventSourceResponse

from api.dependencies import SimulationManagerDep
from schemas.simulation import Simulation, SimulationLogs, StartSimulationRequest

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])


@router.get("")
async def get_simulation(manager: SimulationManagerDep) -> Simulation:
    """Return the simulation Studio runs; ``idle`` when there is none."""
    return manager.state()


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start_simulation(request: StartSimulationRequest, manager: SimulationManagerDep) -> Simulation:
    """Start the simulation of a project's robot.

    Returns once the process runs, in the ``starting`` state; follow ``/api/simulation/events``
    until it is ``running`` or ``failed``.
    """
    return await manager.start(request)


@router.delete("")
async def stop_simulation(manager: SimulationManagerDep) -> Simulation:
    """Stop the simulation, if any, or clear a failed one. Returns ``stopping`` or ``idle``."""
    return await manager.stop()


@router.get("/events", response_class=EventSourceResponse)
async def simulation_events(manager: SimulationManagerDep) -> EventSourceResponse:
    """Stream the simulation's state: the current state first, then the full state on every change."""

    async def events() -> AsyncIterator[dict[str, Any]]:
        async with manager.subscribe() as queue:
            while True:
                state = await queue.get()
                yield {"data": state.model_dump_json()}

    return EventSourceResponse(events())


@router.get("/logs")
async def get_simulation_logs(manager: SimulationManagerDep) -> SimulationLogs:
    """Return the latest log lines of the current or last simulation."""
    return SimulationLogs(lines=manager.logs())
