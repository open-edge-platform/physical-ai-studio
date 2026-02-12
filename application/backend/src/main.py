# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.camera import router as camera_router
from api.dataset import router as dataset_router
from api.dependencies import CameraRegistryDep, RobotRegistryDep
from api.environments import router as project_environments_router
from api.hardware import router as hardware_router
from api.job import router as job_router
from api.models import router as models_router
from api.project import router as project_router
from api.project_camera import router as project_cameras_router
from api.record import router as record_router
from api.robot_calibration import router as robot_calibration_router
from api.robot_control import router as robot_control_router
from api.robots import router as project_robots_router
from api.settings import router as settings_router
from api.webui import webui_router
from core import lifespan
from exception_handlers import register_application_exception_handlers
from settings import get_settings
from utils.device import get_torch_device

settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    openapi_url=settings.openapi_url,
    version=settings.version,
    description=settings.description,
    lifespan=lifespan,
)

app.include_router(project_router)
app.include_router(project_robots_router)
app.include_router(project_cameras_router)
app.include_router(robot_calibration_router)
app.include_router(robot_control_router)
app.include_router(project_environments_router)
app.include_router(hardware_router)
app.include_router(camera_router)
app.include_router(dataset_router)
app.include_router(record_router)
app.include_router(settings_router)
app.include_router(models_router)
app.include_router(job_router)

register_application_exception_handlers(app)


@app.get("/api/health")
async def health_check(camera_registry: CameraRegistryDep, robot_registry: RobotRegistryDep) -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "camera_workers": camera_registry.get_status_summary(),
        "robot_workers": robot_registry.get_status_summary(),
    }


# In docker deployment, the UI is built and served statically
if (
    settings.static_files_dir
    and Path(settings.static_files_dir).is_dir()
    and (Path(settings.static_files_dir) / "index.html").exists()
):
    static_dir = Path(settings.static_files_dir)
    app.mount("/static", StaticFiles(directory=static_dir / "static"), name="static")
    app.include_router(webui_router)

if __name__ == "__main__":
    if get_torch_device() == "xpu" and mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    uvicorn_port = int(os.environ.get("HTTP_SERVER_PORT", settings.port))
    uvicorn.run("main:app", host=settings.host, port=uvicorn_port)
