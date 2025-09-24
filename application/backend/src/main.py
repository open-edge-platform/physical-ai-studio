import os

import uvicorn
from fastapi import FastAPI

from api.camera import router as camera_router
from api.hardware import router as hardware_router
from api.project import router as project_router
from core import lifespan
from settings import get_settings

settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    openapi_url=settings.openapi_url,
    version=settings.version,
    description=settings.description,
    lifespan=lifespan
)

app.include_router(project_router, prefix="/api/projects")
app.include_router(hardware_router, prefix="/api/hardware")
app.include_router(camera_router, prefix="/api/cameras")


if __name__ == "__main__":
    uvicorn_port = int(os.environ.get("HTTP_SERVER_PORT", "7860"))
    uvicorn.run("main:app", host="0.0.0.0", port=uvicorn_port)  # noqa: S104
