import os

import uvicorn
from fastapi import FastAPI

from api.hardware import router as hardware_router
from api.project import router as project_router

app = FastAPI(title="Geti Action", openapi_url="/api/openapi.json")
app.include_router(project_router, prefix="/api/projects")
app.include_router(hardware_router, prefix="/api/hardware")


if __name__ == "__main__":
    uvicorn_port = int(os.environ.get("HTTP_SERVER_PORT", "7860"))
    uvicorn.run("main:app", host="0.0.0.0", port=uvicorn_port)  # noqa: S104
