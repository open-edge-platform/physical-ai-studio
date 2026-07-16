# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer service entrypoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from loguru import logger

from trainer.api import router as jobs_router
from trainer.devices import get_training_devices
from trainer.queue_worker import QueueManager
from trainer.schemas import DeviceInfo
from trainer.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Create storage dirs and start the queue manager."""
    settings = get_settings()
    for directory in (settings.datasets_dir, settings.models_dir, settings.archives_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manager = QueueManager()
    await manager.start()
    app.state.queue_manager = manager
    logger.info("Trainer service ready on {}:{}", settings.host, settings.port)

    yield

    await manager.shutdown()


app = FastAPI(title="Physical AI Trainer", lifespan=lifespan)
app.include_router(jobs_router)


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "healthy"}


@app.get("/devices", response_model=list[DeviceInfo])
async def devices() -> list[DeviceInfo]:
    """Report the compute devices this trainer can use for training."""
    return get_training_devices()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=int(os.environ.get("PORT", settings.port)))
