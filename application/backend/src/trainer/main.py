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
from trainer.log_setup import setup_logging, setup_uvicorn_logging
from trainer.queue_worker import QueueManager
from trainer.schemas import DeviceInfo, HealthInfo, StorageInfo
from trainer.settings import get_settings
from trainer.storage import get_storage_info

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Configure logging at import time so it applies regardless of whether the
# app is served via `physicalai-trainer`, `uvicorn trainer.main:app`, or the
# `__main__` block below.
setup_logging()
setup_uvicorn_logging()


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


@app.get("/health", response_model=HealthInfo, response_model_by_alias=False)
async def health() -> HealthInfo:
    """Return liveness plus non-sensitive image compatibility metadata."""
    return HealthInfo()  # type: ignore[call-arg]


@app.get("/devices", response_model=list[DeviceInfo])
async def devices() -> list[DeviceInfo]:
    """Report the compute devices this trainer can use for training."""
    return get_training_devices()


@app.get("/storage")
async def storage() -> StorageInfo:
    """Report the available storage capacity on this trainer's storage volume."""
    return get_storage_info()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=int(os.environ.get("TRAINER_PORT", settings.port)),
        log_config=None,
    )
