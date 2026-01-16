from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from services.event_processor import EventProcessor
from settings import get_settings
from webrtc.manager import WebRTCManager
from workers.camera_worker_registry import CameraWorkerRegistry

from .scheduler import Scheduler


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    settings = get_settings()
    app.state.settings = settings

    app.state.camera_registry = CameraWorkerRegistry(
        max_workers=10,
        shutdown_timeout_s=10.0,
    )
    logger.info("Starting %s application...", settings.app_name)
    webrtc_manager = WebRTCManager()
    app.state.webrtc_manager = webrtc_manager

    app_scheduler = Scheduler()
    app_scheduler.start_workers()
    app.state.scheduler = app_scheduler
    app.state.event_processor = EventProcessor(app_scheduler.event_queue)
    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Shutting down %s application...", settings.app_name)
    await webrtc_manager.cleanup()

    camera_registry: CameraWorkerRegistry = app.state.camera_registry
    await camera_registry.shutdown_all()

    app_scheduler.shutdown()
    app.state.event_processor.shutdown()
    logger.info("Application shutdown completed")
