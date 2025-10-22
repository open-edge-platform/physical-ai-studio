import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

# from app.core.scheduler import Scheduler
from db import MigrationManager
from settings import get_settings
from webrtc.manager import WebRTCManager

from .scheduler import Scheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    settings = get_settings()
    app.state.settings = settings
    logger.info("Starting %s application...", settings.app_name)
    webrtc_manager = WebRTCManager()
    app.state.webrtc_manager = webrtc_manager

    app_scheduler = Scheduler()
    app.state.scheduler = app_scheduler
    logger.info("Application startup completed")

    migration_manager = MigrationManager(settings)
    if not migration_manager.initialize_database():
        logger.error("Failed to initialize database. Application cannot start.")
        raise RuntimeError("Database initialization failed")

    yield

    # Shutdown
    logger.info("Shutting down %s application...", settings.app_name)
    await webrtc_manager.cleanup()
    app_scheduler.shutdown()
    logger.info("Application shutdown completed")
