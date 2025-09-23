import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

#from app.core.scheduler import Scheduler
#from app.db import MigrationManager
from settings import get_settings
from webrtc.manager import WebRTCManager

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
    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Shutting down %s application...", settings.app_name)
    await webrtc_manager.cleanup()
    logger.info("Application shutdown completed")