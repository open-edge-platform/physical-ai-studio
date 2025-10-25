from loguru import logger
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from settings import get_settings
from webrtc.manager import WebRTCManager

from .scheduler import Scheduler




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
    app_scheduler.start_workers()
    app.state.scheduler = app_scheduler
    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Shutting down %s application...", settings.app_name)
    await webrtc_manager.cleanup()
    app_scheduler.shutdown()
    logger.info("Application shutdown completed")
