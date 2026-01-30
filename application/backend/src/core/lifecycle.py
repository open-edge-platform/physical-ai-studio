from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from services.event_processor import EventProcessor
from settings import get_settings
from utils.robot import RobotConnectionManager
from workers.camera_worker_registry import CameraWorkerRegistry
from workers.robot_worker_registry import RobotWorkerRegistry

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
    app.state.robot_registry = RobotWorkerRegistry(
        max_workers=10,
        shutdown_timeout_s=10.0,
    )

    logger.info("Starting %s application...", settings.app_name)
    app_scheduler = Scheduler()
    app_scheduler.start_workers()
    app.state.scheduler = app_scheduler
    app.state.event_processor = EventProcessor(app_scheduler.event_queue)
    logger.info("Application startup completed")

    # Initialize RobotHardwareManager
    app.state.robot_manager = RobotConnectionManager()
    await app.state.robot_manager.find_robots()

    yield

    # Shutdown
    logger.info("Shutting down %s application...", settings.app_name)

    camera_registry: CameraWorkerRegistry = app.state.camera_registry
    await camera_registry.shutdown_all()

    robot_registry: RobotWorkerRegistry = app.state.robot_registry
    await robot_registry.shutdown_all()

    # We might want to shutdown the hardware manager too, though releasing workers should handle it.
    # But a global cleanup is safe.
    # Ideally RobotHardwareManager would have a shutdown_all method too.
    # For now, we assume active workers unregistering will trigger releases.

    app_scheduler.shutdown()
    app.state.event_processor.shutdown()
    logger.info("Application shutdown completed")
