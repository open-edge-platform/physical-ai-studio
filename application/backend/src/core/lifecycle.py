from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from core.logging import setup_logging, setup_uvicorn_logging
from core.security import get_ssh_feature_availability
from services.event_processor import EventProcessor
from settings import get_settings
from utils.multiprocessing import ensure_spawn_start_method
from utils.serial_robot_tools import RobotConnectionManager
from workers.model_worker_registry import ModelWorkerRegistry

from .scheduler import Scheduler


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    setup_logging()
    setup_uvicorn_logging()

    settings = get_settings()
    app.state.settings = settings

    # Fail the SSH remote-trainer feature closed if it is enabled on a
    # non-loopback bind address: the feature has no authentication model, so a
    # network-exposed instance would let anyone who can reach the API execute
    # code as root on every registered server. This never touches local or
    # direct-URL training - only routes/dependencies gated behind
    # `require_ssh_feature_active` (see `api.dependencies`) are affected.
    ssh_feature_availability = get_ssh_feature_availability()
    app.state.ssh_feature_availability = ssh_feature_availability
    if ssh_feature_availability.enabled and ssh_feature_availability.network_exposed:
        logger.critical(
            "SSH remote-trainer feature disabled at startup: {}. Bind the backend to a loopback "
            "address (127.0.0.1 or ::1) to use it.",
            ssh_feature_availability.reason,
        )

    # Camera fingerprints locked by an active recording/teleop session.
    # Mutated by the robot_control WS handler; checked by camera CRUD and
    # camera-stream WS endpoints. Keyed by fingerprint (not ProjectCamera ID)
    # so aliased project rows for the same physical device share one lock.
    app.state.recording_locked_camera_fingerprints = set()
    logger.info(f"Starting {settings.app_name} application...")
    ensure_spawn_start_method()
    app_scheduler = Scheduler()
    app_scheduler.start_workers()

    app.state.model_registry = ModelWorkerRegistry(
        max_workers=1,
        stop_event=app_scheduler.mp_stop_event,
    )
    app.state.scheduler = app_scheduler
    app.state.event_processor = EventProcessor(app_scheduler.event_queue)
    logger.info("Application startup completed")

    # Initialize RobotHardwareManager
    app.state.robot_manager = RobotConnectionManager()
    await app.state.robot_manager.find_robots()

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.app_name} application...")

    # We might want to shutdown the hardware manager too, though releasing workers should handle it.
    # But a global cleanup is safe.
    # Ideally RobotHardwareManager would have a shutdown_all method too.
    # For now, we assume active workers unregistering will trigger releases.

    app_scheduler.shutdown()
    app.state.event_processor.shutdown()
    logger.info("Application shutdown completed")
