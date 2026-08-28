import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from core.logging import setup_logging, setup_uvicorn_logging
from services.camera_claims import CameraClaimRegistry
from services.event_processor import EventProcessor
from services.health_service import HealthService
from settings import get_settings
from utils.multiprocessing import ensure_spawn_start_method
from utils.serial_robot_tools import RobotConnectionManager

from .scheduler import Scheduler


def _restart_argv_candidates() -> list[list[str]]:
    """Build candidate argv lists for re-executing the current server process."""
    candidates: list[list[str]] = []

    orig_argv = list(getattr(sys, "orig_argv", []) or [])
    if orig_argv and orig_argv[0]:
        candidates.append(orig_argv)

    python_argv = [sys.executable, *sys.argv]
    if python_argv[0] and python_argv not in candidates:
        candidates.append(python_argv)

    return candidates


def _restart_process() -> None:
    """Replace this process image after graceful application shutdown."""
    for argv in _restart_argv_candidates():
        executable = argv[0]
        try:
            if os.path.sep in executable:
                os.execv(executable, argv)  # noqa: S606
            else:
                os.execvp(executable, argv)  # noqa: S606
        except OSError:
            logger.exception("Restart exec failed for argv={}", argv)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    setup_logging()
    setup_uvicorn_logging()

    settings = get_settings()
    app.state.settings = settings
    app.state.health_service = HealthService()

    # Camera settings pinned by live runtime sessions in this API process.
    # In-memory on purpose: see CameraClaimRegistry. Keyed by fingerprint so
    # aliased project rows for the same physical device share one pin.
    app.state.camera_claim_registry = CameraClaimRegistry()
    logger.info(f"Starting {settings.app_name} application...")
    ensure_spawn_start_method()
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
    logger.info(f"Shutting down {settings.app_name} application...")

    # We might want to shutdown the hardware manager too, though releasing workers should handle it.
    # But a global cleanup is safe.
    # Ideally RobotHardwareManager would have a shutdown_all method too.
    # For now, we assume active workers unregistering will trigger releases.

    app_scheduler.shutdown()
    app.state.event_processor.shutdown()
    logger.info("Application shutdown completed")

    if app.state.health_service.plugin_restart_required:
        _restart_process()
