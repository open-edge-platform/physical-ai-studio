# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

import anyio
from loguru import logger
from sse_starlette import ServerSentEvent

from core.logging.utils import get_job_logs_path
from schemas.base_job import JobType
from schemas.job import TrainJobPayload
from schemas.logs import LogSource
from services.job_service import JobService
from settings import Settings


@dataclass(frozen=True, slots=True)
class StaticLogSource:
    name: str
    filename: str
    type: Literal["application", "worker"]


STATIC_SOURCES: dict[str, StaticLogSource] = {
    "application": StaticLogSource(name="Application", filename="app.log", type="application"),
    "training": StaticLogSource(name="Training", filename="training.log", type="worker"),
    "inference": StaticLogSource(name="Inference", filename="inference.log", type="worker"),
    "teleoperate": StaticLogSource(name="Teleoperate", filename="teleoperate.log", type="worker"),
}


class LogService:
    def __init__(self, settings: Settings, job_service: JobService):
        self.settings = settings
        self.job_service = job_service

    def _get_static_log_path(self, source_id: str) -> Path | None:
        """Resolve a static source id to an absolute log file path."""
        source = STATIC_SOURCES.get(source_id)
        if source is None:
            return None
        return self.settings.log_dir / source.filename

    def resolve_source_path(self, source_id: str) -> Path | None:
        """Resolve any source id to an absolute log file path.

        Returns None if the source id is not recognized.
        """
        # Static source (application + workers)
        path = self._get_static_log_path(source_id)
        if path is not None:
            return path

        # Job source (format: "job-{type}-{uuid}")
        if source_id.startswith("job-"):
            try:
                job_id = source_id.removeprefix("job-")
                return Path(get_job_logs_path(job_id))
            except ValueError:
                return None

        return None

    def _get_file_created_at(self, path: Path) -> datetime | None:
        """Return the creation time of a file as a UTC datetime, or None on error."""
        try:
            stat = path.stat()
            # Use birth time if available (Linux 4.11+ with statx), fall back to mtime.
            ts = getattr(stat, "st_birthtime", None) or stat.st_mtime
            return datetime.fromtimestamp(ts, tz=UTC)
        except OSError:
            return None

    def _short_id(self, uuid_str: str) -> str:
        """Return the first 8 characters of a UUID string for display."""
        return uuid_str[:8]

    async def _get_job_source_name_map(self, job_ids: list[str]) -> dict[str, str]:
        """Return job source names for ids using a single DB round-trip."""
        names = {job_id: f"Job: {self._short_id(job_id)}" for job_id in job_ids}

        jobs = await self.job_service.get_jobs_by_ids([UUID(job_id) for job_id in job_ids])

        for job in jobs:
            if job.type == JobType.TRAINING:
                payload = TrainJobPayload.model_validate(job.payload)
                names[str(job.id)] = f"{payload.model_name} ({payload.policy})"

        return names

    async def _discover_job_sources(self) -> list[LogSource]:
        """List per-job log files on disk and return them as LogSource entries.

        Job log files are named ``{type}_{job_id}.log`` (e.g. ``training_abc123.log``).
        """
        jobs_dir = self.settings.log_dir / "jobs"
        sources: list[LogSource] = []
        if not jobs_dir.is_dir():
            return sources

        job_log_paths = sorted(jobs_dir.glob("*.log"))
        job_ids = [file_path.stem for file_path in job_log_paths]
        source_name_map = await self._get_job_source_name_map(job_ids)

        for file_path in job_log_paths:
            created_at = self._get_file_created_at(file_path)
            job_id = file_path.stem
            sources.append(
                LogSource(
                    id=f"job-{job_id}",
                    name=source_name_map[job_id],
                    type="job",
                    created_at=created_at,
                )
            )
        return sources

    async def get_log_sources(self) -> list[LogSource]:
        """Return all available log sources."""
        sources: list[LogSource] = []

        # Static sources (application + workers)
        for source_id, source in STATIC_SOURCES.items():
            sources.append(LogSource(id=source_id, name=source.name, type=source.type))

        # Dynamic: per-job logs
        sources.extend(await self._discover_job_sources())

        return sources

    async def source_exists(self, path: Path) -> bool:
        """Check whether a source path exists on disk and is non-empty."""
        source_path = anyio.Path(path)
        if not await source_path.exists():
            return False
        return (await source_path.stat()).st_size > 0

    async def tail_log_file(self, path: Path) -> AsyncGenerator[ServerSentEvent]:
        """Async generator that live-tails a log file and yields SSE events."""
        try:
            async with await anyio.open_file(path, encoding="utf-8") as f:
                while True:
                    line = await f.readline()
                    if not line:
                        await asyncio.sleep(0.5)
                        continue
                    yield ServerSentEvent(data=line.rstrip())
        except asyncio.CancelledError:
            logger.debug(f"SSE log stream cancelled for {path}")
        except GeneratorExit:
            logger.debug(f"SSE log stream closed for {path}")

    async def empty_log_stream(self) -> AsyncGenerator[ServerSentEvent]:
        """Yield a terminal SSE event for sources with no log file yet."""
        yield ServerSentEvent(data="DONE")
