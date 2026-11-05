import asyncio
import csv
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import anyio
import httpx
from loguru import logger
from sse_starlette import ServerSentEvent

from exceptions import InvalidResourceError, ResourceType
from schemas.job import Job, JobType
from schemas.model import Model
from settings import Settings


class ModelMetricsService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @classmethod
    async def get_model_metrics_path(cls, model: Model) -> Path:
        return Path(model.path) / "version_0" / "metrics.csv"

    async def get_model_job_metrics_path(self, job: Job) -> Path:
        if job.type != JobType.TRAINING:
            raise InvalidResourceError(ResourceType.JOB, f"{str(job.id)} {job.type} is not of type {JobType.TRAINING}")
        return self.settings.cache_dir / str(job.id) / "version_0" / "metrics.csv"

    @classmethod
    async def tail_csv_file(cls, path: Path) -> AsyncGenerator[ServerSentEvent]:
        """Async generator that live-tails a log file and yields SSE events."""
        try:
            async with await anyio.open_file(path, encoding="utf-8") as f:
                header_line = await f.readline()
                headers = next(csv.reader([header_line]))
                yield ServerSentEvent(event="issues", data="[]")
                headers = [header.replace("/", "_") for header in headers]

                while True:
                    line = await f.readline()
                    if not line:
                        await asyncio.sleep(0.5)
                        continue

                    row = next(csv.reader([line]))
                    data = {key: cls._parse_value(value) for key, value in zip(headers, row)}
                    yield ServerSentEvent(data=json.dumps(data))
        except asyncio.CancelledError:
            logger.debug(f"SSE log stream cancelled for {path}")
        except GeneratorExit:
            logger.debug(f"SSE log stream closed for {path}")

    @staticmethod
    async def tail_remote_csv_file(base_url: str, remote_job_id: str) -> AsyncGenerator[ServerSentEvent]:
        """Relay metric SSE events from a remote trainer without exposing it to clients."""
        url = f"{base_url.rstrip('/')}/jobs/{remote_job_id}/metrics"
        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream("GET", url, headers={"Accept": "text/event-stream"}) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        yield ServerSentEvent(data=line.removeprefix("data:").lstrip())
        except httpx.HTTPError as exc:
            logger.warning("Remote metrics stream failed for {}: {}", remote_job_id, exc)
        except asyncio.CancelledError:
            logger.debug("Remote metrics stream cancelled for {}", remote_job_id)
        except GeneratorExit:
            logger.debug("Remote metrics stream closed for {}", remote_job_id)

    @staticmethod
    async def tail_batch_csv_file(trainer_url: str, job_id: str) -> AsyncGenerator[ServerSentEvent]:
        """Stream successive metric snapshots published by a Batch trainer to S3."""
        from botocore.exceptions import BotoCoreError, ClientError

        from services.aws_auth_service import authentication_issue, operational_authentication_issue
        from services.aws_batch import _make_session, aws_batch_connection_from_url

        connection = aws_batch_connection_from_url(trainer_url)
        client, session_created, rows_sent, needs_issue_snapshot = None, 0.0, 0, True
        try:
            while True:
                try:
                    if client is None or time.monotonic() - session_created > 600:
                        session = await asyncio.to_thread(_make_session, connection)
                        client = session.client("s3")
                        session_created = time.monotonic()
                    response = await asyncio.to_thread(
                        client.get_object, Bucket=connection.bucket, Key=f"jobs/{job_id}/metrics.csv"
                    )
                    body = response["Body"]
                    try:
                        contents = (await asyncio.to_thread(body.read)).decode("utf-8")
                    finally:
                        body.close()
                except (BotoCoreError, ClientError) as error:
                    if issue := authentication_issue(error, connection.profile_name):
                        client = None
                        needs_issue_snapshot = True
                        yield ServerSentEvent(
                            event="issues", data=json.dumps([operational_authentication_issue(issue).model_dump()])
                        )
                        await asyncio.sleep(5)
                        continue
                    if not isinstance(error, ClientError) or error.response["Error"]["Code"] not in {
                        "NoSuchKey",
                        "404",
                    }:
                        raise
                    contents = ""
                lines = contents.splitlines(keepends=True)
                if lines and not lines[-1].endswith("\n"):
                    lines.pop()
                rows = list(csv.DictReader(lines))
                if needs_issue_snapshot:
                    yield ServerSentEvent(event="issues", data="[]")
                    needs_issue_snapshot = False
                for row in rows[rows_sent:]:
                    data = {
                        key.replace("/", "_"): ModelMetricsService._parse_value(value) for key, value in row.items()
                    }
                    yield ServerSentEvent(data=json.dumps(data))
                rows_sent = max(rows_sent, len(rows))
                await asyncio.sleep(5)
        except (BotoCoreError, ClientError) as error:
            logger.warning("Batch metrics stream failed for {}: {}", job_id, error)
        except asyncio.CancelledError:
            logger.debug("Batch metrics stream cancelled for {}", job_id)
        except GeneratorExit:
            logger.debug("Batch metrics stream closed for {}", job_id)

    @staticmethod
    async def empty_metrics_stream() -> AsyncGenerator[ServerSentEvent]:
        """Yield a terminal SSE event for sources with no log file yet."""
        yield ServerSentEvent(event="issues", data="[]")
        yield ServerSentEvent(data="DONE")

    @staticmethod
    def _parse_value(s: str) -> int | float | str | None:
        try:
            f = float(s)
            return int(f) if f.is_integer() else f
        except ValueError:
            if s == "":
                return None
            return s
