# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Entry point for one multi-device (DDP) training job, launched under ``torchrun``.

The trainer service runs ordinary jobs in a thread or a spawned child process
(see :mod:`trainer.queue_worker`). A DDP job cannot run there: Lightning's
default DDP launcher re-executes ``sys.argv`` once per extra device, which in
the service is the API server, and ``ddp_spawn`` would have to pickle the
progress and cancel callbacks. Under ``torchrun`` every rank is already a
process of its own, so Lightning launches nothing and each rank simply trains.

Every rank loads the same :class:`~trainer.schemas.SubmitJobRequest` and trains.
Global rank 0 owns the job's outputs (see :meth:`TrainerRunner.run`) and writes
the job's events, one JSON object per line, to the events file:

- ``{"type": "progress", "progress": int, "message": str | None, "extra_info": dict | None}``
- ``{"type": "completed", "archive": str}``: the model archive is ready.
- ``{"type": "failed", "error": str}``: rank 0 hit an exception; the process exits non-zero.

Other ranks write nothing. Cancellation is not cooperative here: the launching
process stops the whole process group, because ranks that stop on different
steps would hang in the next collective call.

The Hugging Face token is read from ``HF_TOKEN`` in the environment, never
from the request file.

Example:
    >>> build_command("job-1", Path("request.json"), Path("events.jsonl"), nproc=2)[1:5]
    ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import SecretStr

from trainer.log_setup import setup_logging
from trainer.runner import TrainerRunner
from trainer.schemas import SubmitJobRequest


class EventLog:
    """Append-only JSON-lines sink for a job's progress, written by rank 0."""

    def __init__(self, path: Path) -> None:
        """Bind the log to ``path``; the file is created on the first event."""
        self._path = path

    def _write(self, event: dict[str, Any]) -> None:
        # One short write per line, so a reader polling the file never sees a partial line.
        with self._path.open("a", encoding="utf-8") as events:
            events.write(json.dumps(event, default=str) + "\n")

    def progress(self, progress: int, message: str | None, extra_info: dict[str, Any] | None) -> None:
        """Record a progress update; matches :data:`trainer.runner.ProgressFn`."""
        self._write({"type": "progress", "progress": progress, "message": message, "extra_info": extra_info})

    def completed(self, archive: Path) -> None:
        """Record that the job finished and its model archive is at ``archive``."""
        self._write({"type": "completed", "archive": str(archive)})

    def failed(self, error: str) -> None:
        """Record that the job failed with ``error``."""
        self._write({"type": "failed", "error": error})


class EventReader:
    """Incrementally read the events an :class:`EventLog` appended, from the launching process."""

    def __init__(self, path: Path) -> None:
        """Start reading ``path`` from its beginning; the file may not exist yet."""
        self._path = path
        self._offset = 0

    def read_new(self) -> list[dict[str, Any]]:
        """Return the events appended since the last call.

        Only complete lines are consumed: a line still being written is left
        for the next call. A line that is not valid JSON is logged and skipped.
        """
        try:
            with self._path.open("rb") as events:
                events.seek(self._offset)
                chunk = events.read()
        except FileNotFoundError:
            return []
        complete, _, _partial = chunk.rpartition(b"\n")
        if not complete and not chunk.endswith(b"\n"):
            return []
        self._offset += len(complete) + 1
        parsed = []
        for line in complete.splitlines():
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed DDP event line: {!r}", line[:200])
        return parsed


def build_command(job_id: str, request_path: Path, events_path: Path, *, nproc: int) -> list[str]:
    """Return the ``torchrun`` command line that runs one DDP job on ``nproc`` local devices.

    Args:
        job_id: The trainer job id; selects the dataset, model, and cache paths.
        request_path: JSON file holding the job's ``SubmitJobRequest``.
        events_path: JSON-lines file rank 0 appends the job's events to.
        nproc: Number of processes, one per device.

    Returns:
        The argv to execute, using the current Python interpreter.
    """
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "-m",
        "trainer.ddp_entry",
        "--job-id",
        job_id,
        "--request",
        str(request_path),
        "--events",
        str(events_path),
    ]


def load_request(path: Path) -> SubmitJobRequest:
    """Load a job request, taking the Hugging Face token from the environment.

    The token is excluded from serialized requests, but a request dumped with
    its ``run_options`` set would carry a masked ``**********`` value; it is
    always replaced so a placeholder can never reach training.
    """
    request = SubmitJobRequest.model_validate_json(path.read_text(encoding="utf-8"))
    token = os.environ.get("HF_TOKEN") or None
    request.spec.run_options.hf_token = SecretStr(token) if token else None
    return request


def run(job_id: str, request: SubmitJobRequest, events: EventLog, *, rank: int) -> None:
    """Run this rank's share of a DDP job.

    Raises:
        Exception: Whatever training raised. On rank 0 a ``failed`` event is
            written first.
    """
    runner = TrainerRunner()
    if rank != 0:
        runner.train_worker_rank(job_id, request)
        return
    try:
        archive = runner.run(job_id, request, should_stop=lambda: False, report=events.progress)
    except Exception as exc:
        events.failed(str(exc) or type(exc).__name__)
        raise
    events.completed(archive)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run this rank's share of the job."""
    parser = argparse.ArgumentParser(description="Run one multi-device training job (launched by torchrun).")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args(argv)

    rank = int(os.environ.get("RANK", "0"))
    setup_logging()
    logger.info("DDP job {} rank {}/{} starting", args.job_id, rank, os.environ.get("WORLD_SIZE", "1"))
    run(args.job_id, load_request(args.request), EventLog(args.events), rank=rank)
    logger.info("DDP job {} rank {} finished", args.job_id, rank)


if __name__ == "__main__":
    main()
