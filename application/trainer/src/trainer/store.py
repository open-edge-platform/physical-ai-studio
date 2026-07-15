# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed job queue for the trainer service.

A single table persists job state so the queue survives restarts. Access is
serialized with a lock; the store is small and write-light, so this is simpler
and safer than a connection pool.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from typing import TYPE_CHECKING, Any

from trainer.schemas import JobState, SubmitJobRequest, TrainerJobStatus

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    progress    INTEGER NOT NULL DEFAULT 0,
    message     TEXT,
    extra_info  TEXT,
    request     TEXT NOT NULL,
    artifact    TEXT,
    created_at  REAL NOT NULL
);
"""

# One fully static UPDATE statement per updatable column. `update()` picks
# from this fixed mapping by column name, so no SQL text is ever assembled
# from caller input -- only bound `?` parameter values change per call.
_UPDATE_STATEMENTS: dict[str, str] = {
    "status": "UPDATE jobs SET status = ? WHERE id = ?",
    "progress": "UPDATE jobs SET progress = ? WHERE id = ?",
    "message": "UPDATE jobs SET message = ? WHERE id = ?",
    "extra_info": "UPDATE jobs SET extra_info = ? WHERE id = ?",
    "artifact": "UPDATE jobs SET artifact = ? WHERE id = ?",
}


class JobStore:
    """Thread-safe persistence for trainer jobs."""

    def __init__(self, db_path: Path) -> None:
        """Open (creating if needed) the SQLite database at ``db_path``."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def create(self, request: SubmitJobRequest) -> str:
        """Persist a job and return its id; jobs await their dataset upload."""
        job_id = str(uuid.uuid4())
        status, message = TrainerJobStatus.AWAITING_DATASET, "Awaiting dataset upload"
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, status, progress, message, request, created_at) "
                "VALUES (?, ?, 0, ?, ?, julianday('now'))",
                (job_id, status, message, request.model_dump_json()),
            )
            self._conn.commit()
        return job_id

    def mark_dataset_ready(self, job_id: str) -> None:
        """Transition an awaiting-dataset job to queued after its upload lands."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, message = ? WHERE id = ? AND status = ?",
                (TrainerJobStatus.QUEUED, "Queued", job_id, TrainerJobStatus.AWAITING_DATASET),
            )
            self._conn.commit()

    def get(self, job_id: str) -> JobState | None:
        """Return the job's state, or None if it does not exist."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_state(row) if row else None

    def get_request(self, job_id: str) -> SubmitJobRequest | None:
        """Return the original submission request for a job."""
        with self._lock:
            row = self._conn.execute("SELECT request FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return SubmitJobRequest.model_validate_json(row["request"]) if row else None

    def get_artifact(self, job_id: str) -> str | None:
        """Return the model archive path once a job has completed."""
        with self._lock:
            row = self._conn.execute("SELECT artifact FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row["artifact"] if row and row["artifact"] else None

    def next_queued(self) -> str | None:
        """Return the oldest queued job id, if any."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at ASC, rowid ASC LIMIT 1",
                (TrainerJobStatus.QUEUED,),
            ).fetchone()
        return row["id"] if row else None

    def running_count(self) -> int:
        """Return the number of jobs currently running."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE status = ?",
                (TrainerJobStatus.RUNNING,),
            ).fetchone()
        return int(row["c"])

    def update(
        self,
        job_id: str,
        *,
        status: TrainerJobStatus | None = None,
        progress: int | None = None,
        message: str | None = None,
        extra_info: dict[str, Any] | None = None,
        artifact: str | None = None,
    ) -> None:
        """Apply a partial update to a job row."""
        values: dict[str, Any] = {}
        if status is not None:
            values["status"] = status
        if progress is not None:
            values["progress"] = max(0, min(100, progress))
        if message is not None:
            values["message"] = message
        if extra_info is not None:
            values["extra_info"] = json.dumps(extra_info)
        if artifact is not None:
            values["artifact"] = artifact
        if not values:
            return
        with self._lock:
            for column, value in values.items():
                statement = _UPDATE_STATEMENTS[column]
                self._conn.execute(statement, (value, job_id))
            self._conn.commit()

    def reset_orphans(self) -> None:
        """Fail running jobs and incomplete HTTP uploads after a restart."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, message = ? WHERE status = ?",
                (TrainerJobStatus.FAILED, "Aborted on trainer restart", TrainerJobStatus.RUNNING),
            )
            self._conn.execute(
                "UPDATE jobs SET status = ?, message = ? WHERE status = ?",
                (TrainerJobStatus.FAILED, "Dataset upload never completed", TrainerJobStatus.AWAITING_DATASET),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_state(row: sqlite3.Row) -> JobState:
        return JobState(
            remote_job_id=row["id"],
            status=TrainerJobStatus(row["status"]),
            progress=row["progress"],
            message=row["message"],
            extra_info=json.loads(row["extra_info"]) if row["extra_info"] else None,
        )
