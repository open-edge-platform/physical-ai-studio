"""backfill training_target for legacy job payloads

Jobs persisted before the ``training_target`` discriminator existed (and
before SSH-provisioned training was added) fail to load with a
``union_tag_not_found`` validation error, because the payload dict has no
``training_target`` key at all. This migration backfills the legacy target
directly in the database: ``remote`` if the payload carries a
``remote_trainer_id``, otherwise ``local``. Once resolved to ``local``, stale
remote-only fields (``remote_trainer_id``, ``remote_trainer_url``,
``remote_trainer_name``) are dropped too, since ``LocalTrainJobPayload``
forbids them as extras.

Revision ID: 1510153d39a2
Revises: b7a4c1e9f0d2
Create Date: 2026-09-08 00:00:00.000000
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "1510153d39a2"
down_revision: str | Sequence[str] | None = "b7a4c1e9f0d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LOCAL_ONLY_LEGACY_KEYS = ("remote_trainer_id", "remote_trainer_url", "remote_trainer_name")


def upgrade() -> None:
    """Backfill `training_target` on legacy training job payloads."""
    conn = op.get_bind()

    jobs = conn.execute(sa.text("SELECT id, payload FROM jobs WHERE type = 'training'")).fetchall()

    for job_id, payload_raw in jobs:
        if not isinstance(payload_raw, str):
            continue
        payload = json.loads(payload_raw)
        changed = False

        if "training_target" not in payload:
            payload["training_target"] = "remote" if payload.get("remote_trainer_id") else "local"
            changed = True

        if payload["training_target"] == "local":
            for key in _LOCAL_ONLY_LEGACY_KEYS:
                if key in payload:
                    del payload[key]
                    changed = True

        if changed:
            conn.execute(
                sa.text("UPDATE jobs SET payload = :payload WHERE id = :id"),
                {"payload": json.dumps(payload), "id": job_id},
            )


def downgrade() -> None:
    """Best-effort: remove `training_target` and stale remote-only keys are not restorable.

    This cannot perfectly undo the upgrade: jobs that already had
    `training_target` set before this migration ran are indistinguishable
    from ones backfilled by it, and dropped remote-only keys aren't
    recoverable. Downgrading only removes the discriminator key so a
    re-upgrade backfills it identically.
    """
    conn = op.get_bind()

    jobs = conn.execute(sa.text("SELECT id, payload FROM jobs WHERE type = 'training'")).fetchall()

    for job_id, payload_raw in jobs:
        if not isinstance(payload_raw, str):
            continue
        payload = json.loads(payload_raw)
        if payload.pop("training_target", None) is None:
            continue
        conn.execute(
            sa.text("UPDATE jobs SET payload = :payload WHERE id = :id"),
            {"payload": json.dumps(payload), "id": job_id},
        )
