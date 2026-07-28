"""backfill SO101 calibration into robot payload

Existing SO101 robots store their calibration in robot_calibrations +
calibration_values tables, linked via project_robots.active_calibration_id.
This migration reads those values and embeds them as a ``calibration`` key
in the robot's JSON payload column so the active_calibration_id FK can
eventually be dropped.

Revision ID: c9d8e7f6a5b4
Revises: a7c1e9f4b2d3
Create Date: 2026-07-20 00:00:00.000000
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c9d8e7f6a5b4"
down_revision: str | Sequence[str] | None = "a7c1e9f4b2d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_ROBOT_TYPES = ("SO101_Follower", "SO101_Leader", "SO101_FOLLOWER", "SO101_LEADER")


def upgrade() -> None:
    """Backfill calibration data from old tables into robot payload JSON."""
    conn = op.get_bind()

    robots = conn.execute(
        sa.text(
            "SELECT id, payload, active_calibration_id "
            "FROM project_robots "
            "WHERE type IN :types AND active_calibration_id IS NOT NULL"
        ).bindparams(sa.bindparam("types", expanding=True)),
        {"types": _ROBOT_TYPES},
    ).fetchall()

    for robot_id, payload_raw, cal_id in robots:
        if not isinstance(payload_raw, str):
            continue
        payload = json.loads(payload_raw)
        if "calibration" in payload:
            continue

        rows = conn.execute(
            sa.text(
                "SELECT joint_name, id, drive_mode, homing_offset, range_min, range_max "
                "FROM calibration_values WHERE calibration_id = :cal_id"
            ),
            {"cal_id": cal_id},
        ).fetchall()

        calibration = {
            row[0]: {
                "id": row[1],
                "drive_mode": row[2],
                "homing_offset": row[3],
                "range_min": row[4],
                "range_max": row[5],
            }
            for row in rows
        }

        payload["calibration"] = calibration
        conn.execute(
            sa.text("UPDATE project_robots SET payload = :payload WHERE id = :id"),
            {"payload": json.dumps(payload), "id": robot_id},
        )


def downgrade() -> None:
    """Remove calibration key from robot payloads."""
    conn = op.get_bind()

    robots = conn.execute(
        sa.text("SELECT id, payload FROM project_robots WHERE type IN :types").bindparams(
            sa.bindparam("types", expanding=True)
        ),
        {"types": _ROBOT_TYPES},
    ).fetchall()

    for robot_id, payload_raw in robots:
        if not isinstance(payload_raw, str):
            continue
        payload = json.loads(payload_raw)
        payload.pop("calibration", None)
        conn.execute(
            sa.text("UPDATE project_robots SET payload = :payload WHERE id = :id"),
            {"payload": json.dumps(payload), "id": robot_id},
        )
