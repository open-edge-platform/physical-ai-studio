"""robot payload json column

Revision ID: a1b2c3d4e5f6
Revises: 96ebe046cb09
Create Date: 2026-04-22 00:00:00.000000

"""

import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "96ebe046cb09"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: replace connection_string + serial_number with payload JSON."""
    conn = op.get_bind()

    # Step 1: add nullable payload column
    with op.batch_alter_table("project_robots", schema=None) as batch_op:
        batch_op.add_column(sa.Column("payload", sa.JSON(), nullable=True))

    # Step 2: backfill payload from existing columns
    rows = conn.execute(sa.text("SELECT id, type, connection_string, serial_number FROM project_robots")).fetchall()

    for row in rows:
        robot_id, robot_type, connection_string, serial_number = row

        payload = (
            {
                "serial_number": serial_number or "",
            }
            if robot_type in ("SO101_Follower", "SO101_LEADER")
            else {
                "connection_string": connection_string or "",
            }
        )

        conn.execute(
            sa.text("UPDATE project_robots SET payload = :payload WHERE id = :id"),
            {"payload": json.dumps(payload), "id": str(robot_id)},
        )

    # Step 3: set payload NOT NULL and drop old columns
    with op.batch_alter_table("project_robots", schema=None) as batch_op:
        batch_op.alter_column("payload", nullable=False)
        batch_op.drop_column("connection_string")
        batch_op.drop_column("serial_number")


def downgrade() -> None:
    """Downgrade schema: restore connection_string + serial_number from payload JSON."""
    conn = op.get_bind()

    # Step 1: re-add old columns as nullable
    with op.batch_alter_table("project_robots", schema=None) as batch_op:
        batch_op.add_column(sa.Column("connection_string", sa.String(255), nullable=True))
        batch_op.add_column(sa.Column("serial_number", sa.String(255), nullable=True))

    # Step 2: repopulate from payload
    rows = conn.execute(sa.text("SELECT id, payload FROM project_robots")).fetchall()

    for row in rows:
        robot_id, payload_raw = row
        try:
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
        except (ValueError, TypeError):
            payload = {}

        conn.execute(
            sa.text("UPDATE project_robots SET connection_string = :cs, serial_number = :sn WHERE id = :id"),
            {
                "cs": payload.get("connection_string", ""),
                "sn": payload.get("serial_number", ""),
                "id": str(robot_id),
            },
        )

    # Step 3: drop payload column
    with op.batch_alter_table("project_robots", schema=None) as batch_op:
        batch_op.drop_column("payload")
