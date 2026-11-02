"""environment robot/camera join tables

Move project_environments.robots / camera_ids JSON columns into dedicated join tables
(environment_robots, environment_cameras) that carry a per-environment name.

Revision ID: e1f2a3b4c5d6
Revises: d8f4b7c1a2e9
Create Date: 2026-06-08 00:00:00.000000

"""

import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1f2a3b4c5d6"
down_revision: str | Sequence[str] | None = "d8f4b7c1a2e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _load_json(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            # Handle double-encoded JSON (stored as a JSON string on main)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return []
    return value if isinstance(value, list) else []


def upgrade() -> None:
    """Migrate existing JSON data into dedicated join tables.

    The JSON columns are read into memory and dropped *before* the join tables are created: the
    SQLite ``batch_alter_table`` rebuild of ``project_environments`` would otherwise cascade-delete
    (ON DELETE CASCADE) any join rows already inserted.
    """
    conn = op.get_bind()

    # Read existing JSON data before touching any schema.
    environments = conn.execute(sa.text("SELECT id, robots, camera_ids FROM project_environments")).fetchall()

    conn.execute(sa.text("PRAGMA foreign_keys = OFF"))
    with op.batch_alter_table("project_environments", schema=None) as batch_op:
        batch_op.drop_column("robots")
        batch_op.drop_column("camera_ids")
    conn.execute(sa.text("PRAGMA foreign_keys = ON"))

    op.create_table(
        "environment_robots",
        sa.Column("environment_id", sa.Text(), nullable=False),
        sa.Column("robot_id", sa.Text(), nullable=False),
        sa.Column("tele_operator_type", sa.String(length=16), nullable=False),
        sa.Column("tele_operator_robot_id", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("environment_id", "robot_id"),
        sa.ForeignKeyConstraint(["environment_id"], ["project_environments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["robot_id"], ["project_robots.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tele_operator_robot_id"], ["project_robots.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "environment_cameras",
        sa.Column("environment_id", sa.Text(), nullable=False),
        sa.Column("camera_id", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("environment_id", "camera_id"),
        sa.ForeignKeyConstraint(["environment_id"], ["project_environments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["camera_id"], ["project_cameras.id"], ondelete="CASCADE"),
    )

    for raw_env_id, robots_raw, camera_ids_raw in environments:
        env_id = str(raw_env_id)

        for robot_cfg in _load_json(robots_raw):
            robot_id = str(robot_cfg.get("robot_id"))
            tele = robot_cfg.get("tele_operator") or {"type": "none"}
            tele_type = tele.get("type", "none")
            tele_robot_id = str(tele["robot_id"]) if tele_type == "robot" and tele.get("robot_id") else None
            conn.execute(
                sa.text(
                    "INSERT INTO environment_robots "
                    "(environment_id, robot_id, tele_operator_type, tele_operator_robot_id) "
                    "VALUES (:env_id, :robot_id, :tele_type, :tele_robot_id)"
                ),
                {
                    "env_id": env_id,
                    "robot_id": robot_id,
                    "tele_type": tele_type,
                    "tele_robot_id": tele_robot_id,
                },
            )

        for raw_camera_id in _load_json(camera_ids_raw):
            camera_id = str(raw_camera_id)
            conn.execute(
                sa.text("INSERT INTO environment_cameras (environment_id, camera_id) VALUES (:env_id, :camera_id)"),
                {
                    "env_id": env_id,
                    "camera_id": camera_id,
                },
            )


def downgrade() -> None:
    """Restore the JSON columns from the join tables, then drop them.

    Join rows are read into memory before re-adding the columns, since the SQLite
    ``batch_alter_table`` rebuild of ``project_environments`` cascade-deletes them.
    """
    conn = op.get_bind()

    robots_by_env: dict[str, list] = {}
    for env_id, robot_id, tele_type, tele_robot_id in conn.execute(
        sa.text("SELECT environment_id, robot_id, tele_operator_type, tele_operator_robot_id FROM environment_robots")
    ).fetchall():
        robots_by_env.setdefault(str(env_id), []).append(
            {
                "robot_id": str(robot_id),
                "tele_operator": (
                    {"type": "robot", "robot_id": str(tele_robot_id)}
                    if tele_type == "robot" and tele_robot_id is not None
                    else {"type": "none"}
                ),
            }
        )

    cameras_by_env: dict[str, list] = {}
    for env_id, camera_id in conn.execute(
        sa.text("SELECT environment_id, camera_id FROM environment_cameras")
    ).fetchall():
        cameras_by_env.setdefault(str(env_id), []).append(str(camera_id))

    # Drop join tables before batch_alter_table: SQLite's table rebuild internally
    # DROPs project_environments, which would fail the FK constraints referencing it.
    op.drop_table("environment_cameras")
    op.drop_table("environment_robots")

    conn.execute(sa.text("PRAGMA foreign_keys = OFF"))
    with op.batch_alter_table("project_environments", schema=None) as batch_op:
        batch_op.add_column(sa.Column("robots", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("camera_ids", sa.JSON(), nullable=True))
    conn.execute(sa.text("PRAGMA foreign_keys = ON"))

    for env_id in {*robots_by_env, *cameras_by_env}:
        conn.execute(
            sa.text("UPDATE project_environments SET robots = :robots, camera_ids = :camera_ids WHERE id = :env_id"),
            {
                "robots": json.dumps(json.dumps(robots_by_env.get(env_id, []))),
                "camera_ids": json.dumps(json.dumps(cameras_by_env.get(env_id, []))),
                "env_id": env_id,
            },
        )
