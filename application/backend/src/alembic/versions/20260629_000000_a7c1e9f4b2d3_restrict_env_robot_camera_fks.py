"""restrict environment robot/camera FKs

Switch the robot/camera foreign keys on the environment join tables from ON DELETE CASCADE to
ON DELETE NO ACTION. A robot or camera that is used by an environment can no longer be deleted
silently: the delete is rejected (the service layer turns this into a 409). NO ACTION (rather than
RESTRICT) defers the check to end-of-statement, so deleting a whole project — which cascades both
environments and robots/cameras in a single statement — still succeeds.

The ``environment_id`` FKs keep ON DELETE CASCADE so deleting an environment still removes its links.

SQLite cannot ALTER a foreign key, so each join table is rebuilt: data is read into memory, the
table is dropped and recreated with the new constraints, then the rows are reinserted.

Revision ID: a7c1e9f4b2d3
Revises: f3a1c9d2b8e4
Create Date: 2026-06-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7c1e9f4b2d3"
down_revision: str | Sequence[str] | None = "f3a1c9d2b8e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_environment_robots(robot_ondelete: str) -> None:
    op.create_table(
        "environment_robots",
        sa.Column("environment_id", sa.Text(), nullable=False),
        sa.Column("robot_id", sa.Text(), nullable=False),
        sa.Column("tele_operator_type", sa.String(length=16), nullable=False),
        sa.Column("tele_operator_robot_id", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("environment_id", "robot_id"),
        sa.ForeignKeyConstraint(["environment_id"], ["project_environments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["robot_id"], ["project_robots.id"], ondelete=robot_ondelete),
        sa.ForeignKeyConstraint(["tele_operator_robot_id"], ["project_robots.id"], ondelete=robot_ondelete),
    )


def _create_environment_cameras(camera_ondelete: str) -> None:
    op.create_table(
        "environment_cameras",
        sa.Column("environment_id", sa.Text(), nullable=False),
        sa.Column("camera_id", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("environment_id", "camera_id"),
        sa.ForeignKeyConstraint(["environment_id"], ["project_environments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["camera_id"], ["project_cameras.id"], ondelete=camera_ondelete),
    )


def _rebuild(robot_ondelete: str, camera_ondelete: str) -> None:
    conn = op.get_bind()

    robots = conn.execute(
        sa.text("SELECT environment_id, robot_id, tele_operator_type, tele_operator_robot_id FROM environment_robots")
    ).fetchall()
    cameras = conn.execute(sa.text("SELECT environment_id, camera_id FROM environment_cameras")).fetchall()

    # Disable FK enforcement during the rebuild: dropping/recreating the child tables would otherwise
    # trip constraint checks mid-operation.
    conn.execute(sa.text("PRAGMA foreign_keys = OFF"))

    op.drop_table("environment_robots")
    op.drop_table("environment_cameras")
    _create_environment_robots(robot_ondelete)
    _create_environment_cameras(camera_ondelete)

    for environment_id, robot_id, tele_operator_type, tele_operator_robot_id in robots:
        conn.execute(
            sa.text(
                "INSERT INTO environment_robots "
                "(environment_id, robot_id, tele_operator_type, tele_operator_robot_id) "
                "VALUES (:environment_id, :robot_id, :tele_operator_type, :tele_operator_robot_id)"
            ),
            {
                "environment_id": environment_id,
                "robot_id": robot_id,
                "tele_operator_type": tele_operator_type,
                "tele_operator_robot_id": tele_operator_robot_id,
            },
        )

    for environment_id, camera_id in cameras:
        conn.execute(
            sa.text("INSERT INTO environment_cameras (environment_id, camera_id) VALUES (:environment_id, :camera_id)"),
            {"environment_id": environment_id, "camera_id": camera_id},
        )

    conn.execute(sa.text("PRAGMA foreign_keys = ON"))


def upgrade() -> None:
    """Rebuild the join tables with NO ACTION robot/camera foreign keys."""
    _rebuild(robot_ondelete="NO ACTION", camera_ondelete="NO ACTION")


def downgrade() -> None:
    """Restore the original CASCADE robot/camera foreign keys."""
    _rebuild(robot_ondelete="CASCADE", camera_ondelete="CASCADE")
