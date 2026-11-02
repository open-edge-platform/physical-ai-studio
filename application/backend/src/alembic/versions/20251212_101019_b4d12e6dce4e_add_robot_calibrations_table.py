"""add_robot_calibrations_table

Revision ID: b4d12e6dce4e
Revises: 4a70ddb2d384
Create Date: 2025-12-12 10:10:19.680472

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b4d12e6dce4e"
down_revision: str | Sequence[str] | None = "4a70ddb2d384"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "robot_calibrations",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("file_path", sa.String(length=255), nullable=False),
        sa.Column("robot_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["robot_id"], ["project_robots.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "calibration_values",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("joint_name", sa.String(length=255), nullable=False),
        sa.Column("calibration_id", sa.Text(), nullable=False),
        sa.Column("drive_mode", sa.Integer(), nullable=False),
        sa.Column("homing_offset", sa.Integer(), nullable=False),
        sa.Column("range_min", sa.Integer(), nullable=False),
        sa.Column("range_max", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", "calibration_id"),
        sa.ForeignKeyConstraint(["calibration_id"], ["robot_calibrations.id"], ondelete="CASCADE"),
    )

    op.add_column(
        "project_robots",
        sa.Column("active_calibration_id", sa.Text(), nullable=True),
    )

    with op.batch_alter_table("project_robots", schema=None) as batch_op:
        batch_op.create_foreign_key(
            "fk_project_robots_calibration_id",
            "robot_calibrations",
            ["active_calibration_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("project_robots", schema=None) as batch_op:
        batch_op.drop_constraint("fk_project_robots_calibration_id", type_="foreignkey")

    op.drop_table("calibration_values")
    op.drop_table("robot_calibrations")
