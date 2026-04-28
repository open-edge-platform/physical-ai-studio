"""add bimanual widowxai robot types

Revision ID: b9f3e2a1c7d8
Revises: a1b2c3d4e5f6
Create Date: 2026-04-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b9f3e2a1c7d8"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


PREVIOUS_TYPES = (
    "SO101_FOLLOWER",
    "SO101_LEADER",
    "TROSSEN_WIDOWXAI_LEADER",
    "TROSSEN_WIDOWXAI_FOLLOWER",
)


def upgrade() -> None:
    """Move project_robots.type to plain string semantics."""
    with op.batch_alter_table("project_robots") as batch_op:
        batch_op.alter_column(
            "type",
            type_=sa.String(length=255),
            existing_nullable=False,
        )


def downgrade() -> None:
    """Remove incompatible rows and restore constraint semantics."""
    conn = op.get_bind()

    conn.execute(
        sa.text(
            """
            DELETE FROM project_robots
            WHERE type NOT IN :types
            """
        ).bindparams(sa.bindparam("types", expanding=True)),
        {"types": PREVIOUS_TYPES},
    )

    with op.batch_alter_table("project_robots", schema=None) as batch_op:
        previous_enum = sa.Enum(*PREVIOUS_TYPES, name="robottype")

        batch_op.alter_column(
            "type",
            existing_type=sa.String(length=255),
            type_=previous_enum,
            existing_nullable=False,
        )
