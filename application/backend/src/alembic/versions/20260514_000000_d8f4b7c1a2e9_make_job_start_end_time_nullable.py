"""Make job start_time and end_time nullable.

Revision ID: d8f4b7c1a2e9
Revises: b9f3e2a1c7d8
Create Date: 2026-05-14 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d8f4b7c1a2e9"
down_revision: str | Sequence[str] | None = "b9f3e2a1c7d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.alter_column("start_time", existing_type=sa.DateTime(), nullable=True, server_default=None)
        batch_op.alter_column("end_time", existing_type=sa.DateTime(), nullable=True, server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.alter_column(
            "end_time",
            existing_type=sa.DateTime(),
            nullable=False,
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
        )
        batch_op.alter_column(
            "start_time",
            existing_type=sa.DateTime(),
            nullable=False,
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
        )
