"""Make model dataset_id and snapshot_id nullable

Revision ID: c3e4e04cc003
Revises: d1a2b3c4d5e6
Create Date: 2026-02-27 16:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3e4e04cc003"
down_revision: str | Sequence[str] | None = "c3a8f1e2b456"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Make dataset_id and snapshot_id nullable on the models table.

    This is needed to support imported models that do not have a local
    dataset or snapshot.
    """
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.alter_column("dataset_id", existing_type=sa.Text(), nullable=True)
        batch_op.alter_column("snapshot_id", existing_type=sa.Text(), nullable=True)


def downgrade() -> None:
    """Restore NOT NULL constraints on dataset_id and snapshot_id."""
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.alter_column("dataset_id", existing_type=sa.Text(), nullable=False)
        batch_op.alter_column("snapshot_id", existing_type=sa.Text(), nullable=False)
