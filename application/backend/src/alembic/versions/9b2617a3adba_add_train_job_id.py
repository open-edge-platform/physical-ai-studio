"""add_train_job_id

Revision ID: 9b2617a3adba
Revises: aa0f562acb23
Create Date: 2026-03-02 20:05:01.013578

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b2617a3adba"
down_revision: str | Sequence[str] | None = "aa0f562acb23"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.add_column(sa.Column("train_job_id", sa.Text(), nullable=True))
        batch_op.create_foreign_key("fk_models_train_job_id", "jobs", ["train_job_id"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.drop_column("train_job_id")
