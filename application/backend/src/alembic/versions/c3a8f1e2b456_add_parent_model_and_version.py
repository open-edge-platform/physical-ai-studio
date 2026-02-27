"""Add parent_model_id and version to models

Revision ID: c3a8f1e2b456
Revises: aa0f562acb23
Create Date: 2026-02-27 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3a8f1e2b456"
down_revision: str | Sequence[str] | None = "aa0f562acb23"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.add_column(sa.Column("parent_model_id", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("version", sa.Integer(), nullable=False, server_default="1"))
        batch_op.create_foreign_key(
            "fk_models_parent_model_id",
            "models",
            ["parent_model_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.drop_constraint("fk_models_parent_model_id", type_="foreignkey")
        batch_op.drop_column("version")
        batch_op.drop_column("parent_model_id")
