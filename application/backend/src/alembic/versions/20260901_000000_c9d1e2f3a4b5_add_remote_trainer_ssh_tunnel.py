"""Add optional SSH tunnel config to remote trainers.

Revision ID: c9d1e2f3a4b5
Revises: b7a4c1e9f0d2
Create Date: 2026-09-01 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9d1e2f3a4b5"
down_revision: str | Sequence[str] | None = "a1c2d3e4f5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add columns for an optional standing SSH port-forward tunnel per direct trainer."""
    with op.batch_alter_table("remote_trainers") as batch_op:
        batch_op.add_column(sa.Column("ssh_host_alias", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("ssh_remote_port", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("ssh_local_port", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Drop the SSH tunnel columns."""
    with op.batch_alter_table("remote_trainers") as batch_op:
        batch_op.drop_column("ssh_local_port")
        batch_op.drop_column("ssh_remote_port")
        batch_op.drop_column("ssh_host_alias")
