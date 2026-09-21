"""Add connection modes and manual SSH fields to remote trainers.

Revision ID: 4b8d2f6a1c30
Revises: c9d1e2f3a4b5
Create Date: 2026-09-14 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "4b8d2f6a1c30"
down_revision: str | Sequence[str] | None = "c9d1e2f3a4b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Store the connection mode and non-secret manual SSH fields."""
    with op.batch_alter_table("remote_trainers") as batch_op:
        batch_op.add_column(sa.Column("connection_mode", sa.String(length=16), nullable=False, server_default="direct"))
        batch_op.add_column(sa.Column("ssh_hostname", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("ssh_port", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("ssh_username", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("ssh_identity_file", sa.String(length=4096), nullable=True))

    remote_trainers = sa.table(
        "remote_trainers",
        sa.column("connection_mode", sa.String()),
        sa.column("ssh_host_alias", sa.String()),
    )
    op.execute(
        remote_trainers.update().where(remote_trainers.c.ssh_host_alias.is_not(None)).values(connection_mode="ssh")
    )


def downgrade() -> None:
    """Drop connection modes and manually configured SSH fields."""
    with op.batch_alter_table("remote_trainers") as batch_op:
        batch_op.drop_column("ssh_identity_file")
        batch_op.drop_column("ssh_username")
        batch_op.drop_column("ssh_port")
        batch_op.drop_column("ssh_hostname")
        batch_op.drop_column("connection_mode")
