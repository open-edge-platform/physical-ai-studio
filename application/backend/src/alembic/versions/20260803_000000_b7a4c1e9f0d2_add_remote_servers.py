"""add remote servers and per-job provisioning state

Creates ``remote_servers`` (SSH-provisioned training targets, identified only by
an SSH config alias) and ``job_provisioning`` (per-job container state).

``remote_servers`` deliberately has no ``username``, ``auth_type``,
``ssh_secret_encrypted``, ``ssh_key_passphrase_encrypted``, or ``host_key``
column, and no host/port/username unique constraint: Studio never receives SSH
credentials, and hostname/port/user are resolved from the user's SSH config at
read time so a stored row cannot silently disagree with it.

Revision ID: b7a4c1e9f0d2
Revises: e4b2f1c8a907
Create Date: 2026-08-03 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7a4c1e9f0d2"
down_revision: str | Sequence[str] | None = "e4b2f1c8a907"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the remote server registry and its per-job provisioning table."""
    op.create_table(
        "remote_servers",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("ssh_host_alias", sa.String(length=255), nullable=False),
        sa.Column("device_type", sa.String(), nullable=False),
        sa.Column("last_check_status", sa.String(length=32), nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("last_check_at", sa.DateTime(), nullable=True),
        sa.Column("last_check_latency_ms", sa.Integer(), nullable=True),
        sa.Column("last_check_reason_code", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ssh_host_alias", name="uq_remote_servers_ssh_host_alias"),
    )
    op.create_table(
        "job_provisioning",
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("remote_server_id", sa.Text(), nullable=False),
        sa.Column("ssh_host_alias", sa.String(length=255), nullable=False),
        sa.Column("image_ref", sa.String(length=512), nullable=True),
        sa.Column("image_fallback_reason", sa.String(length=512), nullable=True),
        sa.Column("image_digest", sa.String(length=255), nullable=True),
        sa.Column("container_id", sa.String(length=128), nullable=True),
        sa.Column("container_name", sa.String(length=255), nullable=True),
        sa.Column("remote_port", sa.Integer(), nullable=True),
        sa.Column("local_tunnel_port", sa.Integer(), nullable=True),
        sa.Column("backend_instance_id", sa.String(length=255), nullable=True),
        sa.Column("trainer_build_version", sa.String(length=255), nullable=True),
        sa.Column("trainer_protocol_version", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        # RESTRICT: deleting a server whose job is still running would drop the
        # only record of the container that has to be cleaned up.
        sa.ForeignKeyConstraint(["remote_server_id"], ["remote_servers.id"], ondelete="RESTRICT"),
    )


def downgrade() -> None:
    """Drop both tables, children first."""
    op.drop_table("job_provisioning")
    op.drop_table("remote_servers")
