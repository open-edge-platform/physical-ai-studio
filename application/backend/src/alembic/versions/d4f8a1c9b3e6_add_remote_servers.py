"""Add remote servers and per-job SSH provisioning state.

Revision ID: d4f8a1c9b3e6
Revises: e4b2f1c8a907
Create Date: 2026-07-24 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4f8a1c9b3e6"
down_revision: str | Sequence[str] | None = "e4b2f1c8a907"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create storage for registered SSH remote servers and per-job provisioning state."""
    op.create_table(
        "remote_servers",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("host", sa.String(length=255), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=255), nullable=False),
        sa.Column("auth_type", sa.Enum("KEY", "PASSWORD", name="sshauthtype"), nullable=False),
        sa.Column("device_type", sa.Enum("CPU", "XPU", "CUDA", "NPU", name="devicetype"), nullable=False),
        # Fernet ciphertext only; decrypted solely inside the SSH provisioning
        # boundary (added in a later PR). Never returned by the API.
        sa.Column("ssh_secret_encrypted", sa.Text(), nullable=False),
        sa.Column("ssh_key_passphrase_encrypted", sa.Text(), nullable=True),
        # Pinned public host key (TOFU). Integrity data, not a secret: stored
        # in plaintext, but never returned by the API.
        sa.Column("host_key", sa.Text(), nullable=True),
        sa.Column("last_check_status", sa.String(length=32), nullable=True),
        sa.Column("last_check_at", sa.DateTime(), nullable=True),
        sa.Column("last_check_latency_ms", sa.Integer(), nullable=True),
        sa.Column("last_check_reason_code", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("host", "port", "username", name="uq_remote_servers_host_port_username"),
    )

    op.create_table(
        "job_provisioning",
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("remote_server_id", sa.Text(), nullable=False),
        sa.Column("image_ref", sa.String(length=512), nullable=True),
        sa.Column("image_fallback_reason", sa.String(length=255), nullable=True),
        sa.Column("image_digest", sa.String(length=255), nullable=True),
        sa.Column("container_id", sa.String(length=255), nullable=True),
        sa.Column("container_name", sa.String(length=255), nullable=True),
        sa.Column("remote_port", sa.Integer(), nullable=True),
        sa.Column("local_tunnel_port", sa.Integer(), nullable=True),
        sa.Column("trainer_build_version", sa.String(length=255), nullable=True),
        sa.Column("trainer_protocol_version", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["remote_server_id"], ["remote_servers.id"]),
        sa.PrimaryKeyConstraint("job_id"),
    )


def downgrade() -> None:
    """Drop per-job SSH provisioning state and registered remote servers."""
    op.drop_table("job_provisioning")
    op.drop_table("remote_servers")
