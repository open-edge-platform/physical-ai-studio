"""Remove unsupported SSH jobs and their database tables.

Jobs with ``training_target='ssh'`` cannot be read by the current job schema.
Delete those jobs and their provisioning tables while retaining trained models.

Revision ID: 7c1a9e2b4d6f
Revises: 4b8d2f6a1c30
Create Date: 2026-09-21 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "7c1a9e2b4d6f"
down_revision: str | Sequence[str] | None = "4b8d2f6a1c30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Remove unsupported SSH jobs and their tables."""
    container = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT coalesce(container_name, container_id) FROM job_provisioning "
                "WHERE container_name IS NOT NULL OR container_id IS NOT NULL LIMIT 1"
            )
        )
        .scalar()
    )
    if container is not None:
        raise RuntimeError(f"Remove provisioned container {container} before migrating")

    # Retain trained models without a reference to their deleted jobs.
    op.execute(
        sa.text(
            "UPDATE models SET train_job_id = NULL WHERE train_job_id IN "
            "(SELECT id FROM jobs WHERE json_extract(payload, '$.training_target') = 'ssh')"
        )
    )
    op.drop_table("job_provisioning")
    op.execute(sa.text("DELETE FROM jobs WHERE json_extract(payload, '$.training_target') = 'ssh'"))
    op.drop_table("remote_servers")


def downgrade() -> None:
    """Recreate empty tables; deleted records cannot be restored."""
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
        sa.Column("last_check_checks", sa.JSON(), nullable=True),
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
        sa.ForeignKeyConstraint(["remote_server_id"], ["remote_servers.id"], ondelete="RESTRICT"),
    )
