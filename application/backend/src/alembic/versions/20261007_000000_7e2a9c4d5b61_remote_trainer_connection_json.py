"""Fold remote trainer connection columns into one JSON column.

Revision ID: 7e2a9c4d5b61
Revises: 7c1a9e2b4d6f
Create Date: 2026-10-07 00:00:00.000000
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "7e2a9c4d5b61"
down_revision: str | Sequence[str] | None = "7c1a9e2b4d6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SSH_COLUMNS = (
    "url",
    "ssh_host_alias",
    "ssh_hostname",
    "ssh_port",
    "ssh_username",
    "ssh_identity_file",
    "ssh_remote_port",
    "ssh_local_port",
)


def _connection_json(row: sa.Row) -> str:
    if row.connection_mode == "ssh":
        connection: dict = {
            "connection_mode": "ssh",
            "ssh_host_alias": row.ssh_host_alias,
            "ssh_connection": None,
            "ssh_remote_port": row.ssh_remote_port or 8001,
            "ssh_local_port": row.ssh_local_port or 8001,
        }
        if row.ssh_host_alias is None:
            connection["ssh_connection"] = {
                "hostname": row.ssh_hostname,
                "port": row.ssh_port or 22,
                "user": row.ssh_username,
                "identity_file": row.ssh_identity_file,
            }
        return json.dumps(connection)
    return json.dumps({"connection_mode": "direct", "url": row.url})


def upgrade() -> None:
    """Move mode-specific fields into ``connection`` and drop the old columns."""
    with op.batch_alter_table("remote_trainers") as batch_op:
        batch_op.add_column(sa.Column("connection", sa.JSON(), nullable=True))

    bind = op.get_bind()
    remote_trainers = sa.table(
        "remote_trainers",
        sa.column("id"),
        sa.column("connection_mode"),
        *(sa.column(column) for column in _SSH_COLUMNS),
    )
    rows = bind.execute(sa.select(remote_trainers)).all()
    for row in rows:
        bind.execute(
            sa.text("UPDATE remote_trainers SET connection = :connection WHERE id = :id"),
            {"connection": _connection_json(row), "id": row.id},
        )

    with op.batch_alter_table("remote_trainers") as batch_op:
        batch_op.alter_column("connection", existing_type=sa.JSON(), nullable=False)
        batch_op.alter_column("connection_mode", existing_type=sa.String(length=16), server_default=None)
        for column in _SSH_COLUMNS:
            batch_op.drop_column(column)


def downgrade() -> None:
    """Expand ``connection`` back into per-mode columns (AWS Batch rows are dropped)."""
    with op.batch_alter_table("remote_trainers") as batch_op:
        batch_op.add_column(sa.Column("url", sa.String(length=2048), nullable=True))
        batch_op.add_column(sa.Column("ssh_host_alias", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("ssh_hostname", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("ssh_port", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("ssh_username", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("ssh_identity_file", sa.String(length=4096), nullable=True))
        batch_op.add_column(sa.Column("ssh_remote_port", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("ssh_local_port", sa.Integer(), nullable=True))

    bind = op.get_bind()
    bind.execute(sa.text("DELETE FROM remote_trainers WHERE connection_mode = 'aws_batch'"))
    rows = bind.execute(sa.text("SELECT id, connection FROM remote_trainers")).all()
    for row in rows:
        connection = json.loads(row.connection) if isinstance(row.connection, str) else row.connection
        ssh = connection.get("ssh_connection") or {}
        local_port = connection.get("ssh_local_port")
        bind.execute(
            sa.text(
                "UPDATE remote_trainers SET url = :url, ssh_host_alias = :alias, ssh_hostname = :hostname, "
                "ssh_port = :port, ssh_username = :username, ssh_identity_file = :identity_file, "
                "ssh_remote_port = :remote_port, ssh_local_port = :local_port WHERE id = :id"
            ),
            {
                "url": connection.get("url") or (f"http://127.0.0.1:{local_port}" if local_port else None),
                "alias": connection.get("ssh_host_alias"),
                "hostname": ssh.get("hostname"),
                "port": ssh.get("port"),
                "username": ssh.get("user"),
                "identity_file": ssh.get("identity_file"),
                "remote_port": connection.get("ssh_remote_port"),
                "local_port": local_port,
                "id": row.id,
            },
        )

    with op.batch_alter_table("remote_trainers") as batch_op:
        batch_op.alter_column("url", existing_type=sa.String(length=2048), nullable=False)
        batch_op.create_unique_constraint("uq_remote_trainers_url", ["url"])
        batch_op.drop_column("connection")
