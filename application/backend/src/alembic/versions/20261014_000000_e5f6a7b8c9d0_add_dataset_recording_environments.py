"""Support dataset recording environments and a default per dataset.

Revision ID: e5f6a7b8c9d0
Revises: 7c1a9e2b4d6f
Create Date: 2026-10-14 00:00:00.000000
"""

from collections.abc import Sequence
from uuid import uuid4

import sqlalchemy as sa

from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | Sequence[str] | None = "91d5e3a7c2b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Move legacy assignments into associations and mark a default association."""
    connection = op.get_bind()
    if connection.dialect.name == "sqlite":
        connection.execute(sa.text("PRAGMA foreign_keys = OFF"))

    inspector = sa.inspect(connection)
    if inspector.has_table("dataset_recording_environments"):
        recording_environments = sa.Table("dataset_recording_environments", sa.MetaData(), autoload_with=connection)
        recording_environment_columns = {
            column["name"] for column in inspector.get_columns("dataset_recording_environments")
        }
        if "is_default" not in recording_environment_columns:
            op.add_column(
                "dataset_recording_environments",
                sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
            )
            recording_environments = sa.Table(
                "dataset_recording_environments", sa.MetaData(), autoload_with=connection
            )
    else:
        recording_environments = op.create_table(
            "dataset_recording_environments",
            sa.Column("id", sa.Text(), nullable=False),
            sa.Column("dataset_id", sa.Text(), nullable=False),
            sa.Column("environment_id", sa.Text(), nullable=False),
            sa.Column("mappings", sa.JSON(), nullable=True),
            sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["environment_id"], ["project_environments.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "dataset_id",
                "environment_id",
                name="uq_dataset_recording_environments_dataset_environment",
            ),
        )

    dataset_columns = {column["name"] for column in inspector.get_columns("datasets")}
    if "environment_id" in dataset_columns:
        assignments = connection.execute(sa.text("SELECT id, environment_id FROM datasets")).mappings().all()
        existing = {
            (row["dataset_id"], row["environment_id"])
            for row in connection.execute(
                sa.text("SELECT dataset_id, environment_id FROM dataset_recording_environments")
            ).mappings()
        }
        rows = [
            {"id": str(uuid4()), "dataset_id": row["id"], "environment_id": row["environment_id"]}
            for row in assignments
            if (row["id"], row["environment_id"]) not in existing
        ]
        if rows:
            connection.execute(recording_environments.insert(), rows)
        with op.batch_alter_table("datasets") as batch_op:
            batch_op.drop_column("environment_id")

    op.execute(
        sa.text(
            "UPDATE dataset_recording_environments SET is_default = 1 "
            "WHERE id IN (SELECT MIN(id) FROM dataset_recording_environments GROUP BY dataset_id)"
        )
    )
    op.create_index(
        "uq_dataset_recording_environments_default",
        "dataset_recording_environments",
        ["dataset_id"],
        unique=True,
        sqlite_where=sa.text("is_default = 1"),
        postgresql_where=sa.text("is_default"),
    )

    if connection.dialect.name == "sqlite":
        connection.execute(sa.text("PRAGMA foreign_keys = ON"))


def downgrade() -> None:
    """Restore the legacy environment column and remove recording associations."""
    with op.batch_alter_table("datasets") as batch_op:
        batch_op.add_column(sa.Column("environment_id", sa.Text(), nullable=True))
    op.execute(
        sa.text(
            "UPDATE datasets SET environment_id = (SELECT environment_id FROM dataset_recording_environments "
            "WHERE dataset_recording_environments.dataset_id = datasets.id ORDER BY created_at, id LIMIT 1)"
        )
    )
    op.drop_table("dataset_recording_environments")
