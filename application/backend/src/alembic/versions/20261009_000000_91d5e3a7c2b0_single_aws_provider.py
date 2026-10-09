"""Allow one AWS provider training target."""

import sqlalchemy as sa

from alembic import op

revision = "91d5e3a7c2b0"
down_revision = "7e2a9c4d5b61"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Enforce the AWS singleton without removing existing configurations."""
    connection = op.get_bind()
    count = connection.execute(
        sa.text("SELECT COUNT(*) FROM remote_trainers WHERE connection_mode = 'aws_batch'")
    ).scalar_one()
    if count > 1:
        raise RuntimeError("Multiple AWS providers are configured. Keep one AWS provider before upgrading.")
    op.create_index(
        "uq_remote_trainers_aws_provider",
        "remote_trainers",
        ["connection_mode"],
        unique=True,
        sqlite_where=sa.text("connection_mode = 'aws_batch'"),
    )
    connection.execute(sa.text("UPDATE remote_trainers SET name = 'AWS Provider' WHERE connection_mode = 'aws_batch'"))


def downgrade() -> None:
    """Allow multiple AWS providers."""
    op.drop_index("uq_remote_trainers_aws_provider", table_name="remote_trainers")
