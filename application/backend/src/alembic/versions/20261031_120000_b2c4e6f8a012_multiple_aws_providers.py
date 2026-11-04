"""Allow multiple AWS provider training targets."""

import sqlalchemy as sa

from alembic import op

revision = "b2c4e6f8a012"
down_revision = "91d5e3a7c2b0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Drop the AWS provider singleton index so multiple AWS Batch targets can be created."""
    op.drop_index("uq_remote_trainers_aws_provider", table_name="remote_trainers")


def downgrade() -> None:
    """Restore the AWS singleton index when at most one AWS provider exists."""
    connection = op.get_bind()
    count = connection.execute(
        sa.text("SELECT COUNT(*) FROM remote_trainers WHERE connection_mode = 'aws_batch'")
    ).scalar_one()
    if count > 1:
        raise RuntimeError("Multiple AWS providers are configured. Keep one AWS provider before downgrading.")
    op.create_index(
        "uq_remote_trainers_aws_provider",
        "remote_trainers",
        ["connection_mode"],
        unique=True,
        sqlite_where=sa.text("connection_mode = 'aws_batch'"),
    )
