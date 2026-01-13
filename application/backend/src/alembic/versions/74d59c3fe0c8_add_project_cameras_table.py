"""add project_cameras table

Revision ID: 74d59c3fe0c8
Revises: 679bf09bd098
Create Date: 2025-11-25 13:48:40.165508

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "74d59c3fe0c8"
down_revision: str | Sequence[str] | None = "679bf09bd098"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "project_cameras",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("project_id", sa.Text(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("driver", sa.String(length=50), nullable=False),
        sa.Column("fingerprint", sa.String(length=255), nullable=False),
        sa.Column("hardware_name", sa.String(length=255), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )

    # Remove the old column from project robots without migrating the old cameras
    op.drop_column("project_robots", "cameras")

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("project_cameras")
