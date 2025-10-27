"""empty message

Revision ID: 83896f9ae316
Revises:
Create Date: 2025-09-23 12:51:04.790062

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "83896f9ae316"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "projects",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "project_configs",
        sa.Column("project_id", sa.Text(), nullable=False),
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("fps", sa.Integer(), nullable=False),
        sa.Column("robot_type", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )
    op.create_table(
        "datasets",
        sa.Column("project_id", sa.Text(), nullable=False),
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("path", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )
    op.create_table(
        "camera_configs",
        sa.Column("project_config_id", sa.Text(), nullable=False),
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("port_or_device_id", sa.String(length=255), nullable=False),
        sa.Column("driver", sa.String(length=255), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("fps", sa.Integer(), nullable=False),
        sa.Column("use_depth", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["project_config_id"], ["project_configs.id"], ondelete="CASCADE"),
    )

    op.create_table('jobs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('project_id', sa.Text(), nullable=False),
        sa.Column('type', sa.String(length=64), nullable=False),
        sa.Column('progress', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=64), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('start_time', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('end_time', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_table('models',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('path', sa.String(length=255), nullable=False),
        sa.Column('policy', sa.String(length=255), nullable=False),
        sa.Column('properties', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('dataset_id', sa.Text(), nullable=False),
        sa.Column('project_id', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("projects")
    op.drop_table("project_configs")
    op.drop_table("datasets")
    op.drop_table("camera_configs")
    op.drop_table('models')
    op.drop_table('jobs')
