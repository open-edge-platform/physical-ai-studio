# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""add remote_servers.last_check_checks

Persists the per-check detail (image resolution, signature, device probe,
protocol) of the most recent Tier 2 ``/check`` run alongside the existing
``last_check_*`` summary columns. Previously only the rolled-up summary was
stored, so the "Image pull & verification" card had nothing to render after a
page refresh and always reset to "Not verified yet" even though the server
was last verified successfully.

Revision ID: a1c2d3e4f5a6
Revises: b7a4c1e9f0d2
Create Date: 2026-08-24 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1c2d3e4f5a6"
down_revision: str | Sequence[str] | None = "b7a4c1e9f0d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable ``last_check_checks`` JSON column."""
    op.add_column("remote_servers", sa.Column("last_check_checks", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Drop ``last_check_checks``."""
    op.drop_column("remote_servers", "last_check_checks")
