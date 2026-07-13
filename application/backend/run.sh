#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Entry point to start the Physical AI Studio server
#
# Runs database migrations (idempotent via Alembic) and starts the backend
# with the bundled UI via the physicalai-studio serve CLI.
#
# Usage:
#   ./run.sh
# -----------------------------------------------------------------------------

export PYTHONUNBUFFERED=1

exec uv run --no-sync physicalai-studio serve
