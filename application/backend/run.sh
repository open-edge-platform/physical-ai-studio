#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Script to run the Physical AI Studio server
#
# Features:
# - Runs database migrations on every start (idempotent via Alembic)
#
# Usage:
#   ./run.sh                    # Run server
#
# Environment variables:
#   APP_MODULE    Python module to run (default: src/main.py)
#   UV_CMD        Command to launch Uvicorn (default: "uv run")
#
# Requirements:
# - 'uv' CLI tool (Uvicorn) installed and available in PATH
# - Python modules and dependencies installed correctly
# -----------------------------------------------------------------------------

APP_MODULE=${APP_MODULE:-src/main.py}
UV_CMD=${UV_CMD:-uv run --no-sync}

export PYTHONUNBUFFERED=1

# Always run migrations — Alembic is idempotent and will skip
# already-applied migrations. This ensures the persistent volume
# has an up-to-date schema regardless of how it was created.
echo "Running database migrations..."
$UV_CMD physicalai-studio db migrate

echo "Starting FastAPI server..."
echo $UV_CMD "$APP_MODULE"
exec $UV_CMD "$APP_MODULE"
