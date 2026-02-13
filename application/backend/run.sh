#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Script to run the Geti Action server
#
# Features:
# - Runs database migrations on every start (idempotent via Alembic)
# - Optionally seeds the database before starting the server by setting:
#     SEED_DB=true
#
# Usage:
#   SEED_DB=true ./run.sh       # Seed database before launching server
#   ./run.sh                    # Run server without seeding
#
# Environment variables:
#   SEED_DB       If set to "true", runs database seeding before starting.
#   APP_MODULE    Python module to run (default: src/main.py)
#   UV_CMD        Command to launch Uvicorn (default: "uv run")
#
# Requirements:
# - 'uv' CLI tool (Uvicorn) installed and available in PATH
# - Python modules and dependencies installed correctly
# -----------------------------------------------------------------------------

SEED_DB=${SEED_DB:-false}
APP_MODULE=${APP_MODULE:-src/main.py}
UV_CMD=${UV_CMD:-uv run}

export PYTHONUNBUFFERED=1
export PYTHONPATH=.

# Always run migrations — Alembic is idempotent and will skip
# already-applied migrations. This ensures the persistent volume
# has an up-to-date schema regardless of how it was created.
echo "Running database migrations..."
uv run src/cli.py migrate

if [[ "$SEED_DB" == "true" ]]; then
    echo "Seeding the database..."
    $UV_CMD application/cli.py init-db
    $UV_CMD application/cli.py seed --with-model=True
fi

echo "Starting FastAPI server..."

exec $UV_CMD "$APP_MODULE"
