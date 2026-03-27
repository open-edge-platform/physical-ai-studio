#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Script to run the Physical AI Studio server
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
#
# Requirements:
# - Python virtual environment activated (PATH includes .venv/bin)
# - Dependencies installed at build time
# -----------------------------------------------------------------------------

SEED_DB=${SEED_DB:-false}
APP_MODULE=${APP_MODULE:-src/main.py}

export PYTHONUNBUFFERED=1
export PYTHONPATH=.

# Always run migrations — Alembic is idempotent and will skip
# already-applied migrations. This ensures the persistent volume
# has an up-to-date schema regardless of how it was created.
echo "Running database migrations..."
python src/cli.py migrate

if [[ "$SEED_DB" == "true" ]]; then
    echo "Seeding the database..."
    python application/cli.py init-db
    python application/cli.py seed --with-model=True
fi

echo "Starting FastAPI server..."

echo python "$APP_MODULE"
exec python "$APP_MODULE"
