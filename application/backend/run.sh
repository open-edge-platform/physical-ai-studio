#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Script to run the Physical AI Studio server
#
# Features:
# - Runs database migrations on every start (idempotent via Alembic)
# - Remembers the hardware variant (cpu/cuda/xpu) across restarts
# - Optionally seeds the database before starting the server by setting:
#     SEED_DB=true
#
# Usage:
#   ./run.sh --setup xpu        # Persist hardware variant and sync deps
#   ./run.sh --setup cpu        # Switch to CPU-only PyTorch
#   ./run.sh --setup cuda       # Switch to CUDA PyTorch
#   ./run.sh                    # Run server (uses previously set variant)
#   SEED_DB=true ./run.sh       # Seed database before launching server
#
# The --setup flag writes the chosen variant to .uv-extra and runs
# `uv sync --extra <variant>`. On subsequent runs, the variant is read
# from .uv-extra and passed to all `uv run` invocations automatically.
#
# You can also set up manually:
#   echo "xpu" > .uv-extra && uv sync --extra xpu
#
# Environment variables:
#   SEED_DB       If set to "true", runs database seeding before starting.
#   APP_MODULE    Python module to run (default: src/main.py)
#   UV_CMD        Command to launch Uvicorn (default: "uv run")
#
# Requirements:
# - 'uv' CLI tool installed and available in PATH
# - Python modules and dependencies installed correctly
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
UV_EXTRA_FILE="$SCRIPT_DIR/.uv-extra"

# Handle --setup flag: persist hardware variant and run uv sync
if [[ "${1:-}" == "--setup" ]]; then
    VARIANT="${2:?Usage: ./run.sh --setup <cpu|xpu|cuda>}"
    echo "$VARIANT" >"$UV_EXTRA_FILE"
    echo "Syncing with hardware variant: $VARIANT"
    uv sync --extra "$VARIANT"
    shift 2
fi

# Read persisted hardware variant for all uv run invocations
UV_EXTRA_FLAG=""
if [[ -f "$UV_EXTRA_FILE" ]]; then
    UV_EXTRA_FLAG="--extra $(cat "$UV_EXTRA_FILE")"
    echo "Using hardware variant: $(cat "$UV_EXTRA_FILE")"
fi

SEED_DB=${SEED_DB:-false}
APP_MODULE=${APP_MODULE:-src/main.py}
UV_CMD=${UV_CMD:-uv run}

export PYTHONUNBUFFERED=1
export PYTHONPATH=.

# Always run migrations — Alembic is idempotent and will skip
# already-applied migrations. This ensures the persistent volume
# has an up-to-date schema regardless of how it was created.
echo "Running database migrations..."
uv run $UV_EXTRA_FLAG src/cli.py migrate

if [[ "$SEED_DB" == "true" ]]; then
    echo "Seeding the database..."
    $UV_CMD $UV_EXTRA_FLAG application/cli.py init-db
    $UV_CMD $UV_EXTRA_FLAG application/cli.py seed --with-model=True
fi

echo "Starting FastAPI server..."

exec $UV_CMD $UV_EXTRA_FLAG "$APP_MODULE"
