#!/bin/bash
set -euo pipefail
# -----------------------------------------------------------------------------
# run.sh - Entry point to start Physical AI Studio components.
#
# Forwards to the physicalai-studio CLI. With no argument it starts the backend
# with in-process (local) training via `serve` (TRAINING_MODE defaults to local).
# Other subcommands load the matching .env, run `uv sync` for the chosen DEVICE,
# run migrations, and start the requested component:
#
#   serve     Backend with in-process (local) training (default).
#   remote    Backend with training offloaded to a remote trainer service.
#   trainer   Remote trainer service (run this on the GPU box).
#
# Usage:
#   ./run.sh [serve|remote|trainer]
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
exec uv run --no-sync physicalai-studio "${1:-serve}" "${@:2}"
