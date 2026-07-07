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
#
# The remote trainer service is launched from the trainer project with its own
# `physicalai-trainer` command (see application/trainer/README.md).
#
# Usage:
#   ./run.sh [serve|remote]
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
exec uv run --no-sync physicalai-studio "${1:-serve}" "${@:2}"
