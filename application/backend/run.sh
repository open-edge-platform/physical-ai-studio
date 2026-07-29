#!/bin/bash
set -euo pipefail
# -----------------------------------------------------------------------------
# run.sh - Entry point to start Physical AI Studio components.
#
# Forwards to the physicalai-studio CLI. With no argument it starts the backend
# via `serve`, supporting local and remote training at the same time.

# The remote trainer service is launched from the trainer project with its own
# `physicalai-trainer` command (see application/trainer/README.md).
#
# Usage:
#   ./run.sh [physicalai-studio arguments]
#   ./run.sh [serve]
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
exec uv run --no-sync physicalai-studio "${1:-serve}" "${@:2}"
