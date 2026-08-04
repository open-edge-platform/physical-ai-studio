#!/bin/bash
set -euo pipefail
# -----------------------------------------------------------------------------
# run.sh - Entry point to start Physical AI Studio components.
#
# Forwards to the physicalai-studio CLI. With no argument it starts the backend
# via `serve`, supporting local and remote training at the same time.

# The remote trainer service is a separate entry point in this same project:
# `uv run physicalai-trainer` (see docs/remote-trainer.md). Local training does
# not need it; it calls the training code in-process.
#
# Usage:
#   ./run.sh [physicalai-studio arguments]
#   ./run.sh [serve]
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
exec uv run --no-sync physicalai-studio "${1:-serve}" "${@:2}"
