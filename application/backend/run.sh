#!/bin/bash
set -euo pipefail
# -----------------------------------------------------------------------------
# run.sh - Entry point to start Physical AI Studio components.
#
# Forwards to the physicalai-studio CLI. With no argument it starts the backend
# via `serve`. Training jobs can run locally or on a trainer URL configured in
# the Studio UI.
#
# Usage: ./run.sh [physicalai-studio arguments]
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
exec uv run --no-sync physicalai-studio "${1:-serve}" "${@:2}"
