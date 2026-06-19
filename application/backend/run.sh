#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Launch Physical AI Studio components
#
# Commands:
#   ./run.sh [local]      Start the backend with in-process (local) training.
#                         This is the default when no command is given.
#   ./run.sh remote       Start the backend with training offloaded to a remote
#                         trainer service. Requires TRAINER_URL to be set.
#   ./run.sh trainer      Start the remote trainer service (run this on the GPU
#                         box). Pulls dataset snapshots and trains.
#   ./run.sh help         Show this message.
#
# Each command first syncs its dependencies with `uv sync` (skip with SYNC=false)
# using the hardware extra from DEVICE (cpu/cuda/xpu). The backend `local` command
# additionally installs the `train` extra (the in-process torch training stack);
# `remote` and `trainer` omit it where it is not needed.
#
# DEVICE defaults to cpu. On a `remote` (recording) node that is the right choice:
# training is offloaded, and recording + OpenVINO inference need only cpu torch.
# Set DEVICE=cuda/xpu there only to run local `torch`-backend GPU inference.
#
# Backend features (local/remote):
# - Runs database migrations on every start (idempotent via Alembic).
# - Optionally seeds the database first by setting SEED_DB=true.
#
# Examples:
#   ./run.sh                                            # backend, local training (cpu)
#   DEVICE=cuda ./run.sh local                          # backend, local training (cuda)
#   SYNC=false ./run.sh local                           # skip uv sync, just run
#   TRAINER_URL=http://gpu-host:8001 ./run.sh remote    # recording node, cpu torch
#   DEVICE=cuda HF_TOKEN=hf_xxx ./run.sh trainer        # remote trainer service
#
# Environment variables:
#   DEVICE                 Hardware extra to sync: cpu (default), cuda, or xpu.
#   SYNC                   If "false", skip `uv sync` before launching. Default "true".
#   SEED_DB                If "true", seed the database before starting (backend).
#   TRAINING_MODE          Set automatically by the chosen command (local/remote).
#   TRAINER_URL            Remote trainer base URL. Required for `remote`.
#   TRAINER_HF_NAMESPACE   HF namespace for ephemeral snapshot repos (remote).
#   HF_TOKEN               HF token: write access on the backend (remote),
#                          read access on the trainer.
#   APP_MODULE             Backend entrypoint (default: src/main.py).
#   UV_CMD                 Launch command (default: "uv run --no-sync").
#
# Requirements:
# - 'uv' CLI installed and available in PATH.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Track whether the user explicitly chose a DEVICE so commands can pick sensible
# defaults (e.g. remote/recording nodes default to cpu — training is offloaded).
if [[ -n "${DEVICE:-}" ]]; then DEVICE_EXPLICIT=true; else DEVICE_EXPLICIT=false; fi
DEVICE=${DEVICE:-cpu}
SYNC=${SYNC:-true}
SEED_DB=${SEED_DB:-false}
APP_MODULE=${APP_MODULE:-src/main.py}
UV_CMD=${UV_CMD:-uv run --no-sync}

usage() {
	sed -n '5,49p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# Sync dependencies in the current directory with the given extras.
# Honors SYNC=false to skip and always includes the DEVICE hardware extra.
maybe_sync() {
	case "$DEVICE" in
		cpu | cuda | xpu) ;;
		*)
			echo "Error: DEVICE must be one of cpu, cuda, xpu (got '${DEVICE}')." >&2
			exit 1
			;;
	esac

	if [[ "$SYNC" != "true" ]]; then
		echo "Skipping dependency sync (SYNC=${SYNC})."
		return
	fi

	local extras=(--extra "$DEVICE" "$@")
	echo "Syncing dependencies: uv sync ${extras[*]}"
	uv sync "${extras[@]}"
}

run_backend() {
	local mode="$1"
	export TRAINING_MODE="$mode"
	export PYTHONUNBUFFERED=1
	export PYTHONPATH=.

	if [[ "$mode" == "remote" && -z "${TRAINER_URL:-}" ]]; then
		echo "Error: 'remote' mode requires TRAINER_URL to point at a running trainer service." >&2
		echo "Example: TRAINER_URL=http://gpu-host:8001 ./run.sh remote" >&2
		exit 1
	fi

	# Local training runs in-process and needs the heavy `train` extra (torch
	# stack); remote offloads training, so it stays lightweight.
	if [[ "$mode" == "local" ]]; then
		maybe_sync --extra train
	else
		# Remote/recording nodes don't train locally, so cpu torch is enough for
		# recording and OpenVINO inference. A GPU extra only matters here if you
		# deploy a model with the `torch` inference backend on a local GPU.
		if [[ "$DEVICE_EXPLICIT" == "true" && "$DEVICE" != "cpu" ]]; then
			echo "Note: DEVICE=${DEVICE} on a remote node only affects local torch-backend GPU inference;"
			echo "      training is offloaded, so cpu torch suffices for recording. Using ${DEVICE} as requested."
		fi
		maybe_sync
	fi

	# Always run migrations — Alembic is idempotent and skips already-applied
	# migrations, keeping the persistent volume's schema up to date.
	echo "Running database migrations..."
	$UV_CMD src/cli.py migrate

	if [[ "$SEED_DB" == "true" ]]; then
		echo "Seeding the database..."
		$UV_CMD application/cli.py init-db
		$UV_CMD application/cli.py seed --with-model=True
	fi

	echo "Starting FastAPI server (TRAINING_MODE=${mode})..."
	echo "$UV_CMD $APP_MODULE"
	exec $UV_CMD "$APP_MODULE"
}

run_trainer() {
	local trainer_dir="${SCRIPT_DIR}/../trainer"

	if [[ ! -d "$trainer_dir" ]]; then
		echo "Error: trainer directory not found at ${trainer_dir}." >&2
		exit 1
	fi

	if [[ -z "${HF_TOKEN:-}" ]]; then
		echo "Warning: HF_TOKEN is not set; the trainer cannot pull dataset snapshots." >&2
	fi

	export PYTHONUNBUFFERED=1
	cd "$trainer_dir"

	maybe_sync

	echo "Starting remote trainer service..."
	exec $UV_CMD python -m trainer.main
}

COMMAND=${1:-local}
case "$COMMAND" in
	local) run_backend local ;;
	remote) run_backend remote ;;
	trainer) run_trainer ;;
	-h | --help | help) usage ;;
	*)
		echo "Unknown command: ${COMMAND}" >&2
		usage
		exit 1
		;;
esac
