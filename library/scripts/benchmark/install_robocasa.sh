#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Install robocasa + robosuite into the active PAS library venv.
#
# Why this is a script and not a pyproject extra:
#   - robocasa is not on PyPI.
#   - robocasa's setup.py pins `lerobot==0.3.3`, which collides with
#     this repo's `lerobot>=0.5.1` and makes uv's unified resolve
#     unsolvable. We must install with `--no-deps`.
#   - robosuite needs the master branch (>=1.5dev) for HybridMobileBase
#     and composite controllers; PyPI only has 1.4.0.
#
# SHAs match lerobot/docker/Dockerfile.benchmark.robocasa. Bump together.
#
# Mutually exclusive with the [libero] extra: libero pulls robosuite==1.4.0
# from PyPI, which this script would overwrite with robosuite master
# (1.5dev). Use a separate venv for each benchmark; do NOT pass --extra all
# (which transitively includes [libero]).
#
# Usage:
#   uv venv .venv-robocasa
#   source .venv-robocasa/bin/activate
#   # Pick exactly one torch backend extra. --extra all is wrong here
#   # because it pulls [libero] -> robosuite==1.4.0.
#   uv sync --active --extra cu128       # or --extra cpu / --extra xpu
#   bash library/scripts/benchmark/install_robocasa.sh

set -euo pipefail

ROBOCASA_SHA="${ROBOCASA_SHA:-56e355ccc64389dfc1b8a61a33b9127b975ba681}"
ROBOSUITE_SHA="${ROBOSUITE_SHA:-aaa8b9b214ce8e77e82926d677b4d61d55e577ab}"
CLONE_ROOT="${CLONE_ROOT:-$HOME/.cache/physicalai/robocasa-src}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "error: no active virtual environment; activate one first." >&2
    exit 1
fi

if python -c "import libero" 2>/dev/null; then
    echo "error: libero is installed in the active environment." >&2
    echo "This script installs robosuite master (1.5dev), which conflicts with" >&2
    echo "libero's pinned robosuite==1.4.0. Use a separate venv for robocasa." >&2
    echo "  uv venv .venv-robocasa" >&2
    echo "  source .venv-robocasa/bin/activate" >&2
    echo "  uv sync --active --extra cu128  # or --extra cpu / --extra xpu" >&2
    echo "  bash library/scripts/benchmark/install_robocasa.sh" >&2
    exit 1
fi

mkdir -p "$CLONE_ROOT"

if [[ ! -d "$CLONE_ROOT/robocasa" ]]; then
    git clone https://github.com/robocasa/robocasa.git "$CLONE_ROOT/robocasa"
fi
git -C "$CLONE_ROOT/robocasa" fetch --quiet origin "$ROBOCASA_SHA"
git -C "$CLONE_ROOT/robocasa" checkout --quiet "$ROBOCASA_SHA"

if [[ ! -d "$CLONE_ROOT/robosuite" ]]; then
    git clone https://github.com/ARISE-Initiative/robosuite.git "$CLONE_ROOT/robosuite"
fi
git -C "$CLONE_ROOT/robosuite" fetch --quiet origin "$ROBOSUITE_SHA"
git -C "$CLONE_ROOT/robosuite" checkout --quiet "$ROBOSUITE_SHA"

# --no-deps on robocasa: skip the shadowed lerobot==0.3.3 pin.
uv pip install --no-cache -e "$CLONE_ROOT/robocasa" --no-deps
uv pip install --no-cache -e "$CLONE_ROOT/robosuite"

# robocasa's actual runtime deps (the ones its setup.py would have pulled
# minus lerobot and tianshou — tianshou is never imported by robocasa).
uv pip install --no-cache \
    "numpy==2.2.5" "numba==0.61.2" "scipy==1.15.3" "mujoco==3.3.1" \
    pygame Pillow opencv-python pyyaml pynput tqdm termcolor \
    imageio h5py lxml hidapi "gymnasium>=0.29.1"

ROBOCASA_MACROS_PRIVATE="$CLONE_ROOT/robocasa/robocasa/macros_private.py"
if [[ -f "$ROBOCASA_MACROS_PRIVATE" ]]; then
    echo "robocasa macros already configured at $ROBOCASA_MACROS_PRIVATE; skipping setup_macros."
else
    # Keep setup non-interactive for CI.
    yes y | python -m robocasa.scripts.setup_macros
fi

echo
echo "robocasa + robosuite installed."
echo "Next: download kitchen assets (~4.4 GB on disk; the script prompts"
echo "with a hardcoded '~10 Gb' that counts the two object packs we skip):"
echo "    yes y | python -m robocasa.scripts.download_kitchen_assets \\"
echo "        --type tex tex_generative fixtures_lw objs_lw"
echo
echo "For headless servers also set:    export MUJOCO_GL=egl"
