# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manual deployment entrypoint: modal deploy application/modal/train.py."""

import sys
from pathlib import Path

# Allow the standalone Modal CLI to load the bundled worker from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from trainer.modal_app import app  # noqa: E402, F401
