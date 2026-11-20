# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run LeRobot's native trainer and record the loss of each update.

The training parity test runs this script with the LeRobot worker environment's Python, so native
LeRobot and the engine train with the same packages. It starts training the way the engine's
worker does, as ``lerobot-train --config_path=<path>``.

Usage: ``python _native_lerobot_train.py <train_config.json> <losses.json>``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(config_path: str, losses_path: str) -> int:
    """Train from a saved LeRobot config and write per-update losses as JSON.

    Returns:
        Process exit status.
    """
    from lerobot.scripts import lerobot_train

    losses: list[float] = []
    original_update = lerobot_train.update_policy

    def record_update(*args: object, **kwargs: object) -> object:
        result = original_update(*args, **kwargs)
        losses.append(float(result[0].to_dict(use_avg=False)["loss"]))
        return result

    lerobot_train.update_policy = record_update
    sys.argv = ["lerobot-train", f"--config_path={config_path}"]
    lerobot_train.main()
    Path(losses_path).write_text(json.dumps(losses), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
