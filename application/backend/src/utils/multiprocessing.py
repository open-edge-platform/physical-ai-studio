# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Multiprocessing helpers."""

from __future__ import annotations

import multiprocessing as mp

from loguru import logger


def ensure_spawn_start_method() -> None:
    """Ensure multiprocessing uses ``spawn`` to stay CUDA/XPU-safe.

    CUDA and XPU cannot be safely re-initialized in forked subprocesses. This helper enforces
    ``spawn`` before any worker processes/queues/events are created.
    """
    current = mp.get_start_method(allow_none=True)
    if current == "spawn":
        return

    mp.set_start_method("spawn", force=True)
    if current is None:
        logger.info("Set multiprocessing start method to 'spawn'")
    else:
        logger.warning(f"Overrode multiprocessing start method from '{current}' to 'spawn'")
