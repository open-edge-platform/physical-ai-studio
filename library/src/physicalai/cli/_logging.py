# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Console logging setup for studio CLI subcommands.

Library modules only ever call ``logging.getLogger(__name__)``; none of them
configure handlers, which is correct for a library but means ``logger.info``
output is dropped when the CLI runs, because the root logger defaults to
``WARNING``. Lightning solves this for itself by attaching a handler to its own
``lightning.pytorch`` logger; this module does the same for ``physicalai``.

Deliberately not ``logging.basicConfig``: that mutates the *root* logger and
would hijack logging for anything embedding the library.
"""

from __future__ import annotations

import logging
import sys

_PACKAGE_LOGGER = "physicalai"


def configure_console_logging(level: int = logging.INFO) -> None:
    """Attach a stderr handler to the ``physicalai`` logger if it has none.

    Idempotent, and a no-op when the application has already configured logging
    (either on the ``physicalai`` logger directly or on the root logger), so an
    embedding application's setup always wins.

    Args:
        level: Level to set on the ``physicalai`` logger.
    """
    package_logger = logging.getLogger(_PACKAGE_LOGGER)
    package_logger.setLevel(level)

    if package_logger.handlers or logging.getLogger().hasHandlers():
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    package_logger.addHandler(handler)
    package_logger.propagate = False
