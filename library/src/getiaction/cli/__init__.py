# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI module for getiaction.

The CLI requires training dependencies. Install with::

    pip install getiaction[train]

"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from getiaction.cli.cli import CLI

__all__ = ["CLI", "cli"]


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str) -> type[CLI]:
    """Lazy import CLI class to avoid importing training dependencies at module load.

    Args:
        name: Name of the attribute being accessed.

    Returns:
        The CLI class.

    Raises:
        ImportError: If training dependencies are not installed.
        AttributeError: If the attribute does not exist.
    """
    if name == "CLI":
        try:
            from getiaction.cli.cli import CLI  # noqa: PLC0415

            return CLI  # noqa: TRY300
        except ImportError as e:
            msg = (
                f"CLI requires training dependencies (missing: {e.name}).\nInstall with: pip install getiaction[train]"
            )
            raise ImportError(msg) from e

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def cli() -> None:
    """Entry point for getiaction CLI.

    This wrapper provides a helpful error message when training
    dependencies are not installed, instead of a raw ImportError.
    """
    try:
        from getiaction.cli.cli import cli as _cli  # noqa: PLC0415
    except ImportError as e:
        print(  # noqa: T201 - CLI entry point, logging not available yet
            f"Error: Training dependencies not installed.\n\n"
            f"The getiaction CLI requires PyTorch and Lightning.\n"
            f"Install with: pip install getiaction[train]\n\n"
            f"Missing: {e.name}",
            file=sys.stderr,
        )
        sys.exit(1)

    _cli()
