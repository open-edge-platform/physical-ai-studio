# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Shared jsonargparse builders and dispatch helpers for studio CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from physicalai.cli._logging import configure_console_logging  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Callable

_SKIP_BY_METHOD: Final[dict[str, set[int | str]]] = {
    "fit": {"self", "model", "train_dataloaders", "val_dataloaders", "datamodule"},
    "validate": {"self", "model", "dataloaders", "datamodule"},
    "test": {"self", "model", "dataloaders", "datamodule"},
    "predict": {"self", "model", "dataloaders", "datamodule"},
}


def _build_lightning_parser(method_name: str) -> ArgumentParser:
    """Build a parser for a Trainer-backed subcommand.

    Args:
        method_name: Trainer method to invoke.

    Returns:
        Parser configured with model, data, trainer, and method arguments.
    """
    from physicalai.data import DataModule  # noqa: PLC0415
    from physicalai.policies.base import Policy  # noqa: PLC0415
    from physicalai.train import Trainer  # noqa: PLC0415

    parser = ArgumentParser(prog=f"physicalai {method_name}", description=f"Run `Trainer.{method_name}()`.")
    parser.add_argument("--config", action=ActionConfigFile, help="YAML/JSON config file.")
    parser.add_subclass_arguments(Policy, "model", required=True)
    parser.add_subclass_arguments(DataModule, "data", required=True)
    parser.add_class_arguments(Trainer, "trainer")
    parser.add_method_arguments(
        Trainer,
        method_name,
        method_name,
        skip=cast("set[int | str]", _SKIP_BY_METHOD[method_name]),
    )
    return parser


def _dispatch(method_name: str) -> Callable[[ArgumentParser, Namespace], int]:
    """Create a dispatcher that instantiates and invokes ``Trainer.<method_name>``.

    The method-level arguments registered by :func:`_build_lightning_parser`
    (``--fit.ckpt_path``, ``--validate.verbose``, ...) live in a namespace keyed
    by ``method_name`` and are forwarded verbatim, so warm-starting or resuming
    from a checkpoint works from the CLI.

    Args:
        method_name: Trainer method to invoke.

    Returns:
        Dispatcher for the runtime CLI host.
    """

    def _run(parser: ArgumentParser, cfg: Namespace) -> int:
        configure_console_logging()
        cfg_init = cast("Namespace", parser.instantiate_classes(cfg))
        trainer = cfg_init.trainer
        method_ns = getattr(cfg_init, method_name, None)
        # Drop unset options so Trainer's own defaults win over jsonargparse Nones.
        method_args = (
            {key: value for key, value in vars(method_ns).items() if value is not None}
            if isinstance(method_ns, Namespace)
            else {}
        )
        getattr(trainer, method_name)(model=cfg_init.model, datamodule=cfg_init.data, **method_args)
        return 0

    return _run
