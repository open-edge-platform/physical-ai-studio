# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Shared jsonargparse builders and dispatch helpers for studio CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

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
    parser.add_argument(
        "--weights_from",
        type=str,
        default=None,
        help=(
            "Warm-start the model's weights from a Lightning checkpoint (.ckpt) without resuming "
            "training state. Unlike --fit.ckpt_path (a full resume that also restores the optimizer, "
            "LR schedule, and global step), --weights_from loads only the checkpoint's state_dict and "
            "then re-applies this command's --model class_path/init_args on top of the checkpoint's "
            "saved hyperparameters -- so overrides such as train_expert_only or snapflow_enabled take "
            "effect and training starts fresh from step 0."
        ),
    )
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

        # Capture the raw (pre-instantiation) model init_args so they can be
        # replayed as overrides on top of the checkpoint's saved hyperparameters
        # when warm-starting. instantiate_classes() below consumes cfg in place
        # for some jsonargparse versions, so this must run first.
        weights_from = getattr(cfg, "weights_from", None)
        model_init_args = _model_init_args(cfg) if weights_from else {}

        cfg_init = cast("Namespace", parser.instantiate_classes(cfg))
        model = cfg_init.model
        if weights_from:
            model = type(model).load_from_checkpoint(weights_from, map_location="cpu", **model_init_args)

        trainer = cfg_init.trainer
        method_ns = getattr(cfg_init, method_name, None)
        # Drop unset options so Trainer's own defaults win over jsonargparse Nones.
        method_args = (
            {key: value for key, value in vars(method_ns).items() if value is not None}
            if isinstance(method_ns, Namespace)
            else {}
        )
        getattr(trainer, method_name)(model=model, datamodule=cfg_init.data, **method_args)
        return 0

    return _run


def _model_init_args(cfg: Namespace) -> dict[str, Any]:
    """Return the ``--model`` init_args as overrides for ``Policy.load_from_checkpoint``.

    Args:
        cfg: Parsed (not yet instantiated) configuration namespace.

    Returns:
        The model's ``init_args`` as a plain dict, with ``pretrained_name_or_path``
        and an unset (``None``) ``dataset_stats`` stripped so they don't shadow the
        checkpoint's own pretrained weights / saved dataset stats. Empty when the
        model namespace is absent.
    """
    model_ns = cfg.get("model")
    if not isinstance(model_ns, Namespace):
        return {}
    init_args = model_ns.get("init_args")
    if not isinstance(init_args, Namespace):
        return {}
    args = cast("dict[str, Any]", init_args.as_dict())

    # --weights_from supersedes any pretrained warm start: keeping it would trigger a
    # redundant HF fetch whose weights are immediately overwritten by the checkpoint.
    args.pop("pretrained_name_or_path", None)

    # dataset_stats is data-derived state, not config, and jsonargparse fills it with
    # None when the config omits it. Passing that None through would clobber the
    # checkpoint's saved stats, leaving the policy lazily un-built (model/preprocessor
    # stay None) so the strict state_dict load below has no modules to load into.
    # Only override when the config sets it explicitly.
    if args.get("dataset_stats") is None:
        args.pop("dataset_stats", None)

    return args
