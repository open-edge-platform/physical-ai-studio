# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared LoRA/DoRA policy-lifecycle mixin for Studio policies."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Protocol, cast

import torch

from .functions import build_lora_config, inject_lora, is_lora_injected, merge_lora_

if TYPE_CHECKING:
    from os import PathLike

    from torch import nn

    from physicalai.export import ExportBackend

    from .config import PeftConfigMixin
    from .model import PeftModelMixin

    class _PeftCapableModel(PeftModelMixin, nn.Module):
        """Structural type for the ``model`` attribute expected by :class:`PeftPolicyMixin`."""

    class _PeftPolicyHost(Protocol):
        """Structural type for the ``self`` a :class:`PeftPolicyMixin` method is mixed into."""

        config: PeftConfigMixin
        model: _PeftCapableModel | None

        def export(
            self,
            output_path: PathLike | str,
            backend: ExportBackend | str,
            input_sample: dict[str, torch.Tensor] | None = None,
            **export_kwargs: dict,
        ) -> None:
            """Cooperative ``super().export()`` provided further down the MRO."""


logger = logging.getLogger(__name__)


class PeftPolicyMixin:
    """Mixin providing the LoRA injection/export lifecycle for a Studio ``Policy``.

    Expects the concrete ``Policy`` subclass to expose:

    - ``self.config``: a config mixing in :class:`physicalai.policies.mixins.peft.PeftConfigMixin`.
    - ``self.model``: an ``nn.Module`` mixing in
      :class:`physicalai.policies.mixins.peft.PeftModelMixin` (i.e. implementing
      ``get_default_peft_targets()``), once initialized.

    Typical usage in a policy's ``_initialize_model``::

        if self.config.use_lora:
            self._inject_lora()

    ``export()`` is provided by this mixin (see below); place ``PeftPolicyMixin`` ahead of
    ``ExportablePolicyMixin`` (or any other mixin defining ``export()``) in the class's base
    list so it participates in the cooperative ``super().export()`` chain, e.g.::

        class MyPolicy(PeftPolicyMixin, ExportablePolicyMixin, Policy):
            ...
    """

    def _inject_lora(self) -> None:
        """Inject LoRA adapters into ``self.model``, freezing all base parameters.

        Intended to be called from ``_initialize_model`` (or equivalent) once the model
        has been constructed and any pretrained weights have been loaded. Also useful to
        re-inject adapters when a checkpoint is restored, since Lightning's
        ``load_from_checkpoint`` reruns model construction from hyperparameters before
        restoring the state dict.

        Raises:
            RuntimeError: If ``self.model`` has not been initialized yet.
        """
        self_ = cast("_PeftPolicyHost", self)
        if self_.model is None:
            msg = "Cannot inject LoRA before the model has been initialized."
            raise RuntimeError(msg)

        target_modules = self_.config.lora_target_modules or self_.model.get_default_peft_targets()
        adapter_dtype = None if self_.config.lora_adapter_dtype == "auto" else torch.float32
        lora_config = build_lora_config(
            rank=self_.config.lora_rank,
            alpha=self_.config.effective_lora_alpha,
            dropout=self_.config.lora_dropout,
            target_modules=target_modules,
            use_dora=self_.config.lora_use_dora,
        )
        inject_lora(self_.model, lora_config, adapter_dtype=adapter_dtype)

    def on_fit_start(self) -> None:
        """Guard against ``lora_enabled=True`` policies that forgot to inject adapters.

        Runs once training starts (after ``setup()``/``_initialize_model`` has had a
        chance to call ``self._inject_lora()``). Without this guard, a policy that mixes
        in :class:`PeftPolicyMixin` but forgets to call ``self._inject_lora()`` from its
        model-construction path would silently fall back to full fine-tuning of every
        parameter instead of failing loudly.

        Raises:
            RuntimeError: If ``self.config.use_lora`` is True but LoRA adapters are not
                present on ``self.model``.
        """
        super_on_fit_start = getattr(super(), "on_fit_start", None)
        if callable(super_on_fit_start):
            super_on_fit_start()

        self_ = cast("_PeftPolicyHost", self)
        if self_.config.use_lora and self_.model is not None and not is_lora_injected(self_.model):
            msg = (
                f"{type(self).__name__}.config.use_lora is True but no LoRA adapters are "
                "injected into self.model. This policy mixes in PeftPolicyMixin but never "
                "called self._inject_lora() during model construction (e.g. from "
                "_initialize_model()); without adapters, training would silently fall back "
                "to full fine-tuning of every parameter instead of LoRA."
            )
            raise RuntimeError(msg)

    def export(
        self,
        output_path: PathLike | str,
        backend: ExportBackend | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict,
    ) -> None:
        """Export the policy, merging any LoRA adapters into base weights first.

        If LoRA is enabled and currently injected, exporting is done on a disposable deep
        copy of ``self.model`` with adapters merged in-place via ``merge_lora_``, so the
        exported artifact has no ``peft`` dependency and matches the plain (adapter-free)
        export contract consumed by Runtime's ``InferenceModel``. The live training model
        (``self.model``) is left untouched. Merging bfloat16-precision LoRA adapters into
        bfloat16 base weights is lossy; this only affects the exported copy.

        Delegates to ``super().export(...)`` (e.g. ``ExportablePolicyMixin.export()``) to
        perform the actual backend export, with ``self.model`` temporarily swapped to the
        merged copy for the duration of that call.

        Args:
            output_path: The file path where the exported model will be saved.
            backend: The export backend to use.
            input_sample: A sample input tensor dictionary for model tracing.
            **export_kwargs: Additional keyword arguments forwarded to the backend export.
        """
        self_ = cast("_PeftPolicyHost", self)
        super_export = cast("_PeftPolicyHost", super()).export
        merged_model = self._merged_lora_model_for_export()
        if merged_model is None:
            super_export(output_path, backend, input_sample, **export_kwargs)
            return

        logger.info("Merging LoRA adapters into a copy of the model before export.")
        original_model = self_.model
        self_.model = cast("_PeftCapableModel | None", merged_model)
        try:
            super_export(output_path, backend, input_sample, **export_kwargs)
        finally:
            self_.model = original_model

    def _merged_lora_model_for_export(self) -> nn.Module | None:
        """Return a disposable deep copy of ``self.model`` with LoRA adapters merged in.

        Intended for use inside ``export()`` so exported artifacts fold LoRA adaptation
        into the base layer weights and carry no ``peft`` dependency. Returns ``None`` if
        LoRA is not enabled or not currently injected, in which case callers should export
        ``self.model`` directly.

        Returns:
            A merged deep copy of ``self.model``, or ``None`` if there is nothing to merge.
        """
        self_ = cast("_PeftPolicyHost", self)
        if not (self_.config.use_lora and self_.model is not None and is_lora_injected(self_.model)):
            return None
        merged_model = copy.deepcopy(self_.model)
        merge_lora_(merged_model)
        return merged_model
