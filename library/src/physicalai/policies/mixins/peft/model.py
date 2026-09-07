# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared LoRA/DoRA model-side hook for Physical AI policy models."""

from __future__ import annotations


class PeftModelMixin:
    """Mixin declaring the LoRA target-module hook a PEFT-capable model must implement.

    Mix this into a policy's ``Model`` subclass (alongside
    :class:`physicalai.policies.base.Model`) to opt into LoRA support, then implement
    :meth:`get_default_peft_targets` to describe which submodules should be adapted by
    default when the policy's config does not set an explicit ``lora_target_modules``.
    """

    @classmethod
    def get_default_peft_targets(cls) -> str | tuple[str, ...]:
        """Return the default LoRA ``target_modules`` for this model.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        msg = (
            f"{cls.__name__} does not define get_default_peft_targets(); "
            "LoRA target_modules must be provided explicitly via lora_target_modules."
        )
        raise NotImplementedError(msg)
