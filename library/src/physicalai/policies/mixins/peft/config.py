# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared LoRA/DoRA configuration fields for Studio policy configs.

Any policy ``Config`` dataclass that wants LoRA support mixes in
:class:`PeftConfigMixin` alongside :class:`physicalai.config.Config`, e.g.::

    @dataclass(frozen=True)
    class MyPolicyConfig(PeftConfigMixin, Config):
        ...

Because :class:`PeftConfigMixin` fields all have defaults, it can be mixed in
without disturbing dataclass field ordering in the concrete config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal


@dataclass(frozen=True)
class PeftConfigMixin:
    """Dataclass mixin providing LoRA/DoRA fine-tuning fields.

    Attributes:
        lora_enabled: Whether to enable LoRA/DoRA fine-tuning. Defaults to False. All other
            ``lora_*`` fields have sensible defaults so that enabling LoRA does not require
            also tuning ``lora_rank``/``lora_alpha``. Requires ``peft`` to be installed and,
            in practice, a pretrained checkpoint to fine-tune from (LoRA on a randomly
            initialized model is not meaningful).

            LoRA injection freezes *all* pre-existing model parameters and trains only the
            newly created adapter weights (see ``physicalai.policies.mixins.peft.inject_lora``).
            This happens after any policy-specific ``tune_*``/``freeze_*``/``train_expert_only``
            flags have been applied, so those flags become inert once ``lora_enabled`` is True:
            whichever submodules the LoRA target regex reaches will train via their adapters
            regardless of a ``tune_*=False`` request, and submodules the regex does not reach
            will not train at all even if a ``tune_*=True`` request asked for them. To avoid this
            silently overriding your intent, concrete configs that declare
            ``_PEFT_EXCLUSIVE_FLAGS`` will raise a ``ValueError`` in ``__post_init__`` if
            ``lora_enabled=True`` is combined with a non-default value for one of those flags;
            use ``lora_target_modules`` to control which submodules are adapted instead.
        lora_rank: LoRA rank (dimension of the low-rank decomposition). Only takes effect
            when ``lora_enabled`` is True. Higher rank means more trainable parameters and
            closer to full fine-tuning. Defaults to 32, a reasonable middle ground; a lighter
            option is 16, a higher-capacity option is 64.
        lora_alpha: LoRA scaling numerator (``scaling = lora_alpha / lora_rank``). Defaults to
            ``None``, which resolves to ``lora_rank`` (i.e. a scaling of 1.0). This avoids
            PEFT's own default of 8, which under-scales higher-rank adapters (e.g. rank 64
            with alpha 8 gives scaling 0.125). Set explicitly to override, e.g. ``2 * rank``
            for a stronger adaptation signal. See ``effective_lora_alpha``.
        lora_dropout: Dropout probability applied to LoRA adapter inputs. Defaults to 0.05.
        lora_target_modules: Either a regex string or a tuple of module name suffixes
            identifying which ``nn.Linear`` submodules of the model to adapt. When ``None``
            (the default), the model's ``get_default_peft_targets()`` classmethod supplies
            the default. Pass an explicit value to override.
        lora_adapter_dtype: Precision to cast newly created LoRA parameters to, independent
            of the base model's ``dtype``. Defaults to ``"float32"`` to avoid training
            adapters in low precision when the base model uses e.g. bfloat16. Set to
            ``"auto"`` to let adapters inherit the dtype of the base layer they attach to.
        lora_use_dora: Enable DoRA (Weight-Decomposed Low-Rank Adaptation) instead of plain
            LoRA. DoRA additionally learns a per-column magnitude vector on top of the LoRA
            direction update, which typically improves quality at low ranks at the cost of
            slightly more compute/memory. Only takes effect when ``lora_enabled`` is True.
            See arxiv.org/abs/2402.09353. Defaults to False.
    """

    lora_enabled: bool = False
    lora_rank: int = 32
    lora_alpha: int | None = None
    lora_dropout: float = 0.05
    lora_target_modules: str | tuple[str, ...] | None = None
    lora_adapter_dtype: Literal["float32", "auto"] = "float32"
    lora_use_dora: bool = False

    _PEFT_EXCLUSIVE_FLAGS: ClassVar[dict[str, object]] = {}
    """Maps flag name -> default value for fields that ``inject_lora`` overrides.

    Concrete configs should override this with their trainability flags (e.g.
    ``tune_paligemma``/``tune_action_expert``/``tune_vision_encoder`` or
    ``freeze_vision_encoder``/``train_expert_only``) so that ``__post_init__`` can reject
    combining ``lora_enabled=True`` with a non-default value for one of them, since LoRA
    injection freezes all base parameters and would otherwise silently override the flag.
    """

    def __post_init__(self) -> None:
        """Validate LoRA configuration parameters.

        Raises:
            ValueError: If any ``lora_*`` field is invalid, or if ``lora_enabled`` is
                combined with a non-default value of a field listed in
                ``_PEFT_EXCLUSIVE_FLAGS``.
        """
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()

        if self.lora_rank < 0:
            msg = f"lora_rank must be >= 0, got {self.lora_rank}"
            raise ValueError(msg)

        if self.lora_enabled and self.lora_rank == 0:
            msg = "lora_rank must be > 0 when lora_enabled is True"
            raise ValueError(msg)

        if self.lora_enabled and self.lora_alpha is not None and self.lora_alpha <= 0:
            msg = f"lora_alpha must be > 0, got {self.lora_alpha}"
            raise ValueError(msg)

        if not 0.0 <= self.lora_dropout < 1.0:
            msg = f"lora_dropout must be in [0, 1), got {self.lora_dropout}"
            raise ValueError(msg)

        if self.lora_adapter_dtype not in {"float32", "auto"}:
            msg = f"Invalid lora_adapter_dtype: {self.lora_adapter_dtype}"
            raise ValueError(msg)

        if self.lora_enabled and self._PEFT_EXCLUSIVE_FLAGS:
            conflicting = {
                name: getattr(self, name)
                for name, default in self._PEFT_EXCLUSIVE_FLAGS.items()
                if getattr(self, name) != default
            }
            if conflicting:
                msg = (
                    f"lora_enabled=True is incompatible with non-default value(s) for "
                    f"{sorted(conflicting)}: LoRA injection freezes all base parameters, "
                    "so these flags would be silently overridden. Remove them (use their "
                    "defaults) and control which submodules train via lora_target_modules "
                    "instead. Got: "
                    f"{', '.join(f'{name}={value!r}' for name, value in conflicting.items())}."
                )
                raise ValueError(msg)

    @property
    def use_lora(self) -> bool:
        """Whether LoRA/DoRA fine-tuning is enabled."""
        return self.lora_enabled

    @property
    def effective_lora_alpha(self) -> int:
        """Resolve ``lora_alpha``, defaulting to ``lora_rank`` (scaling = 1.0) when unset."""
        return self.lora_alpha if self.lora_alpha is not None else self.lora_rank
