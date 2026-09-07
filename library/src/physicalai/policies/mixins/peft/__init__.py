# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LoRA/DoRA (PEFT) support shared across Studio policies.

Composition for a new policy that wants LoRA support:

- ``config.py``: mix in :class:`PeftConfigMixin` into the policy's ``Config`` dataclass.
- ``model.py``: mix in :class:`PeftModelMixin` into the policy's ``Model`` and implement
  ``get_default_peft_targets()``.
- ``policy.py``: mix in :class:`PeftPolicyMixin` into the policy's ``Policy`` and call
  ``self._inject_lora()`` once the model is built, and
  ``self._merged_lora_model_for_export()`` from ``export()``.

See ``physicalai.policies.pi05`` for a full reference implementation.
"""

from __future__ import annotations

from .config import PeftConfigMixin
from .functions import (
    build_lora_config,
    inject_lora,
    is_lora_injected,
    log_trainable_parameters,
    merge_lora_,
)
from .model import PeftModelMixin
from .policy import PeftPolicyMixin

__all__ = [
    "PeftConfigMixin",
    "PeftModelMixin",
    "PeftPolicyMixin",
    "build_lora_config",
    "inject_lora",
    "is_lora_injected",
    "log_trainable_parameters",
    "merge_lora_",
]
