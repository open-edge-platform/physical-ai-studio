# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot config equivalence tests for the curated starter YAMLs.

For every starter YAML under ``library/configs/lerobot/`` this test:

1. Constructs the upstream ``PreTrainedConfig`` subclass from the YAML's
   ``model.init_args`` (the wrapper passes those kwargs through unchanged
   to the upstream dataclass; see
   ``physicalai/policies/lerobot/policy.py:634-651``).
2. Constructs the same subclass with no arguments (pure upstream defaults).
3. Asserts that every dataclass field is equal between the two, EXCEPT
   for an explicit per-policy allow-list of deliberate overrides.

Any field that drifts from upstream defaults without being declared in
``DELIBERATE_OVERRIDES`` fails the test with the field name -- catching
silent schema drift between a LeRobot version bump and our starter
configs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.xvla.configuration_xvla import XVLAConfig

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs" / "lerobot"


# Map from starter YAML name to the upstream LeRobot dataclass it configures.
UPSTREAM_CONFIG: dict[str, type] = {
    "diffusion.yaml": DiffusionConfig,
    "groot.yaml": GrootConfig,
    "pi0.yaml": PI0Config,
    "pi05.yaml": PI05Config,
    "pi0_fast.yaml": PI0FastConfig,
    "xvla.yaml": XVLAConfig,
}


# Per-policy fields that are deliberately overridden away from upstream
# defaults in the starter YAML. Any OTHER drift is treated as a bug and
# fails the test. Keep this list small and justified -- it is the only
# escape hatch for legitimate divergence from upstream defaults.
#
# Justifications:
#   diffusion.crop_shape: starter uses [84, 84] for faster training on the
#     pinned 50fps demo dataset; upstream default ``None`` falls back to
#     full image size which is slower for the smoke-training use case.
#   pi0/pi05/pi0_fast/xvla.chunk_size + n_action_steps: tuned for the
#     pinned 50fps starter dataset's action horizon.
#   xvla.florence_config: upstream default ``{}`` is non-functional --
#     ``XVLAConfig.get_florence_config()`` raises ``ValueError`` because
#     it requires both ``vision_config`` and ``text_config`` keys to
#     exist. The starter spells out the canonical Florence-2 DaViT
#     config so the policy is actually constructible from the YAML
#     without further wiring.
DELIBERATE_OVERRIDES: dict[str, set[str]] = {
    "diffusion.yaml": {"crop_shape"},
    "groot.yaml": set(),
    "pi0.yaml": {"chunk_size", "n_action_steps"},
    "pi05.yaml": {"chunk_size", "n_action_steps"},
    "pi0_fast.yaml": {"chunk_size", "n_action_steps"},
    "xvla.yaml": {"chunk_size", "n_action_steps", "florence_config"},
}


# Fields the upstream base ``PreTrainedConfig.__post_init__`` may rewrite
# regardless of YAML input (e.g. ``device`` is autodetected). These are
# always ignored when comparing dataclass field values.
POST_INIT_DERIVED_FIELDS: frozenset[str] = frozenset({"device", "use_amp"})


def _load(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open("r") as f:
        return yaml.safe_load(f)


def _config_field_dict(cfg: DataclassInstance) -> dict[str, Any]:
    """Return ``{field_name: value}`` for every dataclass field on ``cfg``.

    Avoids ``dataclasses.asdict`` because it deep-copies nested dataclasses
    and chokes on non-dataclass attributes that some configs cache in
    ``__post_init__`` (e.g. ``XVLAConfig._florence_config_obj``).
    """
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


def _values_equivalent(a: Any, b: Any) -> bool:
    """Compare two config field values with tuple/list normalization.

    YAML cannot express Python tuples, so any sequence loaded from YAML
    arrives as a ``list`` even when the upstream dataclass default is a
    ``tuple``. Treat ``list`` and ``tuple`` of equal element-wise values
    as equivalent (recursively, for nested sequences). All other types
    fall back to ``==``.
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_values_equivalent(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("yaml_name", sorted(UPSTREAM_CONFIG.keys()))
def test_yaml_init_args_match_upstream_defaults(yaml_name: str) -> None:
    """Every YAML model.init_args field must match upstream defaults except for declared overrides.

    This catches the class of bugs where a LeRobot version bump silently
    changes a default (e.g. tokenizer_max_length) and our starter YAML
    keeps the old hard-coded value.
    """
    config_cls = UPSTREAM_CONFIG[yaml_name]
    yaml_init_args = _load(CONFIGS_DIR / yaml_name)["model"]["init_args"]

    from_yaml = _config_field_dict(config_cls(**yaml_init_args))
    upstream_default = _config_field_dict(config_cls())

    allowed_drift = DELIBERATE_OVERRIDES[yaml_name] | POST_INIT_DERIVED_FIELDS
    drift = {
        name: (upstream_default[name], from_yaml[name])
        for name in from_yaml
        if name not in allowed_drift and not _values_equivalent(from_yaml[name], upstream_default[name])
    }
    assert not drift, (
        f"{yaml_name}: fields drifted from upstream defaults without being "
        f"declared in DELIBERATE_OVERRIDES. Either revert the YAML to the "
        f"upstream default or extend DELIBERATE_OVERRIDES['{yaml_name}']:\n"
        + "\n".join(f"  {name}: upstream={u!r} yaml={y!r}" for name, (u, y) in sorted(drift.items()))
    )


@pytest.mark.parametrize("yaml_name", sorted(DELIBERATE_OVERRIDES.keys()))
def test_declared_overrides_actually_differ_from_upstream(yaml_name: str) -> None:
    """Every field in DELIBERATE_OVERRIDES must actually differ from upstream defaults.

    Stops the override list from accumulating dead entries when upstream
    eventually moves to match our value.
    """
    overrides = DELIBERATE_OVERRIDES[yaml_name]
    if not overrides:
        pytest.skip(f"{yaml_name}: no deliberate overrides declared")

    config_cls = UPSTREAM_CONFIG[yaml_name]
    yaml_init_args = _load(CONFIGS_DIR / yaml_name)["model"]["init_args"]
    from_yaml = _config_field_dict(config_cls(**yaml_init_args))
    upstream_default = _config_field_dict(config_cls())

    no_op_overrides = {name for name in overrides if _values_equivalent(from_yaml[name], upstream_default[name])}
    assert not no_op_overrides, (
        f"{yaml_name}: DELIBERATE_OVERRIDES contains fields that no longer "
        f"differ from upstream defaults; remove them: {sorted(no_op_overrides)}"
    )
