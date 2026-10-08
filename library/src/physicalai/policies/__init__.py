# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action trainer policies.

Policy implementations are imported on demand so their optional dependencies remain optional.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from . import lerobot
from .base import Policy
from .lerobot import get_lerobot_policy

if TYPE_CHECKING:
    from .act import ACT, ACTConfig, ACTModel
    from .cosmos3 import Cosmos3, Cosmos3Config, Cosmos3Model
    from .molmoact2 import MolmoAct2, MolmoAct2Config, MolmoAct2Model
    from .pi05 import Pi05, Pi05Config, Pi05Model
    from .rldx1 import Rldx1, Rldx1Config, Rldx1Model
    from .smolvla import SmolVLA, SmolVLAConfig, SmolVLAModel
    from .xr0 import XR0, XR0Config, XR0Model


_POLICY_EXPORTS = {
    "ACT": (".act", "ACT"),
    "ACTConfig": (".act", "ACTConfig"),
    "ACTModel": (".act", "ACTModel"),
    "Cosmos3": (".cosmos3", "Cosmos3"),
    "Cosmos3Config": (".cosmos3", "Cosmos3Config"),
    "Cosmos3Model": (".cosmos3", "Cosmos3Model"),
    "MolmoAct2": (".molmoact2", "MolmoAct2"),
    "MolmoAct2Config": (".molmoact2", "MolmoAct2Config"),
    "MolmoAct2Model": (".molmoact2", "MolmoAct2Model"),
    "Pi05": (".pi05", "Pi05"),
    "Pi05Config": (".pi05", "Pi05Config"),
    "Pi05Model": (".pi05", "Pi05Model"),
    "Rldx1": (".rldx1", "Rldx1"),
    "Rldx1Config": (".rldx1", "Rldx1Config"),
    "Rldx1Model": (".rldx1", "Rldx1Model"),
    "SmolVLA": (".smolvla", "SmolVLA"),
    "SmolVLAConfig": (".smolvla", "SmolVLAConfig"),
    "SmolVLAModel": (".smolvla", "SmolVLAModel"),
    "XR0": (".xr0", "XR0"),
    "XR0Config": (".xr0", "XR0Config"),
    "XR0Model": (".xr0", "XR0Model"),
}

_PHYSICALAI_POLICY_CLASSES = {
    "act": "ACT",
    "cosmos3": "Cosmos3",
    "molmoact2": "MolmoAct2",
    "pi05": "Pi05",
    "rldx1": "Rldx1",
    "smolvla": "SmolVLA",
    "xr0": "XR0",
}

__all__ = [  # noqa: RUF022  # grouped by policy family, not isort-sorted
    # ACT
    "ACT",
    "ACTConfig",
    "ACTModel",
    # Cosmos3
    "Cosmos3",
    "Cosmos3Config",
    "Cosmos3Model",
    # MolmoAct2
    "MolmoAct2",
    "MolmoAct2Config",
    "MolmoAct2Model",
    # Pi05
    "Pi05",
    "Pi05Config",
    "Pi05Model",
    # Base
    "Policy",
    # RLDX
    "Rldx1",
    "Rldx1Config",
    "Rldx1Model",
    # SmolVLA
    "SmolVLA",
    "SmolVLAConfig",
    "SmolVLAModel",
    # XR0
    "XR0",
    "XR0Config",
    "XR0Model",
    # Utils
    "get_physicalai_policy_class",
    "get_policy",
    "lerobot",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401  # PEP 562 module exports are dynamic.
    """Load a first-party policy export only when a caller requests it.

    Import errors from the requested policy module are propagated.

    Returns:
        The requested policy export.

    Raises:
        AttributeError: If the requested name is not a policy export.
    """
    try:
        module_name, attribute_name = _POLICY_EXPORTS[name]
    except KeyError:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from None

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily loaded policy exports in module introspection.

    Returns:
        Names available from this module, including lazy policy exports.
    """
    return sorted(set(globals()) | set(_POLICY_EXPORTS))


def get_policy(policy_name: str, *, source: str = "physicalai", **kwargs) -> Policy:  # noqa: ANN003
    """Factory function to create policy instances by name.

    This is a convenience function for dynamically creating policies based on a string name.
    Useful for parameterized tests, CLI tools, or configuration-driven policy selection. Only the selected
    policy is imported, so its optional dependencies are required only when that policy is requested.

    Args:
        policy_name: Name of the policy to create. Supported values depend on source:
            - physicalai: "act", "cosmos3", "molmoact2", "pi05", "rldx1", "smolvla", "xr0"
            - lerobot: "act", "diffusion", "smolvla", "pi0", "pi05", "pi0_fast", "groot", "xvla"
        source: Where the policy implementation comes from. Options:
            - "physicalai": First-party implementations (default)
            - "lerobot": LeRobot framework wrappers
        **kwargs: Additional keyword arguments passed to the policy constructor.

    Returns:
        Policy: Instance of the requested policy.

    Raises:
        ValueError: If the policy name or source is unknown.

    Examples:
        Create first-party ACT policy (default source):

            >>> from physicalai.policies import get_policy
            >>> policy = get_policy("act", learning_rate=1e-4)

        Create first-party Pi0.5 policy:

            >>> policy = get_policy("pi05", pretrained_name_or_path="lerobot/pi05_base")

        Create LeRobot ACT policy explicitly:

            >>> policy = get_policy("act", source="lerobot", optimizer_lr=1e-4)

        Create LeRobot-only policy (Diffusion):

            >>> policy = get_policy("diffusion", source="lerobot", optimizer_lr=1e-4)

        Use in parameterized tests:

            >>> @pytest.mark.parametrize(
            ...     ("policy_name", "source"),
            ...     [("act", "physicalai"), ("pi05", "physicalai"), ("diffusion", "lerobot")],
            ... )
            >>> def test_policy(policy_name, source):
            ...     policy = get_policy(policy_name, source=source)
            ...     assert policy is not None

        Dynamic source selection:

            >>> use_lerobot = True
            >>> policy = get_policy("act", source="lerobot" if use_lerobot else "physicalai")
    """
    source = source.lower()

    if source == "physicalai":
        return get_physicalai_policy_class(policy_name)(**kwargs)

    if source == "lerobot":
        return get_lerobot_policy(policy_name, **kwargs)

    msg = f"Unknown source: {source}. Supported sources: physicalai, lerobot"
    raise ValueError(msg)


def get_physicalai_policy_class(policy_name: str) -> type[Policy]:
    """Get a first-party policy class by name, importing its optional stack on demand.

    Only the selected policy's module is imported, so its model-specific optional dependencies are needed
    only when that policy is used.

    Args:
        policy_name: Name of the policy class to retrieve.

    Returns:
        Policy class corresponding to the given name.

    Raises:
        ValueError: If the policy name or source is unknown.
    """
    normalized_name = policy_name.lower()
    try:
        export_name = _PHYSICALAI_POLICY_CLASSES[normalized_name]
    except KeyError:
        supported = ", ".join(_PHYSICALAI_POLICY_CLASSES)
        msg = f"Unknown physicalai policy: {normalized_name}. Supported policies: {supported}"
        raise ValueError(msg) from None

    return cast("type[Policy]", __getattr__(export_name))
