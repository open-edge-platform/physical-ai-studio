# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action trainer policies."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import lerobot
from .base import Policy
from .lerobot import get_lerobot_policy

if TYPE_CHECKING:
    from .act import ACT, ACTConfig, ACTModel  # noqa: F401
    from .cosmos3 import Cosmos3, Cosmos3Config, Cosmos3Model  # noqa: F401
    from .molmoact2 import MolmoAct2, MolmoAct2Config, MolmoAct2Model  # noqa: F401
    from .pi05 import Pi05, Pi05Config, Pi05Model  # noqa: F401
    from .rldx1 import Rldx1, Rldx1Config, Rldx1Model  # noqa: F401
    from .smolvla import SmolVLA, SmolVLAConfig, SmolVLAModel  # noqa: F401
    from .xr0 import XR0, XR0Config, XR0Model  # noqa: F401


def _discover_policy_paths() -> dict[str, tuple[str, str]]:
    """Discover short names and classes without importing optional policy dependencies.

    Returns:
        Policy names mapped to their package directory and class name.

    Raises:
        ValueError: If a policy file does not have exactly one direct Policy subclass.
    """
    paths: dict[str, tuple[str, str]] = {}
    for path in sorted(Path(__file__).parent.glob("*/policy.py")):
        if path.parent.name in {"base", "lerobot"}:
            continue
        classes = [
            node.name
            for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body
            if isinstance(node, ast.ClassDef)
            and any(isinstance(base, ast.Name) and base.id == "Policy" for base in node.bases)
        ]
        if len(classes) != 1:
            msg = f"Expected one Policy subclass in {path}, found {len(classes)}"
            raise ValueError(msg)
        paths[path.parent.name] = (path.parent.name, classes[0])
    return paths


_POLICY_PATHS = _discover_policy_paths()
_LAZY_EXPORTS = {
    f"{class_name}{suffix}": directory
    for directory, class_name in _POLICY_PATHS.values()
    for suffix in ("", "Config", "Model")
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Load a policy export only when it is requested.

    Returns:
        The policy class, config, or model.

    Raises:
        AttributeError: If the export is unknown.
    """
    if name not in _LAZY_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(import_module(f".{_LAZY_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = [  # noqa: PLE0604  # exports are derived from the policy mapping
    "Policy",
    "get_lerobot_policy",
    "get_physicalai_policy_class",
    "get_policy",
    "lerobot",
    "list_physicalai_policies",
    *_LAZY_EXPORTS,
]


def get_policy(policy_name: str, *, source: str = "physicalai", **kwargs) -> Policy:  # noqa: ANN003
    """Factory function to create policy instances by name.

    This is a convenience function for dynamically creating policies based on a string name.
    Useful for parameterized tests, CLI tools, or configuration-driven policy selection.

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


def list_physicalai_policies() -> list[str]:
    """List first-party short names without importing their policy modules.

    Returns:
        Sorted first-party policy names.
    """
    return sorted(_POLICY_PATHS)


def get_physicalai_policy_class(policy_name: str) -> type[Policy]:
    """Get a first-party policy class by name.

    Args:
        policy_name: Name of the policy class to retrieve.

    Returns:
        Policy class corresponding to the given name.

    Raises:
        ValueError: If the policy name is unknown.
    """
    try:
        _, class_name = _POLICY_PATHS[policy_name.lower()]
    except KeyError:
        msg = f"Unknown physicalai policy: {policy_name}. Supported policies: {', '.join(sorted(_POLICY_PATHS))}"
        raise ValueError(msg) from None
    return __getattr__(class_name)
