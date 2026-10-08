# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action trainer policies."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from typing import Any

from . import lerobot
from .base import Policy
from .lerobot import get_lerobot_policy


def _discover_policy_paths() -> dict[str, tuple[str, str]]:
    """Discover names without importing optional policy dependencies.

    Returns:
        Policy short names and their module/class locations.

    Raises:
        ValueError: If a policy has incomplete or duplicate metadata.
    """
    paths: dict[str, tuple[str, str]] = {}
    for path in sorted(Path(__file__).parent.glob("*/policy.py")):
        if path.parent.name in {"base", "lerobot"}:
            continue
        constants = {
            target.id: node.value.value
            for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id in {"POLICY_NAME", "POLICY_CLASS"}
        }
        if set(constants) != {"POLICY_NAME", "POLICY_CLASS"}:
            msg = f"Incomplete policy metadata in {path}"
            raise ValueError(msg)
        name = constants["POLICY_NAME"]
        if name in paths:
            msg = f"Duplicate policy name: {name}"
            raise ValueError(msg)
        paths[name] = (path.parent.name, constants["POLICY_CLASS"])
    return paths


_POLICY_PATHS = _discover_policy_paths()


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Keep root-level exports available without importing every policy.

    Returns:
        The requested policy class, configuration, or model.

    Raises:
        AttributeError: If the export is unknown.
    """
    for directory, class_name in _POLICY_PATHS.values():
        if name in {class_name, f"{class_name}Config", f"{class_name}Model"}:
            value = getattr(import_module(f".{directory}", __name__), name)
            globals()[name] = value
            return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Show lazy exports in package introspection.

    Returns:
        All available package attributes, including policy exports.
    """
    return sorted(set(globals()) | set(__all__))


__all__ = ["Policy", "get_physicalai_policy_class", "get_policy", "lerobot", "get_lerobot_policy"] + [
    export
    for _, class_name in _POLICY_PATHS.values()
    for export in (class_name, f"{class_name}Config", f"{class_name}Model")
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
        directory, class_name = _POLICY_PATHS[policy_name.lower()]
    except KeyError:
        msg = f"Unknown physicalai policy: {policy_name}. Supported policies: {', '.join(sorted(_POLICY_PATHS))}"
        raise ValueError(msg) from None
    return getattr(import_module(f".{directory}.policy", __name__), class_name)
