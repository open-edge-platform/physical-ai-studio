"""Discover trainable policies in the process that will execute the training job."""

from collections.abc import Callable
from importlib import metadata

from loguru import logger
from packaging.requirements import Requirement
from pydantic import BaseModel


class AvailablePolicy(BaseModel):
    id: str
    name: str
    description: str
    min_vram: int | None


class PolicyCatalog(BaseModel):
    available: list[AvailablePolicy]
    backends: dict[str, list[str]]


# Only policies whose constructors support Studio's training job are selectable.
_STUDIO_POLICIES = {
    "act": AvailablePolicy(
        id="act",
        name="ACT",
        description="Action Chunking with Transformers, lightweight and fast to train (BSD 3-Clause license)",
        min_vram=2 * 1024**3,
    ),
    "molmoact2": AvailablePolicy(
        id="molmoact2",
        name="MolmoAct2",
        description="Vision-Language-Action model with flow-matching action generation (Apache 2.0 license)",
        min_vram=80 * 1024**3,
    ),
    "pi05": AvailablePolicy(
        id="pi05",
        name="Pi0.5",
        description="Flow-matching VLA with discrete state encoding and longer context (Gemma license)",
        min_vram=40 * 1024**3,
    ),
    "rldx1": AvailablePolicy(
        id="rldx1",
        name="RLDX-1",
        description="General-purpose robot foundation model for dexterous manipulation (RLWRLD Model License v1.0)",
        min_vram=40 * 1024**3,
    ),
    "smolvla": AvailablePolicy(
        id="smolvla",
        name="SmolVLA",
        description="Small Vision-Language-Action model based on SmolVLM2-500M (Apache 2.0 license)",
        min_vram=3 * 1024**3,
    ),
    "xr0": AvailablePolicy(
        id="xr0",
        name="XR0",
        description=(
            "Vision-Language-Action model with a Qwen3-VL backbone and flow-matching action expert (Apache 2.0 license)"
        ),
        min_vram=60 * 1024**3,
    ),
}


def missing_policy_dependencies(name: str) -> list[str]:
    """Check extras without loading deferred model dependencies."""
    missing = []
    for raw in metadata.requires("physicalai-train") or []:
        requirement = Requirement(raw)
        marker = requirement.marker
        if marker is None or not marker.evaluate({"extra": name}) or marker.evaluate({"extra": ""}):
            continue
        try:
            installed_version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            missing.append(requirement.name)
        else:
            if installed_version not in requirement.specifier:
                missing.append(f"{requirement.name}{requirement.specifier} (installed {installed_version})")
    return missing


def available_policy_classes(
    missing_dependencies: Callable[[str], list[str]] = missing_policy_dependencies,
    warning: Callable[..., None] = logger.warning,
) -> dict[str, tuple[type, list[str]]]:
    """Resolve policies and backends installed in this Python environment."""
    from physicalai.policies import get_physicalai_policy_class, list_physicalai_policies

    classes = {}
    for name in list_physicalai_policies():
        if name not in _STUDIO_POLICIES:
            continue
        if missing := missing_dependencies(name):
            warning("Policy {} is unavailable (missing optional dependencies): {}", name, ", ".join(missing))
            continue
        try:
            cls = get_physicalai_policy_class(name)
            backends = (
                [str(b) for b in cls.get_supported_export_backends()]
                if hasattr(cls, "get_supported_export_backends")
                else []
            )
            classes[name] = (cls, backends)
        except ImportError as exc:
            warning("Policy {} is unavailable: {}", name, exc)
    return classes


def policy_catalog(classes: dict[str, tuple[type, list[str]]]) -> PolicyCatalog:
    """Describe the policies and backends of this process."""
    return PolicyCatalog(
        available=[_STUDIO_POLICIES[name] for name in classes],
        backends={name: backends for name, (_, backends) in classes.items()},
    )


def declared_policy_catalog(backends: dict[str, list[str]]) -> PolicyCatalog:
    """Describe an image's declared capabilities without importing local policies."""
    return PolicyCatalog(available=[_STUDIO_POLICIES[name] for name in backends], backends=backends)
