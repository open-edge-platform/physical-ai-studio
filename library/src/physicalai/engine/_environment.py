# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Adapt Engine worker metadata to Runtime-managed isolated environments."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # GPU probes use argv lists with shell disabled.
import tempfile
import tomllib
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from physicalai import isolate

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from ._types import WorkerEnv

logger = logging.getLogger(__name__)
_MARKER = ".physicalai-engine.json"
_MARKER_VERSION = 3
_COMPONENT_RE = re.compile(r"[a-zA-Z0-9_.-]{1,128}")
_ACCELERATOR_ALIASES = {"cpu": "cpu", "cuda": "cu128", "cu128": "cu128", "xpu": "xpu"}
_PCI_DISPLAY_CLASS = 0x03
_PCI_VENDOR_INTEL = "0x8086"
_PCI_VENDOR_NVIDIA = "0x10de"


@dataclass(frozen=True)
class PreparedEnvironment:
    """A managed worker environment with Engine metadata.

    The Runtime-owned prepared environment is retained for process launch. The
    remaining fields preserve Engine's framework, environment, and distribution
    metadata for existing callers and the environment-management CLI.

    Attributes:
        framework: Engine identifier.
        name: Environment name within the framework.
        accelerator: Accelerator profile that was synchronized.
        key: Engine content key for the declaration.
        distribution: Plugin distribution that declared the environment.
        path: Environment root containing ``project`` and ``venv``.
        size_bytes: On-disk size when measured.
    """

    framework: str
    name: str
    accelerator: str
    key: str
    distribution: str
    path: Path
    size_bytes: int = 0
    _isolate: isolate.PreparedEnvironment | None = field(default=None, repr=False, compare=False)

    @property
    def python(self) -> Path:
        """The environment's Python executable."""
        return _python_path(self.path / "venv")

    @property
    def project(self) -> Path:
        """The materialized uv project the environment was built from."""
        return self.path / "project"

    @property
    def isolate(self) -> isolate.PreparedEnvironment:
        """The Runtime environment used to launch managed worker processes.

        Raises:
            RuntimeError: If the Engine environment came from a cache listing.
        """
        if self._isolate is None:
            msg = "A listed Engine environment must be prepared before it can launch a worker."
            raise RuntimeError(msg)
        return self._isolate


def engine_cache_root() -> Path:
    """Return the cache root used by Runtime's isolate provider.

    ``PHYSICALAI_ENGINE_CACHE`` remains available for existing Engine
    deployments. Otherwise, Engine and Runtime share the isolate cache.
    """
    configured = os.environ.get("PHYSICALAI_ENGINE_CACHE") or os.environ.get("PHYSICALAI_ISOLATE_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return (cache_home / "physicalai" / "isolate").resolve()


def resolve_accelerator(accelerator: str, *, allowed: tuple[str, ...] | None = None) -> str:
    """Resolve ``auto`` or a device name to one supported worker uv extra.

    Automatic selection checks both the host torch runtime and physical GPU
    hardware, so a CPU-only host build does not hide an available GPU.

    Returns:
        Canonical uv extra name: ``cpu``, ``cu128``, or ``xpu``.

    Raises:
        ValueError: If the requested extra is unknown, unsupported, or conflicts with detected GPU hardware.
    """
    normalized_allowed = None if allowed is None else {_canonical_accelerator(value) for value in allowed}
    if accelerator.strip().lower() == "auto":
        return _automatic_accelerator(normalized_allowed)

    selected = _canonical_accelerator(accelerator)
    if normalized_allowed is not None and selected not in normalized_allowed:
        msg = f"Accelerator {accelerator!r} is not supported by this engine environment."
        raise ValueError(msg)
    return selected


def _automatic_accelerator(allowed: set[str] | None) -> str:
    """Pick the best profile for the host among the environment's profiles.

    Returns:
        Canonical profile name.

    Raises:
        ValueError: If detected GPU hardware has no matching profile.
    """
    if allowed is not None and len(allowed) == 1:
        # A single-profile environment (for example, a CUDA-only upstream lock) has nothing to choose.
        return next(iter(allowed))
    runtime = _torch_accelerators()
    hardware = _hardware_accelerators()
    order = (("xpu", "xpu"), ("cuda", "cu128"))
    candidates = [extra for device, extra in order if runtime.get(device, False)]
    if not candidates:
        candidates = [extra for device, extra in order if device in hardware]
    if allowed is None:
        return candidates[0] if candidates else "cpu"
    for candidate in candidates:
        if candidate in allowed:
            return candidate
    if candidates:
        detected = ", ".join("XPU" if candidate == "xpu" else "CUDA" for candidate in candidates)
        supported = ", ".join(sorted(allowed))
        msg = f"Detected {detected} hardware, but this engine environment supports only {supported}."
        raise ValueError(msg)
    for fallback in ("cpu", "xpu", "cu128"):
        if fallback in allowed:
            return fallback
    msg = "The worker environment does not declare a supported accelerator extra."
    raise ValueError(msg)


def _canonical_accelerator(accelerator: str) -> str:
    try:
        return _ACCELERATOR_ALIASES[accelerator.lower()]
    except KeyError:
        supported = ", ".join((*_ACCELERATOR_ALIASES, "auto"))
        msg = f"Unsupported accelerator {accelerator!r}; choose one of {supported}."
        raise ValueError(msg) from None


def _torch_accelerators() -> dict[str, bool]:
    try:
        import torch  # ruff: ignore[import-outside-top-level]  # Torch is optional when listing engines.
    except (ImportError, OSError, RuntimeError):
        return {"xpu": False, "cuda": False}

    def available(name: str) -> bool:
        try:
            backend = getattr(torch, name, None)
            is_available = getattr(backend, "is_available", None)
            return bool(backend is not None and callable(is_available) and is_available())
        except (AttributeError, OSError, RuntimeError, TypeError):
            return False

    return {"xpu": available("xpu"), "cuda": available("cuda")}


def _hardware_accelerators(devices: Path = Path("/sys/bus/pci/devices")) -> set[str]:
    detected = _pci_gpu_hardware(devices)
    if _command_reports_gpu("xpu-smi", ("discovery",), r"\bGPU\s*#?\s*\d+\b"):
        detected.add("xpu")
    if _command_reports_gpu("nvidia-smi", ("-L",), r"\bGPU\s+\d+\s*:"):
        detected.add("cuda")
    return detected


def _pci_gpu_hardware(devices: Path) -> set[str]:
    detected: set[str] = set()
    if not devices.is_dir():
        return detected
    try:
        entries = tuple(devices.iterdir())
    except OSError:
        return detected
    for device in entries:
        try:
            vendor = (device / "vendor").read_text(encoding="ascii").strip().lower()
            class_code = int((device / "class").read_text(encoding="ascii").strip(), 16)
        except (OSError, ValueError):
            continue
        if class_code >> 16 != _PCI_DISPLAY_CLASS:
            continue
        if vendor == _PCI_VENDOR_INTEL:
            detected.add("xpu")
        elif vendor == _PCI_VENDOR_NVIDIA:
            detected.add("cuda")
    return detected


def _command_reports_gpu(command: str, arguments: tuple[str, ...], pattern: str) -> bool:
    executable = shutil.which(command)
    if executable is None:
        return False
    try:
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [executable, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and re.search(pattern, result.stdout + result.stderr, re.IGNORECASE) is not None


def to_isolate_environment(env: WorkerEnv, accelerator: str = "auto") -> isolate.Environment:
    """Map an Engine worker declaration to Runtime's source and sync profile.

    Returns:
        The Runtime environment with the selected Python version and uv extras.
    """
    selected = resolve_accelerator(accelerator, allowed=env.accelerators)
    profile = isolate.SyncProfile(python=env.python, extras=env.extras_for(selected))
    return isolate.Environment(source=env.source, profile=profile)


def lock_bytes(env: WorkerEnv, project: Path | None = None) -> bytes:
    """Read a worker lock from a prepared project or from a bundled source.

    Args:
        env: Worker environment.
        project: Materialized project directory. Required for sources whose
            lock exists only after materialization, such as Git projects.

    Returns:
        The raw ``uv.lock`` bytes.

    Raises:
        FileNotFoundError: If the project does not include a lock.
        ValueError: If the source has no lock before preparation and no project was given.
    """
    if project is not None:
        lock = project / "uv.lock"
        if not lock.is_file():
            msg = f"Worker project for {env.framework!r} does not include uv.lock."
            raise FileNotFoundError(msg)
        return lock.read_bytes()
    read_lock = getattr(env.source, "lock", None)
    lock_data = read_lock() if callable(read_lock) else None
    if lock_data is None:
        msg = f"The {env.framework!r} {env.name!r} environment's lock is available only after preparation."
        raise ValueError(msg)
    return lock_data


def lock_hash(env: WorkerEnv, project: Path | None = None) -> str:
    """Return the SHA-256 digest of a worker lock."""
    return hashlib.sha256(lock_bytes(env, project)).hexdigest()


def locked_version(env: WorkerEnv, project: Path | None = None) -> Version:
    """Read the exact worker package version from ``uv.lock``.

    Returns:
        The exact locked package version.

    Raises:
        ValueError: If the lock is invalid or does not pin exactly one matching package.
    """
    try:
        lock = tomllib.loads(lock_bytes(env, project).decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        msg = f"Worker lock for {env.framework!r} is not valid TOML."
        raise ValueError(msg) from exc

    package_name = _normalize_distribution_name(env.package)
    versions = {
        Version(package["version"])
        for package in lock.get("package", [])
        if isinstance(package, dict)
        and isinstance(package.get("name"), str)
        and _normalize_distribution_name(package["name"]) == package_name
        and isinstance(package.get("version"), str)
    }
    if len(versions) != 1:
        msg = (
            f"Expected one locked version of {env.package!r} for {env.framework!r}, found {sorted(map(str, versions))}."
        )
        raise ValueError(msg)
    return next(iter(versions))


def supported_versions(env: WorkerEnv) -> SpecifierSet | None:
    """Read the package range declared by the plugin, if it declares one.

    Plugins that depend on a PyPI framework declare their range. Plugins for
    frameworks pinned by Git commit declare none; their environments are
    identified by content key and exact lock instead.

    Returns:
        The declared specifier, or ``None`` when the plugin declares no requirement.

    Raises:
        LookupError: If the plugin declares more than one unmarked requirement on the package.
    """
    package_name = _normalize_distribution_name(env.package)
    requirements = []
    for raw_requirement in metadata.requires(env.distribution) or []:
        requirement = Requirement(raw_requirement)
        if _normalize_distribution_name(requirement.name) != package_name:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        requirements.append(requirement.specifier)

    if not requirements:
        return None
    if len(requirements) != 1:
        msg = (
            f"Expected one unmarked {env.package!r} requirement in distribution "
            f"{env.distribution!r}, found {len(requirements)}."
        )
        raise LookupError(msg)
    return SpecifierSet(str(requirements[0]))


def prepare(env: WorkerEnv, accelerator: str = "auto", *, offline: bool = False) -> Path:
    """Create or synchronize the worker environment and return its interpreter.

    Returns:
        Path to the prepared worker Python executable.
    """
    return prepare_environment(env, accelerator, offline=offline).python


def prepare_environment(env: WorkerEnv, accelerator: str = "auto", *, offline: bool = False) -> PreparedEnvironment:
    """Create or reuse the immutable worker environment for *env*.

    Runtime owns project materialization, cache identity, and uv synchronization.
    Engine resolves accelerator metadata and adds its framework-specific marker.

    Returns:
        The prepared environment.
    """
    selected = resolve_accelerator(accelerator, allowed=env.accelerators)
    for value, label in ((env.framework, "framework"), (env.name, "name"), (env.distribution, "distribution")):
        _validate_component(value, label)
    environment = to_isolate_environment(env, selected)
    prepared = isolate.prepare(
        environment,
        provider=isolate.UvProvider(cache_dir=engine_cache_root()),
        offline=offline,
    )
    key = env.key()
    _write_marker(
        prepared.path,
        {
            "format_version": _MARKER_VERSION,
            "framework": env.framework,
            "name": env.name,
            "accelerator": selected,
            "key": key,
            "distribution": env.distribution,
            "source": env.source.provenance(),
            "isolate_key": prepared.key,
        },
    )
    return PreparedEnvironment(
        env.framework,
        env.name,
        selected,
        key,
        env.distribution,
        prepared.path,
        _isolate=prepared,
    )


def list_prepared() -> list[PreparedEnvironment]:
    """List Engine-managed environments and their on-disk size.

    Returns:
        Environments sorted by framework, name, accelerator, and path.
    """
    prepared: list[PreparedEnvironment] = []
    for root, marker in _markers():
        isolate_marker = root / ".physicalai-isolate.json"
        if isolate_marker.is_symlink() or not isolate_marker.is_file():
            continue
        if any((root / name).is_symlink() for name in ("project", "venv")):
            continue
        if not (root / "project").is_dir() or not (root / "venv").is_dir():
            continue
        if not _python_path(root / "venv").is_file():
            continue
        prepared.append(
            PreparedEnvironment(
                framework=_marker_string(marker, "framework"),
                name=_marker_string(marker, "name"),
                accelerator=_marker_string(marker, "accelerator"),
                key=_marker_string(marker, "key"),
                distribution=_marker_string(marker, "distribution"),
                path=root,
                size_bytes=_directory_size(root),
            ),
        )
    return sorted(prepared, key=lambda item: (item.framework, item.name, item.accelerator, str(item.path)))


def prune_prepared(current: Iterable[WorkerEnv]) -> list[Path]:
    """Remove Engine-managed cache entries no current engine or enablement declares.

    Runtime-owned cache entries without an Engine marker are left untouched.

    Args:
        current: Every environment declared by installed engines and their enablements.

    Returns:
        Paths of removed worker environments.
    """
    cache = engine_cache_root()
    if not cache.is_dir():
        return []

    current_keys = {(env.framework, env.name): env.key() for env in current}
    removed: list[Path] = []
    for root in cache.iterdir():
        if root.is_symlink() or not root.is_dir():
            continue
        marker_path = root / _MARKER
        if marker_path.is_symlink() or not marker_path.is_file():
            continue
        try:
            marker = _read_marker(root)
            current_format = marker.get("format_version") == _MARKER_VERSION
            if current_format:
                identity = (_marker_string(marker, "framework"), _marker_string(marker, "name"))
                key = _marker_string(marker, "key")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            stale = True
        else:
            stale = current_keys.get(identity) != key if current_format else True
        if stale:
            shutil.rmtree(root)
            removed.append(root)
    return sorted(removed)


def prune() -> list[Path]:
    """Prune cached environments that installed engines and enablements no longer declare.

    Returns:
        Paths of removed worker environments.
    """
    from . import Engine  # ruff: ignore[import-outside-top-level]  # Discovery imports this module during package initialization.

    current: list[WorkerEnv] = []
    for name in _prepared_frameworks():
        try:
            current.extend(Engine.get(name).environments())
        except ValueError:
            continue
    return prune_prepared(current)


def _prepared_frameworks() -> set[str]:
    return {_marker_string(marker, "framework") for _, marker in _markers()}


def _markers() -> Iterator[tuple[Path, dict[str, Any]]]:
    cache = engine_cache_root()
    if not cache.is_dir():
        return
    for root in sorted(cache.iterdir()):
        if root.is_symlink() or not root.is_dir() or not (root / _MARKER).is_file():
            continue
        try:
            marker = _read_marker(root)
            for key in ("framework", "name", "accelerator", "key", "distribution"):
                _marker_string(marker, key)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            logger.warning(
                "Skipping worker environment %s with an invalid Engine marker; `physicalai engine prune` removes it.",
                root,
            )
            continue
        if marker.get("format_version") != _MARKER_VERSION:
            logger.warning(
                "Skipping worker environment %s with an outdated Engine marker; `physicalai engine prune` removes it.",
                root,
            )
            continue
        yield root, marker


def interpreter_for(
    env: WorkerEnv,
    *,
    accelerator: str = "auto",
    interpreter: str | Path | None = None,
) -> Path:
    """Choose a supplied interpreter or provision the locked environment.

    Returns:
        The Python executable to run.
    """
    if interpreter is not None:
        return Path(interpreter).expanduser()
    return prepare(env, accelerator)


def scripts_dir(interpreter: str | Path) -> Path:
    """Return the scripts directory adjacent to a Python interpreter.

    Returns:
        Directory containing scripts installed beside the interpreter.
    """
    return Path(interpreter).expanduser().parent


def _write_marker(root: Path, marker: dict[str, Any]) -> None:
    marker_path = root / _MARKER
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as file:
        json.dump(marker, file, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(marker_path)


def _read_marker(root: Path) -> dict[str, Any]:
    marker_path = root / _MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        msg = f"Worker environment marker not found: {marker_path}"
        raise FileNotFoundError(msg)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        msg = f"Worker environment marker is not an object: {marker_path}"
        raise TypeError(msg)
    return marker


def _marker_string(marker: object, key: str) -> str:
    if not isinstance(marker, dict) or not isinstance(marker.get(key), str):
        msg = f"Worker environment marker has an invalid {key!r} field."
        raise TypeError(msg)
    return marker[key]


def _directory_size(root: Path) -> int:
    size = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        size += path.stat().st_size
    return size


def _validate_component(value: str, label: str) -> None:
    if _COMPONENT_RE.fullmatch(value) is None:
        msg = f"Invalid worker environment {label}: {value!r}."
        raise ValueError(msg)


def _python_path(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


__all__ = [
    "PreparedEnvironment",
    "engine_cache_root",
    "interpreter_for",
    "list_prepared",
    "lock_hash",
    "locked_version",
    "prepare",
    "prepare_environment",
    "prune",
    "prune_prepared",
    "resolve_accelerator",
    "scripts_dir",
    "supported_versions",
    "to_isolate_environment",
]
