# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Provision and inspect locked uv worker environments."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Invokes pinned uv with shell=False and a fixed subcommand.
import tempfile
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.requirements import Requirement
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Mapping
    from importlib.resources.abc import Traversable

    from packaging.specifiers import SpecifierSet

    from ._types import WorkerEnv

logger = logging.getLogger(__name__)
_MARKER = ".physicalai-engine.json"
_MARKER_VERSION = 1
_COMPONENT_RE = re.compile(r"[a-zA-Z0-9_.-]{1,128}")
_ACCELERATOR_ALIASES = {"cpu": "cpu", "cuda": "cu128", "cu128": "cu128", "xpu": "xpu"}
_PCI_DISPLAY_CLASS = 0x03
_PCI_VENDOR_INTEL = "0x8086"
_PCI_VENDOR_NVIDIA = "0x10de"


@dataclass(frozen=True)
class PreparedEnvironment:
    """A managed worker environment reported by ``pai engine list``."""

    framework: str
    accelerator: str
    lock_hash: str
    distribution: str
    path: Path
    size_bytes: int


def engine_cache_root() -> Path:
    """Return the user-owned cache root for managed worker environments."""
    configured = os.environ.get("PHYSICALAI_ENGINE_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return (cache_home / "physicalai" / "engines").resolve()


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
        runtime = _torch_accelerators()
        hardware = _hardware_accelerators()
        order = (("xpu", "xpu"), ("cuda", "cu128"))
        candidates = [extra for device, extra in order if runtime.get(device, False)]
        if not candidates:
            candidates = [extra for device, extra in order if device in hardware]
        if normalized_allowed is None:
            if candidates:
                return candidates[0]
            return "cpu"
        for candidate in candidates:
            if candidate in normalized_allowed:
                return candidate
        if candidates:
            detected = ", ".join("XPU" if candidate == "xpu" else "CUDA" for candidate in candidates)
            supported = ", ".join(sorted(normalized_allowed))
            msg = f"Detected {detected} hardware, but this engine environment supports only {supported}."
            raise ValueError(msg)
        if "cpu" in normalized_allowed:
            return "cpu"
        for fallback in ("xpu", "cu128"):
            if fallback in normalized_allowed:
                return fallback
        msg = "The worker environment does not declare a supported accelerator extra."
        raise ValueError(msg)

    selected = _canonical_accelerator(accelerator)
    if normalized_allowed is not None and selected not in normalized_allowed:
        msg = f"Accelerator {accelerator!r} is not supported by this engine environment."
        raise ValueError(msg)
    return selected


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


def _hardware_accelerators(pci_devices: Path = Path("/sys/bus/pci/devices")) -> set[str]:
    detected = _pci_gpu_hardware(pci_devices)
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


def uv_bin() -> str:
    """Return the executable bundled by the pinned ``uv`` Python package.

    Raises:
        RuntimeError: If the ``uv`` package is missing from the host environment.
    """
    try:
        from uv import find_uv_bin  # ruff: ignore[import-outside-top-level]  # Loaded only when an env is prepared.
    except ImportError as exc:
        msg = "Managed engine environments require the pinned 'uv' package; install physicalai-train first."
        raise RuntimeError(msg) from exc
    return str(find_uv_bin())


def lock_bytes(env: WorkerEnv) -> bytes:
    """Read the worker lock from a filesystem path or package resource.

    Returns:
        The raw ``uv.lock`` bytes.

    Raises:
        FileNotFoundError: If the worker project does not include a lock file.
    """
    lock = _join_resource(env.project, "uv.lock")
    if not lock.is_file():
        msg = f"Worker environment for {env.framework!r} does not include uv.lock."
        raise FileNotFoundError(msg)
    return lock.read_bytes()


def lock_hash(env: WorkerEnv) -> str:
    """Return the SHA-256 digest of a worker lock."""
    return hashlib.sha256(lock_bytes(env)).hexdigest()


def locked_version(env: WorkerEnv) -> Version:
    """Read the exact worker package version from ``uv.lock``.

    Returns:
        The exact locked package version.

    Raises:
        ValueError: If the lock is invalid or does not pin exactly one matching package.
    """
    try:
        lock = tomllib.loads(lock_bytes(env).decode("utf-8"))
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


def supported_versions(env: WorkerEnv) -> SpecifierSet:
    """Read the supported package range from the plugin's installed metadata.

    Returns:
        The package version specifier declared by the plugin.

    Raises:
        LookupError: If the plugin distribution does not declare one unmarked requirement.
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

    if len(requirements) != 1:
        msg = (
            f"Expected one unmarked {env.package!r} requirement in distribution "
            f"{env.distribution!r}, found {len(requirements)}."
        )
        raise LookupError(msg)
    return requirements[0]


def prepare(env: WorkerEnv, accelerator: str = "auto", *, offline: bool = False) -> Path:
    """Create or synchronize the immutable worker environment for *env*.

    The uv project is copied outside the caller's working tree so a host
    project's ``[tool.uv]`` configuration cannot affect resolution. ``--frozen``
    makes the plugin's lock the only dependency resolution source.

    Returns:
        Path to the prepared worker Python executable.

    Raises:
        ValueError: If the worker framework or accelerator is invalid.
    """
    selected = resolve_accelerator(accelerator, allowed=env.accelerators)
    _validate_component(env.framework, "framework")
    _validate_component(env.distribution, "distribution")
    digest = lock_hash(env)
    root = engine_cache_root() / f"{env.framework}-{selected}-{digest[:12]}"
    project = root / "project"
    venv = root / "venv"
    if root.is_symlink():
        msg = f"Worker environment cache path cannot be a symbolic link: {root}"
        raise ValueError(msg)
    root.mkdir(parents=True, exist_ok=True)
    _copy_project_once(env.project, project)
    if venv.is_symlink():
        msg = f"Worker virtual environment path cannot be a symbolic link: {venv}"
        raise ValueError(msg)

    command = [uv_bin(), "sync", "--frozen", "--project", str(project), "--extra", selected]
    if offline:
        command.append("--offline")
    child_env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}
    logger.info("Preparing %s worker environment (%s).", env.framework, selected)
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # Only fixed uv subcommands and validated project paths are passed.
        command,
        cwd=project,
        env=child_env,
        check=True,
        shell=False,
    )
    _write_marker(
        root,
        {
            "format_version": _MARKER_VERSION,
            "framework": env.framework,
            "accelerator": selected,
            "lock_hash": digest,
            "distribution": env.distribution,
        },
    )
    return _python_path(venv)


def list_prepared() -> list[PreparedEnvironment]:
    """List valid managed environments and their on-disk size.

    Returns:
        Environments sorted by framework, accelerator, and path.
    """
    cache = engine_cache_root()
    if not cache.is_dir():
        return []

    prepared: list[PreparedEnvironment] = []
    for root in cache.iterdir():
        if root.is_symlink() or not root.is_dir():
            continue
        try:
            marker = _read_marker(root)
            framework = _marker_string(marker, "framework")
            accelerator = _marker_string(marker, "accelerator")
            digest = _marker_string(marker, "lock_hash")
            distribution = _marker_string(marker, "distribution")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Skipping invalid worker environment marker at %s.", root / _MARKER)
            continue
        prepared.append(
            PreparedEnvironment(
                framework=framework,
                accelerator=accelerator,
                lock_hash=digest,
                distribution=distribution,
                path=root,
                size_bytes=_directory_size(root),
            ),
        )
    return sorted(prepared, key=lambda item: (item.framework, item.accelerator, str(item.path)))


def prune_prepared(current: Mapping[str, WorkerEnv]) -> list[Path]:
    """Remove managed environments whose engine or lock is no longer current.

    Returns:
        Paths of removed worker environments.
    """
    cache = engine_cache_root()
    if not cache.is_dir():
        return []

    current_hashes = {name: lock_hash(env) for name, env in current.items()}
    removed: list[Path] = []
    for root in cache.iterdir():
        if root.is_symlink() or not root.is_dir():
            continue
        marker_path = root / _MARKER
        if marker_path.is_symlink() or not marker_path.is_file():
            continue
        try:
            marker = _read_marker(root)
            framework = _marker_string(marker, "framework")
            digest = _marker_string(marker, "lock_hash")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            stale = True
        else:
            stale = current_hashes.get(framework) != digest
        if stale:
            shutil.rmtree(root)
            removed.append(root)
    return sorted(removed)


def prune() -> list[Path]:
    """Prune cached environments whose installed engine or lock is outdated.

    Returns:
        Paths of removed worker environments.
    """
    from . import Engine  # ruff: ignore[import-outside-top-level]  # Discovery imports this module during package initialization.

    current: dict[str, WorkerEnv] = {}
    for name in _prepared_frameworks():
        try:
            engine_env = Engine.get(name).env
        except ValueError:
            continue
        if engine_env is not None:
            current[name] = engine_env
    return prune_prepared(current)


def _prepared_frameworks() -> set[str]:
    cache = engine_cache_root()
    if not cache.is_dir():
        return set()

    frameworks: set[str] = set()
    for root in cache.iterdir():
        if root.is_symlink() or not root.is_dir():
            continue
        try:
            framework = _marker_string(_read_marker(root), "framework")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        frameworks.add(framework)
    return frameworks


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


def _python_path(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _copy_project_once(source: Traversable, destination: Path) -> None:
    if destination.is_symlink():
        msg = f"Worker project cache path cannot be a symbolic link: {destination}"
        raise ValueError(msg)
    if destination.is_dir():
        return
    if destination.exists():
        msg = f"Worker project cache path is not a directory: {destination}"
        raise ValueError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        _copy_resource_tree(source, staged)
        try:
            staged.rename(destination)
        except FileExistsError:
            if not destination.is_dir() or destination.is_symlink():
                raise


def _copy_resource_tree(source: Traversable, destination: Path) -> None:
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_resource_tree(child, destination / child.name)
        return
    if not source.is_file():
        msg = f"Worker project resource is not a file or directory: {source}"
        raise FileNotFoundError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _join_resource(source: Traversable, child: str) -> Traversable:
    return source.joinpath(child)


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
    version = marker.get("format_version")
    if isinstance(version, bool) or version != _MARKER_VERSION:
        msg = f"Unsupported worker environment marker version {version!r}."
        raise ValueError(msg)
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
    "prune",
    "prune_prepared",
    "resolve_accelerator",
    "scripts_dir",
    "supported_versions",
    "uv_bin",
]
