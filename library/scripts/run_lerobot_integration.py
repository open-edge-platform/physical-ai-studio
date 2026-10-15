# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Select the host PyTorch extra and run LeRobot integration tests."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Runs uv and pytest with fixed argv and shell=False.
import sys
from pathlib import Path

from physicalai.engine._integration import (  # noqa: PLC2701  # Shares internal hardware selection with the gated tests.
    detect_gpu_hardware,
    probe_torch_devices,
    require_torch_device,
    select_integration_target,
)

_LIBRARY_ROOT = Path(__file__).resolve().parents[1]
_DEVICE_ENV = "PHYSICALAI_LEROBOT_DEVICE"
_RUN_ENV = "PHYSICALAI_RUN_LEROBOT_INTEGRATION"
_DEFAULT_TESTS = [
    "tests/integration/test_engine_lerobot_hf_training.py::test_native_lerobot_training_matches_engine_on_pinned_pusht",
    "-q",
    "-s",
]


def _environment_path() -> Path:
    configured = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if configured:
        return Path(configured).expanduser().resolve()
    if Path(sys.prefix) != Path(sys.base_prefix):
        return Path(sys.prefix).resolve()
    active = os.environ.get("VIRTUAL_ENV")
    if active:
        return Path(active).expanduser().resolve()
    return _LIBRARY_ROOT / ".venv"


def _environment_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu"),
        default=os.environ.get(_DEVICE_ENV, "auto"),
        help="Override automatic XPU/CUDA/CPU selection.",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Pytest node IDs and arguments.")
    options = parser.parse_args(argv)
    if options.pytest_args and options.pytest_args[0] == "--":
        options.pytest_args = options.pytest_args[1:]
    return options


def main(argv: list[str] | None = None) -> int:
    """Sync the selected uv extra, verify host PyTorch, then run pytest.

    Returns:
        The uv or pytest exit status.
    """
    options = _parse_arguments(argv)
    environment = _environment_path()
    environment_has_python = (environment / "pyvenv.cfg").is_file()
    probe_python = _environment_python(environment) if environment_has_python else Path(sys.executable)
    if not probe_python.is_file():
        print(f"Host Python does not exist: {probe_python}", file=sys.stderr)  # noqa: T201
        return 2

    hardware = detect_gpu_hardware()
    initial_torch = probe_torch_devices(probe_python)
    target = select_integration_target(
        options.device,
        detected_hardware=hardware,
        torch_devices=initial_torch.as_mapping(),
    )
    uv = shutil.which("uv")
    if uv is None:
        print("uv is required; install uv 0.10.x and retry.", file=sys.stderr)  # noqa: T201
        return 2

    sync_command = [
        uv,
        "sync",
        "--locked",
        "--project",
        str(_LIBRARY_ROOT),
        "--extra",
        target.uv_extra,
        "--extra",
        "tests",
        "--extra",
        "lerobot",
    ]
    if not environment_has_python:
        sync_command.extend(("--python", str(probe_python)))
    command_env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)}
    try:
        sync_result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # Fixed argv, shell=False.
            sync_command,
            cwd=_LIBRARY_ROOT,
            env=command_env,
            check=False,
            shell=False,
        )
    except OSError as exc:
        print(f"Could not start uv: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    if sync_result.returncode != 0:
        return sync_result.returncode

    host_python = _environment_python(environment)
    prepared_torch = probe_torch_devices(host_python)
    try:
        require_torch_device(
            target,
            torch_devices=prepared_torch,
            detected_hardware=hardware,
            interpreter=host_python,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201
        return 2

    print(  # noqa: T201
        f"Running LeRobot integration tests on {target.device} (uv extra {target.uv_extra}) with {host_python}.",
        flush=True,
    )
    command_env[_DEVICE_ENV] = target.device
    command_env[_RUN_ENV] = "1"
    pytest_args = options.pytest_args or _DEFAULT_TESTS
    return subprocess.call(  # ruff: ignore[subprocess-without-shell-equals-true]  # Run the selected pytest argv without a shell.
        [str(host_python), "-m", "pytest", *pytest_args],
        cwd=_LIBRARY_ROOT,
        env=command_env,
        shell=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
