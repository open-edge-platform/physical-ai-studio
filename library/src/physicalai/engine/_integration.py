# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hardware selection and readiness checks for native LeRobot test runs."""

from __future__ import annotations

import json
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Runs fixed local hardware probes.
from dataclasses import dataclass
from pathlib import Path

_PCI_DISPLAY_CLASS = 0x03


@dataclass(frozen=True)
class IntegrationTarget:
    """Selected test device and its matching physicalai-train uv extra."""

    device: str
    uv_extra: str


@dataclass(frozen=True)
class TorchDeviceAvailability:
    """GPU availability reported by a selected Python interpreter."""

    cuda: bool
    xpu: bool
    error: str | None = None

    def as_mapping(self) -> dict[str, bool]:
        """Return backend availability keyed by ``cuda`` and ``xpu``."""
        return {"cuda": self.cuda, "xpu": self.xpu}


def select_integration_target(
    requested: str,
    *,
    detected_hardware: set[str] | frozenset[str],
    torch_devices: dict[str, bool],
) -> IntegrationTarget:
    """Select XPU, CUDA, or CPU and map it to the corresponding uv extra.

    Usable XPU wins over usable CUDA. If the selected interpreter cannot use a
    detected GPU, this still selects that GPU so the readiness check can fail
    clearly instead of silently falling back to CPU.

    Returns:
        A target containing the device name and uv extra.

    Raises:
        ValueError: If *requested* is not ``auto``, ``cpu``, ``cuda``, or ``xpu``.
    """
    normalized = requested.strip().lower()
    extras = {"cpu": "cpu", "cuda": "cu128", "xpu": "xpu"}
    if normalized != "auto":
        try:
            return IntegrationTarget(device=normalized, uv_extra=extras[normalized])
        except KeyError:
            msg = f"Unsupported device {requested!r}; choose auto, cpu, cuda, or xpu."
            raise ValueError(msg) from None

    for device in ("xpu", "cuda"):
        if torch_devices.get(device):
            return IntegrationTarget(device=device, uv_extra=extras[device])
    for device in ("xpu", "cuda"):
        if device in detected_hardware:
            return IntegrationTarget(device=device, uv_extra=extras[device])
    return IntegrationTarget(device="cpu", uv_extra="cpu")


def detect_gpu_hardware(pci_devices: Path = Path("/sys/bus/pci/devices")) -> frozenset[str]:
    """Detect Intel/NVIDIA display hardware independently of the host torch build.

    Returns:
        Detected device types: ``xpu`` and/or ``cuda``.
    """
    detected = _pci_gpu_hardware(pci_devices)
    if _command_reports_gpu("xpu-smi", ("discovery",), r"\bGPU\s*#?\s*\d+\b"):
        detected.add("xpu")
    if _command_reports_gpu("nvidia-smi", ("-L",), r"\bGPU\s+\d+\s*:"):
        detected.add("cuda")
    return frozenset(detected)


def probe_torch_devices(interpreter: str | Path) -> TorchDeviceAvailability:
    """Query CUDA/XPU usability in *interpreter* without importing torch here.

    Returns:
        Availability and any import/probe error reported by the selected interpreter.
    """
    probe = """\
import json
try:
    import torch
except Exception as exc:
    print(json.dumps({'cuda': False, 'xpu': False, 'error': str(exc)}))
else:
    def available(name):
        try:
            backend = getattr(torch, name, None)
            check = getattr(backend, 'is_available', None)
            return bool(backend is not None and callable(check) and check())
        except Exception:
            return False

    print(json.dumps({'cuda': available('cuda'), 'xpu': available('xpu')}))
"""
    try:
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [str(interpreter), "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return TorchDeviceAvailability(cuda=False, xpu=False, error=str(exc))
    if result.returncode != 0:
        return TorchDeviceAvailability(cuda=False, xpu=False, error=result.stderr.strip())
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return TorchDeviceAvailability(cuda=False, xpu=False, error=str(exc))
    if not isinstance(data, dict):
        return TorchDeviceAvailability(cuda=False, xpu=False, error="PyTorch probe returned non-object output.")
    return TorchDeviceAvailability(
        cuda=bool(data.get("cuda", False)),
        xpu=bool(data.get("xpu", False)),
        error=data.get("error") if isinstance(data.get("error"), str) else None,
    )


def require_torch_device(
    target: IntegrationTarget,
    *,
    torch_devices: TorchDeviceAvailability,
    detected_hardware: set[str] | frozenset[str],
    interpreter: str | Path,
) -> None:
    """Raise an actionable setup error if the selected host cannot use its GPU.

    Raises:
        RuntimeError: If the selected GPU backend is not available in *interpreter*.
    """
    if target.device == "cpu" or torch_devices.as_mapping().get(target.device, False):
        return

    backend = "torch.xpu" if target.device == "xpu" else "torch.cuda"
    if target.device in detected_hardware:
        detected = f"{target.device.upper()} hardware was detected"
    else:
        detected = f"{target.device.upper()} was explicitly selected"
    detail = f" {torch_devices.error}" if torch_devices.error else ""
    msg = (
        f"{detected}, but host interpreter {str(interpreter)!r} cannot use {backend}"
        f" (is_available() is false).{detail} Prepare the host with uv extra {target.uv_extra!r}, "
        "verify the GPU driver/runtime, or explicitly choose CPU with --device cpu "
        "or PHYSICALAI_LEROBOT_DEVICE=cpu."
    )
    raise RuntimeError(msg)


def _pci_gpu_hardware(devices: Path) -> set[str]:
    detected: set[str] = set()
    if not devices.is_dir():
        return detected
    for device in devices.iterdir():
        try:
            vendor = (device / "vendor").read_text(encoding="ascii").strip().lower()
            class_code = int((device / "class").read_text(encoding="ascii").strip(), 16)
        except (OSError, ValueError):
            continue
        if class_code >> 16 != _PCI_DISPLAY_CLASS:
            continue
        if vendor == "0x8086":
            detected.add("xpu")
        elif vendor == "0x10de":
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


__all__ = [
    "IntegrationTarget",
    "TorchDeviceAvailability",
    "detect_gpu_hardware",
    "probe_torch_devices",
    "require_torch_device",
    "select_integration_target",
]
