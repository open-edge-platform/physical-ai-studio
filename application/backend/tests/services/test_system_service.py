from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from schemas.hardware import DeviceType, InferenceBackend
from services.system_service import SystemService


@pytest.fixture(autouse=True)
def _reset_remote_backend_cache():
    """Ensure no cached RemoteTrainingBackend leaks between tests."""
    SystemService._clear_remote_backend_cache()
    yield
    SystemService._clear_remote_backend_cache()


def _device_props(name: str, total_memory: int) -> SimpleNamespace:
    return SimpleNamespace(name=name, total_memory=total_memory)


def _device_summary(device):
    return device.backend, device.device, device.type, device.name, device.memory, device.index


def test_get_inference_devices_returns_torch_cpu_when_openvino_missing() -> None:
    with (
        patch("services.system_service.import_module", side_effect=ImportError),
        patch("services.system_service.SystemService._get_system_memory", return_value=64_000),
        patch("services.system_service.torch.xpu.is_available", return_value=False),
        patch("services.system_service.torch.cuda.is_available", return_value=False),
    ):
        devices = SystemService.get_inference_devices()

    assert len(devices) == 1
    assert devices[0].backend == InferenceBackend.TORCH
    assert devices[0].device == "cpu"
    assert devices[0].type == DeviceType.CPU
    assert devices[0].name == "CPU"
    assert devices[0].memory == 64_000
    assert devices[0].index is None


def test_get_inference_devices_returns_torch_accelerators() -> None:
    with (
        patch("services.system_service.import_module", side_effect=ImportError),
        patch("services.system_service.SystemService._get_system_memory", return_value=64_000),
        patch("services.system_service.torch.xpu.is_available", return_value=True),
        patch("services.system_service.torch.xpu.device_count", return_value=1),
        patch(
            "services.system_service.torch.xpu.get_device_properties",
            return_value=_device_props("Intel Arc", 8_000),
        ),
        patch("services.system_service.torch.cuda.is_available", return_value=True),
        patch("services.system_service.torch.cuda.device_count", return_value=1),
        patch(
            "services.system_service.torch.cuda.get_device_properties",
            return_value=_device_props("NVIDIA GPU", 16_000),
        ),
    ):
        devices = SystemService.get_inference_devices()

    assert [_device_summary(device) for device in devices] == [
        (InferenceBackend.TORCH, "cpu", DeviceType.CPU, "CPU", 64_000, None),
        (InferenceBackend.TORCH, "xpu:0", DeviceType.XPU, "Intel Arc", 8_000, 0),
        (InferenceBackend.TORCH, "cuda:0", DeviceType.CUDA, "NVIDIA GPU", 16_000, 0),
    ]


def test_get_inference_devices_returns_openvino_devices() -> None:
    core = MagicMock()
    core.available_devices = ["CPU", "GPU.0", "NPU.0"]
    core.get_property.side_effect = lambda device, prop: {
        ("GPU.0", "FULL_DEVICE_NAME"): "Intel GPU",
        ("GPU.0", "GPU_DEVICE_TOTAL_MEM_SIZE"): "32000",
        ("GPU.0", "DEVICE_ID"): "0",
        ("NPU.0", "FULL_DEVICE_NAME"): "Intel NPU",
        ("NPU.0", "DEVICE_ID"): "0",
    }[(device, prop)]
    openvino = SimpleNamespace(Core=MagicMock(return_value=core))

    with (
        patch("services.system_service.import_module", return_value=openvino),
        patch("services.system_service.SystemService._get_system_memory", return_value=64_000),
        patch("services.system_service.torch.xpu.is_available", return_value=False),
        patch("services.system_service.torch.cuda.is_available", return_value=False),
    ):
        devices = SystemService.get_inference_devices()

    assert [_device_summary(device) for device in devices] == [
        (InferenceBackend.TORCH, "cpu", DeviceType.CPU, "CPU", 64_000, None),
        (InferenceBackend.OPENVINO, "CPU", DeviceType.CPU, "CPU", 64_000, None),
        (InferenceBackend.OPENVINO, "GPU.0", DeviceType.XPU, "Intel GPU", 32_000, 0),
        (InferenceBackend.OPENVINO, "NPU.0", DeviceType.NPU, "Intel NPU", 64_000, 0),
    ]


def test_get_inference_devices_uses_openvino_fallback_values() -> None:
    core = MagicMock()
    core.available_devices = ["GPU.1"]
    core.get_property.side_effect = RuntimeError("unsupported property")
    openvino = SimpleNamespace(Core=MagicMock(return_value=core))

    with (
        patch("services.system_service.import_module", return_value=openvino),
        patch("services.system_service.SystemService._get_system_memory", return_value=64_000),
        patch("services.system_service.torch.xpu.is_available", return_value=False),
        patch("services.system_service.torch.cuda.is_available", return_value=False),
    ):
        devices = SystemService.get_inference_devices()

    assert devices[-1].backend == InferenceBackend.OPENVINO
    assert devices[-1].device == "GPU.1"
    assert devices[-1].type == DeviceType.XPU
    assert devices[-1].name == "GPU.1"
    assert devices[-1].memory is None
    assert devices[-1].index == 1


def test_get_available_training_devices_uses_local_in_local_mode() -> None:
    import asyncio

    from schemas.hardware import DeviceInfo

    settings = SimpleNamespace(training_mode="local")
    local_devices = [DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)]
    with (
        patch("settings.get_settings", return_value=settings),
        patch(
            "services.system_service.SystemService.get_training_devices",
            return_value=local_devices,
        ),
    ):
        result = asyncio.run(SystemService.get_available_training_devices())

    assert result.mode == "local"
    assert result.remote_available is True
    assert result.devices == local_devices


def test_get_available_training_devices_queries_remote_in_remote_mode() -> None:
    import asyncio

    from schemas.hardware import DeviceInfo

    settings = SimpleNamespace(training_mode="remote")
    remote_devices = [DeviceInfo(type=DeviceType.CUDA, name="NVIDIA A100", memory=42949672960, index=0)]

    backend = MagicMock()

    async def _fake_get_training_devices():
        return remote_devices

    backend.get_training_devices.side_effect = _fake_get_training_devices

    with (
        patch("settings.get_settings", return_value=settings),
        patch("services.training_backends.remote.RemoteTrainingBackend", return_value=backend),
    ):
        result = asyncio.run(SystemService.get_available_training_devices())

    assert result.mode == "remote"
    assert result.remote_available is True
    assert result.devices == remote_devices


def test_get_available_training_devices_reports_unavailable_when_remote_unreachable() -> None:
    import asyncio

    from services.training_backends.remote import RemoteTrainingError

    settings = SimpleNamespace(training_mode="remote")

    backend = MagicMock()

    async def _boom():
        raise RemoteTrainingError("trainer unreachable")

    backend.get_training_devices.side_effect = _boom

    with (
        patch("settings.get_settings", return_value=settings),
        patch("services.training_backends.remote.RemoteTrainingBackend", return_value=backend),
    ):
        result = asyncio.run(SystemService.get_available_training_devices())

    # Remote unreachable must NOT fall back to local CPU; training is disabled instead.
    assert result.mode == "remote"
    assert result.remote_available is False
    assert result.devices == []
