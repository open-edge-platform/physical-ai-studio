from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from schemas.hardware import DeviceType, InferenceBackend
from services.system_service import SystemService


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
