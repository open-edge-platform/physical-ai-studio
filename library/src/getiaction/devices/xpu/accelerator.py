# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning accelerator for XPU device."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.pytorch.accelerators.accelerator import Accelerator

from getiaction.devices import is_xpu_available


class XPUAccelerator(Accelerator):
    """Support for a XPU, optimized for large-scale machine learning."""

    accelerator_name = "xpu"

    def setup_device(self, device: torch.device) -> None:  # noqa: PLR6301
        """Set up the XPU device for computation.

        This method configures the specified XPU device to be used for torch operations.
        It validates that the provided device is of type 'xpu' and then sets it as the
        active device.

        Args:
            device (torch.device): The torch device to set up. Must be of type 'xpu'.

        Raises:
            RuntimeError: If the provided device type is not 'xpu'.

        Example:
            >>> accelerator = XPUAccelerator()
            >>> device = torch.device('xpu:0')
            >>> accelerator.setup_device(device)
        """
        if device.type != "xpu":
            msg = f"Device should be xpu, got {device} instead"
            raise RuntimeError(msg)

        torch.xpu.set_device(device)

    @staticmethod
    def parse_devices(devices: str | list | torch.device) -> list:
        """Parse device specification into a list of devices.

        This function normalizes different device specification formats into a consistent
        list format for use with PyTorch device management.

        Args:
            devices (str | list | torch.device): Device specification that can be:
                - A string representing a single device (e.g., 'cuda:0', 'cpu')
                - A list of device specifications
                - A torch.device object

        Returns:
            list: A list containing the device specification(s). If input is already a list,
                it is returned as-is. Otherwise, the input is wrapped in a list.
        """
        if isinstance(devices, list):
            return devices
        return [devices]

    @staticmethod
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        """Convert a list of device indices to a list of XPU torch devices.

        Args:
            devices (list): A list of device indices (integers) to be converted to XPU devices.

        Returns:
            list[torch.device]: A list of torch.device objects configured for XPU with the specified indices.
        """
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Returns number of XPU devices available."""
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        """Check if XPU (Intel GPU) acceleration is available on the system.

        Returns:
            bool: True if XPU devices are available and can be used for computation,
                  False otherwise.
        """
        return is_xpu_available()

    def get_device_stats(self, device: str | torch.device) -> dict[str, Any]:  # noqa: PLR6301
        """Returns XPU devices stats."""
        return {"name": device}

    def teardown(self) -> None:
        """Clean up any state created by the accelerator."""


AcceleratorRegistry.register(
    XPUAccelerator.accelerator_name,
    XPUAccelerator,
    override=True,
    description="Accelerator supporting Intel XPU devices.",
)
