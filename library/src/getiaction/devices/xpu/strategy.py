# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning strategy for single XPU device."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch
    from lightning.pytorch.strategies import StrategyRegistry
    from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
except ImportError as e:
    msg = (
        "PyTorch and Lightning not installed. SingleXPUStrategy requires PyTorch and Lightning."
        "\nInstall with: pip install getiaction[torch]"
    )
    raise ImportError(msg) from e

if TYPE_CHECKING:
    from lightning.fabric.plugins import CheckpointIO
    from lightning.pytorch.accelerators.accelerator import Accelerator
    from lightning.pytorch.plugins.precision import Precision
    from lightning_fabric.utilities.types import _DEVICE


class SingleXPUStrategy(SingleDeviceStrategy):
    """Strategy for training on single XPU device."""

    strategy_name = "xpu_single"

    def __init__(
        self,
        device: _DEVICE = "xpu:0",
        accelerator: Accelerator | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
    ) -> None:
        """Initialize the SingleXPUStrategy.

        Args:
            device (_DEVICE, optional): The XPU device to use. Defaults to "xpu:0".
            accelerator (Accelerator | None, optional): The accelerator instance to use.
                Defaults to None.
            checkpoint_io (CheckpointIO | None, optional): The checkpoint I/O plugin to use.
                Defaults to None.
            precision_plugin (Precision | None, optional): The precision plugin to use.
                Defaults to None.

        Raises:
            MisconfigurationException: If XPU devices are not available on the system.
        """
        if not torch.xpu.is_available():
            msg = "`SingleXPUStrategy` requires XPU devices to run"
            raise MisconfigurationException(msg)
        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )


StrategyRegistry.register(
    SingleXPUStrategy.strategy_name,
    SingleXPUStrategy,
    override=True,
    description="Strategy that enables training on single Intel XPU device.",
)
