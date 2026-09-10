# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning strategies for Intel XPU devices."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, override

import torch
from lightning.pytorch.strategies import StrategyRegistry
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException

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


class XPUDDPStrategy(DDPStrategy):
    """Strategy for distributed training on multiple XPU devices."""

    strategy_name = "xpu_ddp"

    def __init__(
        self,
        *,
        process_group_backend: str = "xccl",
        find_unused_parameters: bool = True,
        **kwargs: Any,  # noqa: ANN401 - standard pattern for forwarding to parent
    ) -> None:
        """Initialize the XPUDDPStrategy.

        Args:
            process_group_backend (str): The process group backend to use. Defaults to "xccl"
                which is required for multi-XPU communication via Intel oneCCL.
            find_unused_parameters (bool): Whether to find unused parameters during backward pass.
                Defaults to True as Embodied/VLA policies frequently freeze certain components.
            **kwargs (Any): Additional options to pass to DDPStrategy.

        Raises:
            MisconfigurationException: If XPU devices are not available on the system.
        """
        if not torch.xpu.is_available():
            msg = "`XPUDDPStrategy` requires XPU devices to run"
            raise MisconfigurationException(msg)

        super().__init__(
            process_group_backend=process_group_backend,
            find_unused_parameters=find_unused_parameters,
            **kwargs,
        )

    @override
    def _setup_model(self, model: torch.nn.Module) -> torch.nn.parallel.DistributedDataParallel:
        """Wrap the model in DDP using an XPU stream instead of the hardcoded CUDA one.

        Lightning's ``DDPStrategy._setup_model`` always opens a ``torch.cuda.Stream``
        before wrapping, which is a no-op (and would error) on XPU. Mirror the same
        stream-scoped wrapping using ``torch.xpu`` instead.

        Args:
            model: The model to wrap.

        Returns:
            The wrapped distributed data parallel module.
        """
        device_ids = self.determine_ddp_device_ids()
        ctx = torch.xpu.stream(torch.xpu.Stream()) if device_ids is not None else nullcontext()
        with ctx:
            return torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=device_ids,
                **self._ddp_kwargs,
            )


StrategyRegistry.register(
    XPUDDPStrategy.strategy_name,
    XPUDDPStrategy,
    override=True,
    description="Strategy that enables distributed training on multiple Intel XPU devices.",
)
