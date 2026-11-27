# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Direct OpenVINO IR compiler backend (``.xml`` + ``.bin``) for ``lerobot-export``."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from lerobot.export.configs import ExportBackendConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor, nn
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("openvino")
@dataclass
class OpenVINOConfig(ExportBackendConfig):
    """Compile each traced policy part directly to portable OpenVINO IR (``.xml`` + ``.bin``).

    Tracing runs on CPU and emits hardware-portable OpenVINO IR that can be compiled at runtime
    for Intel CPU, iGPU, dGPU, NPU, or ``AUTO``.
    """

    runtime: ClassVar[str] = "openvino"
    file_suffix: ClassVar[str] = ".xml"
    device: ClassVar[str] = "cpu"
    runtime_imports: ClassVar[tuple[str, ...]] = ("openvino", "lerobot_openvino_enablement.runtime")
    runtime_modules: ClassVar[tuple[str, ...]] = ("openvino",)
    packages: ClassVar[tuple[str, ...]] = ("openvino", "nncf")
    quantizations: ClassVar[tuple[str, ...]] = ("fp16", "int8", "int4")

    ov_device: str = "CPU"
    compression: str | None = None

    def quantized(
        self,
        module: nn.Module,
        example_inputs: tuple[Tensor, ...],
        mode: str,
    ) -> contextlib.AbstractContextManager[None]:
        """Record the requested NNCF/FP16 compression mode for the compile step.

        Weight-only compression (``fp16``, ``int8``, ``int4``) operates directly on the
        converted OpenVINO IR in :meth:`compile`, so tracing runs unmodified.
        """
        del module, example_inputs
        if mode not in self.quantizations:
            offered = ", ".join(self.quantizations)
            msg = f"Unsupported OpenVINO quantization mode {mode!r}; expected one of: {offered}."
            raise ValueError(msg)
        self.compression = mode
        return contextlib.nullcontext()

    def compile(self, program: ExportedProgram, path: Path) -> Path:
        """Convert *program* to OpenVINO IR, optionally compress weights, and save ``.xml`` + ``.bin``."""
        import openvino as ov

        # Decompose high-level ATen ops and normalize PyTorch-only idioms before OpenVINO conversion.
        program = program.run_decompositions()
        _normalize_for_openvino(program)
        ov_model = ov.convert_model(program)

        if self.compression in {"int8", "int4"}:
            import nncf

            if self.compression == "int8":
                ov_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_SYM)
            else:
                ov_model = nncf.compress_weights(
                    ov_model,
                    mode=nncf.CompressWeightsMode.INT4_ASYM,
                    group_size=128,
                )

        compress_to_fp16 = self.compression == "fp16"
        ov.save_model(ov_model, str(path), compress_to_fp16=compress_to_fp16)
        return path.with_suffix(".bin")

    def run(self, path: Path, data: Path | None, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """Run a compiled OpenVINO IR part on PyTorch tensors for per-part parity checking."""
        del data
        import numpy as np
        import openvino as ov
        import torch

        core = ov.Core()
        compiled = core.compile_model(str(path), self.ov_device)
        np_inputs: list[np.ndarray] = []
        for tensor, port in zip(inputs, compiled.inputs, strict=True):
            host = tensor.detach().cpu()
            if host.dtype == torch.bfloat16:
                host = host.to(torch.float32)
            array = np.ascontiguousarray(host.numpy())
            expected_dtype = port.get_element_type().to_dtype()
            if array.dtype != expected_dtype:
                array = array.astype(expected_dtype, copy=False)
            np_inputs.append(array)

        results = compiled(np_inputs)
        return tuple(torch.from_numpy(np.array(results[output], copy=True)) for output in compiled.outputs)


def _normalize_for_openvino(program: ExportedProgram) -> None:
    """Normalize PyTorch FX idioms not directly accepted by OpenVINO's PyTorch frontend.

    - ``aten.empty_permuted.default`` -> ``aten.empty.memory_format`` (stride layout is irrelevant to IR).
    - ``aten.trunc.default(x)`` -> ``where(x >= 0, floor(x), ceil(x))`` (missing translator in OV frontend).
    - Drop 1-D empty ``(0,)`` tensors from ``aten.cat.default`` (PyTorch treats them as no-ops even when
      their rank differs from the concatenation axis, which OpenVINO's Concat shape validation rejects).
    """
    import torch

    graph = program.graph
    changed = False
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is torch.ops.aten.empty_permuted.default:
            # Signature: empty_permuted(size, physical_layout, *, dtype=..., layout=..., device=..., pin_memory=...)
            size = node.args[0]
            with graph.inserting_before(node):
                replacement = graph.call_function(
                    torch.ops.aten.empty.memory_format,
                    args=(size,),
                    kwargs=dict(node.kwargs),
                )
                replacement.meta = dict(node.meta)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            changed = True
        elif node.target is torch.ops.aten.trunc.default:
            x = node.args[0]
            with graph.inserting_before(node):
                ge_zero = graph.call_function(torch.ops.aten.ge.Scalar, args=(x, 0))
                floored = graph.call_function(torch.ops.aten.floor.default, args=(x,))
                ceiled = graph.call_function(torch.ops.aten.ceil.default, args=(x,))
                replacement = graph.call_function(torch.ops.aten.where.self, args=(ge_zero, floored, ceiled))
                replacement.meta = dict(node.meta)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            changed = True
        elif node.target is torch.ops.aten.cat.default and node.args and isinstance(node.args[0], (list, tuple)):
            tensors = list(node.args[0])
            kept = [t for t in tensors if not _is_1d_empty_tensor(t)]
            if len(kept) != len(tensors) and kept:
                if len(kept) == 1:
                    node.replace_all_uses_with(kept[0])
                    graph.erase_node(node)
                else:
                    new_args = (kept, *node.args[1:])
                    node.args = new_args
                changed = True
    if changed:
        graph.eliminate_dead_code()
        program.graph_module.recompile()


def _is_1d_empty_tensor(node: object) -> bool:
    meta = getattr(node, "meta", None)
    if not isinstance(meta, dict):
        return False
    val = meta.get("val")
    shape = getattr(val, "shape", None)
    return shape is not None and len(shape) == 1 and shape[0] == 0
