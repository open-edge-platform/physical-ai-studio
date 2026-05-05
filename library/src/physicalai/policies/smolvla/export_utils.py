# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export-time utilities for the SmolVLA policy."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference
from onnx.external_data_helper import load_external_data_for_model


def patch_onnx_for_ort(onnx_path: str | Path) -> None:
    """Patch a SmolVLA ONNX export so onnxruntime (CPU EP) can load it.

    Three issues from the dynamo exporter are fixed:

    1. ``Expand`` nodes ``node_full`` / ``node_full_1`` carry stale
       ``value_info`` declaring ``int64`` while their inputs are actually
       ``float32``. We drop *all* value_info and re-run shape inference so
       downstream type inference is reliable.
    2. The full-graph export embeds the VLM weights and many activations as
       ``bfloat16``. ORT's CPU EP lacks kernels for most bfloat16 ops
       (e.g. ``Mul`` produces ``NOT_IMPLEMENTED``). We rewrite every
       bfloat16 initializer to ``float32`` and retarget every ``Cast(to=BFLOAT16)``
       to ``Cast(to=FLOAT)``.
    3. After re-inference, ``Where(cond, X, Y)`` nodes can have X/Y branches
       that disagree with the Where's declared output dtype. We insert a
       ``Cast`` on the mismatched branch.

    The patched model is saved back to ``onnx_path`` with external data
    (overwriting the original ``*.onnx.data`` sidecar) since converting
    bfloat16 -> float32 doubles the weight size.

    Args:
        onnx_path: Path to the ``.onnx`` file produced by SmolVLA's exporter.
    """

    onnx_path = Path(onnx_path)
    base_dir = str(Path(str(onnx_path)).parent)
    model = onnx.load(onnx_path, load_external_data=False)
    load_external_data_for_model(model, base_dir)
    graph = model.graph

    # 1. Drop stale value_info entirely; we re-infer from scratch.
    del graph.value_info[:]

    # 2a. Convert bfloat16 initializers to float32.
    new_inis = []
    for ini in graph.initializer:
        if ini.data_type == TensorProto.BFLOAT16:
            arr = numpy_helper.to_array(ini)
            f32 = np.asarray(arr, dtype=np.float32)
            new_ini = numpy_helper.from_array(f32, name=ini.name)
            new_inis.append(new_ini)
        else:
            new_inis.append(ini)
    del graph.initializer[:]
    graph.initializer.extend(new_inis)

    # 2b. Retarget Cast nodes that produce bfloat16 to produce float32.
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.BFLOAT16:
                    attr.i = TensorProto.FLOAT

    # Re-run shape inference now that the type landscape is consistent.
    model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    graph = model.graph

    # 3. Insert Casts where Where's branches disagree with its output dtype.
    name_to_dtype: dict[str, int] = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        name_to_dtype[vi.name] = vi.type.tensor_type.elem_type
    for ini in graph.initializer:
        name_to_dtype[ini.name] = ini.data_type

    new_nodes = []
    inserted = 0
    for node in graph.node:
        if node.op_type == "Where" and len(node.input) == 3:
            out_dt = name_to_dtype.get(node.output[0])
            for branch_idx in (1, 2):  # X, Y
                in_name = node.input[branch_idx]
                in_dt = name_to_dtype.get(in_name)
                if out_dt and in_dt and in_dt != out_dt:
                    cast_out = f"{in_name}__cast_to_{out_dt}_{inserted}"
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            inputs=[in_name],
                            outputs=[cast_out],
                            name=f"patch_cast_{inserted}",
                            to=out_dt,
                        ),
                    )
                    node.input[branch_idx] = cast_out
                    inserted += 1
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)

    model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)

    # Save back with external data; overwrites the existing *.onnx.data
    # sidecar so the new (larger) float32 weights fit.
    external_data_name = onnx_path.name + ".data"
    external_data_path = onnx_path.parent / external_data_name
    if external_data_path.exists():
        external_data_path.unlink()
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=1024,
    )
