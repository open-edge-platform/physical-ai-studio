# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest


def test_compressed_openvino_artifact_remains_loadable(tmp_path: Path) -> None:
    ov = pytest.importorskip("openvino")
    pytest.importorskip("nncf")
    from physicalai.export.hooks import compress_weights_openvino_int8_sym

    rng = np.random.default_rng(42)
    state = ov.opset13.parameter([1, 64], np.float32, name="state")
    weights = ov.opset13.constant(rng.standard_normal((64, 64)).astype(np.float32))
    graph = ov.Model([ov.opset13.matmul(state, weights, False, False)], [state])
    path = tmp_path / "model.xml"
    ov.save_model(graph, path)
    del graph
    compress_weights_openvino_int8_sym(str(path))
    compiled = ov.Core().compile_model(path, "CPU")
    assert compiled([np.ones((1, 64), dtype=np.float32)])[0].shape == (1, 64)
