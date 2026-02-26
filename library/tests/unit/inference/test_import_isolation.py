# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Import isolation tests for decoupled inference modules.

Verifies that key modules can be imported without pulling in torch,
ensuring lightweight deployment scenarios remain viable.

Each test spawns a clean subprocess to avoid pytest's own imports
contaminating sys.modules.
"""

import subprocess
import sys

import pytest


class TestTorchFreeImports:
    """Verify that decoupled modules do not transitively import torch."""

    @pytest.mark.parametrize(
        ("description", "code"),
        [
            pytest.param(
                "export.backends",
                "import sys; from physicalai.export.backends import ExportBackend; assert 'torch' not in sys.modules",
                id="export_backends",
            ),
            pytest.param(
                "data.constants",
                "import sys; from physicalai.data.constants import ACTION, IMAGES, STATE; assert 'torch' not in sys.modules",
                id="data_constants",
            ),
            pytest.param(
                "inference.adapters.base",
                "import sys; from physicalai.inference.adapters.base import RuntimeAdapter; assert 'torch' not in sys.modules",
                id="adapter_base",
            ),
            pytest.param(
                "inference.adapters.onnx",
                "import sys; from physicalai.inference.adapters.onnx import ONNXAdapter; assert 'torch' not in sys.modules",
                id="onnx_adapter",
            ),
            pytest.param(
                "inference.adapters.openvino",
                "import sys; from physicalai.inference.adapters.openvino import OpenVINOAdapter; assert 'torch' not in sys.modules",
                id="openvino_adapter",
            ),
        ],
    )
    def test_single_module_no_torch(self, description: str, code: str) -> None:
        """Importing {description} must not load torch."""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Importing {description} pulled in torch.\nstderr: {result.stderr}"

    def test_adapters_init_no_torch(self) -> None:
        """Importing the adapters package must not load any torch modules."""
        code = (
            "import sys; "
            "from physicalai.inference.adapters import RuntimeAdapter, ONNXAdapter, OpenVINOAdapter, get_adapter; "
            "torch_mods = [m for m in sys.modules if m.startswith('torch')]; "
            "assert not torch_mods, f'torch leaked: {torch_mods}'"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Adapters __init__ pulled in torch.\nstderr: {result.stderr}"

    def test_inference_model_no_torch(self) -> None:
        """Importing InferenceModel must not load any torch modules."""
        code = (
            "import sys; "
            "from physicalai.inference import InferenceModel; "
            "torch_mods = [m for m in sys.modules if m.startswith('torch')]; "
            "assert not torch_mods, f'torch leaked: {torch_mods}'"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"InferenceModel import pulled in torch.\nstderr: {result.stderr}"


class TestBackwardCompatibility:
    """Verify that legacy import paths still work after refactoring."""

    def test_backward_compat_export_backend(self) -> None:
        """ExportBackend is still importable from mixin_export."""
        code = "from physicalai.export.mixin_export import ExportBackend; print(ExportBackend.OPENVINO)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Backward-compat ExportBackend import failed.\nstderr: {result.stderr}"

    def test_backward_compat_constants(self) -> None:
        """Constants are still importable from data.observation."""
        code = "from physicalai.data.observation import ACTION, IMAGES, STATE; print(ACTION)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Backward-compat constants import failed.\nstderr: {result.stderr}"

    def test_backward_compat_adapters(self) -> None:
        """Torch-dependent adapters are still importable via lazy __getattr__."""
        code = "from physicalai.inference.adapters import TorchAdapter, TorchExportAdapter"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Backward-compat adapter import failed.\nstderr: {result.stderr}"
