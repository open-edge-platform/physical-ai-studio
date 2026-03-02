# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests that validate physicalai.inference imports without torch."""

from __future__ import annotations

import subprocess
import sys


def test_inference_model_imports_without_torch():
    """Verify InferenceModel can be imported without torch in sys.modules.
    
    Note: torch submodules may be loaded transitively via other dependencies,
    but the main torch module itself should not be directly imported.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from physicalai.inference.model import InferenceModel; "
            "torch_modules = [k for k in sys.modules if k == 'torch']; "
            "assert not torch_modules, f'torch leaked: {torch_modules}'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import leaked torch:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_adapters_import_without_torch():
    """Verify non-torch adapters import without torch.
    
    Note: torch submodules may be loaded transitively via other dependencies,
    but the main torch module itself should not be directly imported.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from physicalai.inference.adapters import RuntimeAdapter, ONNXAdapter, OpenVINOAdapter; "
            "from physicalai.inference.adapters.executorch import ExecuTorchAdapter; "
            "torch_modules = [k for k in sys.modules if k == 'torch']; "
            "assert not torch_modules, f'torch leaked: {torch_modules}'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import leaked torch:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_export_backends_import_without_torch():
    """Verify ExportBackend enum imports without torch.
    
    Note: torch submodules may be loaded transitively via other dependencies,
    but the main torch module itself should not be directly imported.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from physicalai.export.backends import ExportBackend; "
            "assert hasattr(ExportBackend, 'EXECUTORCH'); "
            "torch_modules = [k for k in sys.modules if k == 'torch']; "
            "assert not torch_modules, f'torch leaked: {torch_modules}'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import leaked torch:\nstdout: {result.stdout}\nstderr: {result.stderr}"
