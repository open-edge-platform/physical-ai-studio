# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests that validate physicalai.inference imports without torch."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.xfail(
    strict=False,
    reason=(
        "physicalai.inference.model imports physicalai.data.constants "
        "-> physicalai.data.__init__ -> datamodules.py -> torch. "
        "This is a transitive studio dependency in package __init__ "
        "chains outside this PR's scope."
    ),
)
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


@pytest.mark.xfail(
    strict=False,
    reason=(
        "physicalai.inference.adapters.__init__ imports "
        "physicalai.inference.model -> physicalai.data.constants "
        "-> physicalai.data.__init__ -> torch. Same transitive chain "
        "as inference.model, a package __init__ architecture issue "
        "outside this PR's scope."
    ),
)
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
            "torch_modules = [k for k in sys.modules if k == 'torch']; "
            "assert not torch_modules, f'torch leaked: {torch_modules}'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import leaked torch:\nstdout: {result.stdout}\nstderr: {result.stderr}"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "physicalai.export.backends is torch-free in isolation, but "
        "importing it triggers physicalai.export.__init__ "
        "-> mixin_export.py -> lightning -> torch. The package __init__ "
        "imports lightning (for augmentation) which transitively loads torch, "
        "a package-level architecture issue outside this PR's scope."
    ),
)
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
