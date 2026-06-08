# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration smoke tests for ExecuTorch export-inference contract.

These tests verify the metadata format written by ``to_executorch()`` is
compatible with what ``ExecuTorchAdapter.load()`` reads, without requiring
an actual ``executorch`` installation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from physicalai.export.mixin_policy import ExportablePolicyMixin, ExportBackend
from physicalai.inference.adapters.executorch import ExecuTorchAdapter
from physicalai.inference.model import InferenceModel


class _ModelWithSampleInput(torch.nn.Module):
    """Minimal model that exposes ``sample_input`` for export."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.linear(batch["obs"])

    @property
    def sample_input(self) -> dict[str, torch.Tensor]:
        return {"obs": torch.randn(1, 4), "goal": torch.randn(1, 4)}


class _ExportWrapper(ExportablePolicyMixin):
    """Thin wrapper around :class:`ExportablePolicyMixin` for testing."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        # Identity preprocessor: returns the sample dict unchanged.
        self._preprocessor = torch.nn.Identity()

    @property
    def sample_input(self) -> dict[str, torch.Tensor] | None:
        model_sample = getattr(self.model, "sample_input", None)
        return model_sample if isinstance(model_sample, dict) else None

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        return [ExportBackend.ONNX, ExportBackend.OPENVINO, ExportBackend.EXECUTORCH]


class TestExecuTorchIntegration:
    """Smoke tests for the ExecuTorch export ↔ inference metadata contract."""

    def test_metadata_contract_export_to_inference(self, tmp_path: Path) -> None:
        """Manifest written by export is correctly consumed by the adapter.

        Creates a manifest.json that mirrors what ``create_manifest`` would
        produce and verifies that ``ExecuTorchAdapter.load()`` reads
        ``input_names`` from it.
        """
        # -- arrange: create model.pte stub and manifest.json
        model_path = tmp_path / "model.pte"
        model_path.write_bytes(b"\x00")  # stub binary

        manifest = {
            "input_names": ["state", "action"],
        }
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w") as fh:
            json.dump(manifest, fh)

        # -- mock executorch.runtime so load() doesn't import the real package
        mock_runtime_instance = MagicMock()
        mock_program = MagicMock()
        mock_method = MagicMock()
        mock_runtime_instance.load_program.return_value = mock_program
        mock_program.load_method.return_value = mock_method
        mock_runtime_class = MagicMock()
        mock_runtime_class.get.return_value = mock_runtime_instance
        mock_et_runtime = MagicMock()
        mock_et_runtime.Runtime = mock_runtime_class

        with patch.dict(
            "sys.modules",
            {
                "executorch": MagicMock(),
                "executorch.runtime": mock_et_runtime,
            },
        ):
            adapter = ExecuTorchAdapter()
            adapter.load(model_path)

        # -- assert
        assert adapter.input_names == ["state", "action"]

    def test_export_creates_metadata_with_input_names(self, tmp_path: Path) -> None:
        """``to_executorch()`` writes ``input_names`` to manifest.json.

        All ``executorch`` internals are mocked so no real .pte file is
        produced; we only verify the side-effect on manifest.json.
        """
        model = _ModelWithSampleInput()
        wrapper = _ExportWrapper(model)

        # -- mock executorch export pipeline
        mock_exec_program = MagicMock()
        mock_exec_program.write_to_file = MagicMock()

        mock_edge_program = MagicMock()
        mock_edge_program.to_executorch.return_value = mock_exec_program

        mock_exir = MagicMock()
        mock_exir.to_edge_transform_and_lower.return_value = mock_edge_program

        with (
            patch.dict(
                sys.modules,
                {
                    "executorch": MagicMock(),
                    "executorch.exir": mock_exir,
                    "executorch.backends": MagicMock(),
                    "executorch.backends.openvino": MagicMock(),
                    "executorch.backends.openvino.partitioner": MagicMock(),
                    "executorch.exir.backend": MagicMock(),
                    "executorch.exir.backend.backend_details": MagicMock(),
                },
            ),
            patch("torch.export.export", return_value=MagicMock()),
        ):
            wrapper.to_executorch(tmp_path / "model.pte")

        # -- read manifest written by create_manifest
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists(), "manifest.json was not created by to_executorch()"

        with manifest_path.open() as fh:
            manifest = json.load(fh)

        assert manifest is not None
        assert "input_names" in manifest
        assert manifest["input_names"] == ["obs", "goal"]

    def test_pte_extension_detected_as_executorch(self, tmp_path: Path) -> None:
        """``InferenceModel._detect_backend()`` maps ``.pte`` → ``executorch``.

        ``_detect_backend`` is an instance method that looks for files in
        ``self.export_dir``, so we create a stub ``.pte`` file and
        instantiate just enough of InferenceModel to call it.
        """
        # Create a stub .pte file so _detect_backend finds it
        pte_file = tmp_path / "model.pte"
        pte_file.write_bytes(b"\x00")

        # _detect_backend is an instance method on InferenceModel that uses
        # self.export_dir.  We construct a minimal instance via __new__ and
        # set the attribute directly to avoid the full __init__ chain.
        model = InferenceModel.__new__(InferenceModel)
        model.export_dir = tmp_path

        result = model._detect_backend()
        assert result == "executorch"
