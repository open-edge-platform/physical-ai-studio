# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests from training to export to inference.

This module contains comprehensive E2E tests that validate the complete pipeline:
1. Train a policy (ACT, Dummy) with real/dummy data
2. Export the trained policy to multiple backends (OpenVINO, ONNX, TorchScript, ExecuTorch)
3. Load exported model for inference
4. Verify numerical consistency between training and inference

Uses parameterized tests with cartesian product to test all policy × backend combinations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from getiaction.data import DataModule
from getiaction.inference import InferenceModel
from getiaction.policies import get_policy
from getiaction.train import Trainer


class TestE2E:
    """Comprehensive end-to-end tests for training → export → inference pipeline."""

    @pytest.fixture
    def temp_export_dir(self):
        """Create temporary directory for export testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.parametrize(
        ("policy_name", "backend"),
        [
            # First-party policies with all backends
            ("dummy", "openvino"),
            ("dummy", "onnx"),
            ("dummy", "torch"),
            ("dummy", "torch_export_ir"),
            ("act", "openvino"),
            ("act", "onnx"),
            ("act", "torch"),
            ("act", "torch_export_ir"),
        ],
    )
    def test_train_export_inference_pipeline(
        self,
        policy_name: str,
        backend: str,
        dummy_dataset,
        temp_export_dir: Path,
    ):
        """Test complete pipeline: train → export → inference → validate.

        This test validates that:
        1. Policy can be trained successfully
        2. Trained policy can be exported to specified backend
        3. Exported model can be loaded for inference
        4. Inference produces consistent results (within tolerance)

        Args:
            policy_name: Name of policy to test ("act", "dummy")
            backend: Export backend ("openvino", "onnx", "torch", "torch_export_ir")
            dummy_dataset: Pytest fixture providing dummy dataset
            temp_export_dir: Temporary directory for export files
        """
        # 1. CREATE POLICY
        if policy_name == "act":
            # ACT requires action chunking configuration
            dataset = dummy_dataset(num_samples=16, state_dim=2, action_dim=2)
            chunk_size = 100
            dataset.delta_indices = {"action": list(range(chunk_size))}
            policy = get_policy(policy_name, source="getiaction")
        else:
            # Dummy policy uses simple configuration
            dataset = dummy_dataset(num_samples=16)
            from getiaction.policies.dummy import DummyConfig

            policy = get_policy(policy_name, source="getiaction", config=DummyConfig(action_shape=(2,)))

        # 2. TRAIN POLICY
        datamodule = DataModule(
            train_dataset=dataset,
            train_batch_size=4,
        )

        trainer = Trainer(
            fast_dev_run=True,  # Single batch for speed
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(policy, datamodule=datamodule)

        # 3. EXPORT POLICY
        export_dir = temp_export_dir / f"{policy_name}_{backend}"
        policy.export(export_dir, backend=backend)

        # Verify export files exist
        assert export_dir.exists()
        assert (export_dir / "metadata.yaml").exists()

        # 4. LOAD FOR INFERENCE
        inference_model = InferenceModel.load(export_dir)

        # Verify backend detection
        assert inference_model.backend.value == backend

        # 5. VALIDATE NUMERICAL CONSISTENCY
        # Get a sample observation from dataset
        sample_batch = next(iter(datamodule.train_dataloader()))

        # Get training policy prediction
        policy.eval()
        with torch.no_grad():
            train_output = policy.select_action(sample_batch)

        # Get inference prediction
        # Convert batch to inference format (numpy dict)
        inference_input = {
            key: value.cpu().numpy() for key, value in sample_batch.items() if key in ["image", "state"]
        }
        inference_output = inference_model.select_action(inference_input)

        # Compare outputs (allowing some tolerance for backend optimizations)
        train_action = train_output.cpu().numpy()
        inference_action = inference_output

        # For chunked policies (ACT), compare first action
        if policy_name == "act":
            train_action = train_action[:, 0, :]  # (batch, chunk, dim) -> (batch, dim)

        torch.testing.assert_close(
            torch.from_numpy(inference_action),
            torch.from_numpy(train_action),
            rtol=0.15,  # 15% relative tolerance
            atol=0.15,  # 0.15 absolute tolerance
        )

    @pytest.mark.parametrize("policy_name", ["act", "dummy"])
    def test_export_creates_complete_structure(
        self,
        policy_name: str,
        dummy_dataset,
        temp_export_dir: Path,
    ):
        """Test that export creates all required files and metadata.

        Args:
            policy_name: Name of policy to test
            dummy_dataset: Pytest fixture providing dummy dataset
            temp_export_dir: Temporary directory for export files
        """
        # Create and setup policy
        if policy_name == "act":
            dataset = dummy_dataset(num_samples=8, state_dim=2, action_dim=2)
            chunk_size = 100
            dataset.delta_indices = {"action": list(range(chunk_size))}
            policy = get_policy(policy_name, source="getiaction")
        else:
            dataset = dummy_dataset(num_samples=8)
            from getiaction.policies.dummy import DummyConfig

            policy = get_policy(policy_name, source="getiaction", config=DummyConfig(action_shape=(2,)))

        # Quick training
        datamodule = DataModule(train_dataset=dataset, train_batch_size=4)
        trainer = Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False, enable_progress_bar=False)
        trainer.fit(policy, datamodule=datamodule)

        # Export
        export_dir = temp_export_dir / policy_name
        policy.export(export_dir, backend="openvino")

        # Verify structure
        assert export_dir.exists()
        assert (export_dir / "metadata.yaml").exists()
        assert (export_dir / "metadata.json").exists()

        # Verify metadata contents
        metadata = InferenceModel.load(export_dir).metadata
        assert "policy_name" in metadata
        assert "policy_class" in metadata
        assert "input_shapes" in metadata
        assert "backend" in metadata
        assert metadata["backend"] == "openvino"

    @pytest.mark.parametrize(
        "backend",
        ["openvino", "onnx", "torch", "torch_export_ir"],
    )
    def test_backend_specific_files_created(
        self,
        backend: str,
        dummy_dataset,
        temp_export_dir: Path,
    ):
        """Test that each backend creates its expected model files.

        Args:
            backend: Export backend to test
            dummy_dataset: Pytest fixture providing dummy dataset
            temp_export_dir: Temporary directory for export files
        """
        # Use dummy policy for fast testing
        from getiaction.policies.dummy import DummyConfig

        policy = get_policy("dummy", source="getiaction", config=DummyConfig(action_shape=(2,)))

        # Quick training
        dataset = dummy_dataset(num_samples=8)
        datamodule = DataModule(train_dataset=dataset, train_batch_size=4)
        trainer = Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False, enable_progress_bar=False)
        trainer.fit(policy, datamodule=datamodule)

        # Export
        export_dir = temp_export_dir / backend
        policy.export(export_dir, backend=backend)

        # Verify backend-specific files
        if backend == "openvino":
            assert (export_dir / "dummy.xml").exists()
            assert (export_dir / "dummy.bin").exists()
        elif backend == "onnx":
            assert (export_dir / "dummy.onnx").exists()
        elif backend == "torch":
            assert (export_dir / "dummy.pt").exists()
        elif backend == "torch_export_ir":
            assert (export_dir / "dummy.ptir").exists()
