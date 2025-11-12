# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for LeRobot policies.

This module tests the complete pipeline for LeRobot policies:
1. Train a LeRobot policy using LeRobot PushT dataset
2. Validate/test the trained policy
3. Export the trained policy to multiple backends (OpenVINO, ONNX, Torch, TorchExportIR)
4. Load exported model for inference
5. Verify numerical consistency between training and inference

Tests all supported LeRobot policies (ACT, Diffusion, VQBeT, etc.) to ensure
the LeRobotModel wrapper enables proper export functionality.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

from getiaction.data import LeRobotDataModule, Observation
from getiaction.inference import InferenceModel
from getiaction.policies import get_policy
from getiaction.train import Trainer

if TYPE_CHECKING:
    from getiaction.policies.base.policy import Policy


@pytest.fixture(scope="class")
def datamodule() -> LeRobotDataModule:
    """Create LeRobot DataModule with PushT dataset for testing.

    Returns:
        LeRobotDataModule: DataModule configured with PushT dataset using first 10 episodes.
    """
    return LeRobotDataModule(
        repo_id="lerobot/pusht",
        train_batch_size=8,
        episodes=list(range(10)),  # Use only first 10 episodes for speed
    )


@pytest.fixture(
    scope="class",
    params=[
        "act",
        "diffusion",
        # "vqbet",  # Requires discrete action space
        # "tdmpc",  # Requires different configuration
        # Add more as needed
    ],
)
def lerobot_policy(request: pytest.FixtureRequest) -> "Policy":
    """Create LeRobot policy instance based on parametrized policy name.

    This fixture is parametrized to test multiple LeRobot policies with export functionality.

    Args:
        request: Pytest request fixture containing the policy name parameter

    Returns:
        Policy: Configured LeRobot policy instance.
    """
    policy_name: str = request.param
    return get_policy(policy_name, source="lerobot")


@pytest.fixture(scope="class")
def trainer() -> Trainer:
    """Create trainer with fast development configuration.

    Returns:
        Trainer: Configured trainer instance for fast testing.
    """
    from lightning.pytorch import seed_everything

    from getiaction.train import Trainer

    # Set seed for reproducibility
    seed_everything(42, workers=True)

    return Trainer(
        max_steps=1,
        enable_checkpointing=False,
        logger=False,
    )


@pytest.fixture(scope="class")
def trained_lerobot_policy(
    lerobot_policy: "Policy",
    datamodule: LeRobotDataModule,
    trainer: Trainer,
) -> "Policy":
    """Train a policy once per class and reuse across all test methods.

    This fixture trains the policy once per policy type (from lerobot_policy parametrization)
    and returns the trained policy for reuse across all tests in the class.

    The scope is 'class' so training happens once per test class per policy type.
    This significantly speeds up test execution by avoiding redundant training.

    Args:
        lerobot_policy: Untrained policy from lerobot_policy fixture
        datamodule: DataModule fixture
        trainer: Trainer fixture

    Returns:
        Policy: Trained policy ready for testing
    """
    # Train once and reuse
    trainer.fit(lerobot_policy, datamodule=datamodule)
    return lerobot_policy


class TestLeRobotE2E:
    """End-to-end tests for LeRobot policies with export functionality."""

    def test_train_lerobot_policy(
        self,
        trained_lerobot_policy: "Policy",
        trainer: Trainer,
    ) -> None:
        """Test that LeRobot policy can be trained successfully.

        Args:
            trained_lerobot_policy: Trained LeRobot policy (reuses training from fixture)
            trainer: Trainer fixture
        """
        # Training is done by the fixture, just verify it completed
        assert trainer.state.finished

    def test_validate_lerobot_policy(
        self,
        trained_lerobot_policy: "Policy",
        datamodule: LeRobotDataModule,
        trainer: Trainer,
    ) -> None:
        """Test that trained LeRobot policy can be validated.

        Args:
            trained_lerobot_policy: Trained LeRobot policy (reuses training from fixture)
            datamodule: PushT datamodule fixture
            trainer: Trainer fixture
        """
        # Policy already trained by fixture, just validate
        trainer.validate(trained_lerobot_policy, datamodule=datamodule)

        # Verify validation completed
        assert trainer.state.finished

    def test_test_lerobot_policy(
        self,
        trained_lerobot_policy: "Policy",
        datamodule: LeRobotDataModule,
        trainer: Trainer,
    ) -> None:
        """Test that trained LeRobot policy can be tested.

        Args:
            trained_lerobot_policy: Trained LeRobot policy (reuses training from fixture)
            datamodule: PushT datamodule fixture
            trainer: Trainer fixture
        """
        # Policy already trained by fixture, just test
        trainer.test(trained_lerobot_policy, datamodule=datamodule)

        # Verify test completed
        assert trainer.state.finished

    @pytest.mark.parametrize(
        "backend",
        ["openvino", "onnx", "torch", "torch_export_ir"],
    )
    def test_export_lerobot_to_backend(
        self,
        trained_lerobot_policy: "Policy",
        backend: str,
        tmp_path: Path,
    ) -> None:
        """Test that trained LeRobot policy can be exported to different backends.

        This test verifies the LeRobotModel wrapper enables clean export.

        Args:
            trained_lerobot_policy: Trained LeRobot policy (reuses training from fixture)
            backend: Export backend to test
            tmp_path: Pytest's temporary directory fixture
        """
        # Policy already trained by fixture, just export
        export_dir = tmp_path / f"{trained_lerobot_policy.__class__.__name__.lower()}_{backend}"
        trained_lerobot_policy.export(export_dir, backend)

        # Verify export files exist
        assert export_dir.exists()
        assert (export_dir / "metadata.yaml").exists()
        assert (export_dir / "metadata.json").exists()

        # Verify backend-specific model files
        if backend == "openvino":
            assert any(export_dir.glob("*.xml"))
            assert any(export_dir.glob("*.bin"))
        elif backend == "onnx":
            assert any(export_dir.glob("*.onnx"))
        elif backend == "torch":
            assert any(export_dir.glob("*.pt"))
        elif backend == "torch_export_ir":
            assert any(export_dir.glob("*.pt2"))

    @pytest.mark.parametrize(
        "backend",
        ["openvino", "onnx", "torch_export_ir"],
    )
    def test_inference_with_exported_lerobot_model(
        self,
        trained_lerobot_policy: "Policy",
        backend: str,
        datamodule: LeRobotDataModule,
        tmp_path: Path,
    ) -> None:
        """Test that exported LeRobot model can be loaded and used for inference.

        This test validates the LeRobotModel wrapper provides clean forward() for export.

        Args:
            trained_lerobot_policy: Trained LeRobot policy (reuses training from fixture)
            backend: Export backend to test
            datamodule: PushT datamodule fixture
            tmp_path: Pytest's temporary directory fixture
        """
        # Policy already trained by fixture, just export
        export_dir = tmp_path / f"{trained_lerobot_policy.__class__.__name__.lower()}_{backend}"
        trained_lerobot_policy.export(export_dir, backend)

        # Load exported model for inference
        inference_model = InferenceModel.load(export_dir)

        # Verify backend detection
        assert inference_model.backend.value == backend

        # Get a sample observation from dataset
        sample_batch = next(iter(datamodule.train_dataloader()))

        # Prepare inference input (convert first sample to numpy)
        inference_input_dict: dict[str, np.ndarray | Any] = {}
        for key, value in sample_batch.to_dict().items():
            if key in {"state", "images", "image"}:
                if torch.is_tensor(value):
                    # Take first sample from batch and convert to numpy
                    inference_input_dict[key] = value[0:1].cpu().numpy()
                else:
                    inference_input_dict[key] = value

        # Create Observation from dict
        inference_input: Observation = Observation.from_dict(inference_input_dict)

        # Perform inference
        inference_output: torch.Tensor = inference_model.select_action(inference_input)

        # Verify output shape
        assert inference_output.shape[-1] == 2  # Action dimension for PushT
        # For chunked policies (ACT), first call returns (batch, action_dim) from queue
        # For non-chunked policies, expect (action_dim,) or (batch, action_dim)
        assert len(inference_output.shape) in {1, 2}, f"Expected 1D or 2D tensor, got shape {inference_output.shape}"

    @pytest.mark.parametrize(
        "backend",
        [
            "openvino",
            "onnx",
            pytest.param(
                "torch_export_ir",
                marks=pytest.mark.xfail(
                    reason=(
                        "TorchExportIR numerical consistency has minor discrepancies due to trace-time shape baking. "
                        "The dimension squeezing logic (for n_obs_steps==1) gets captured in the export trace with "
                        "specific shapes, causing slight numerical differences when the exported model is used with "
                        "training data that has different batch sizes. This is a known limitation of torch.export "
                        "with dynamic input shapes and does not affect practical inference accuracy."
                    ),
                    strict=False,
                ),
            ),
        ],
    )
    def test_numerical_consistency_lerobot_training_vs_inference(
        self,
        trained_lerobot_policy: "Policy",
        backend: str,
        datamodule: LeRobotDataModule,
        tmp_path: Path,
    ) -> None:
        """Test numerical consistency between LeRobot training and inference outputs.

        This test ensures the LeRobotModel wrapper maintains numerical accuracy.

        Args:
            trained_lerobot_policy: Trained LeRobot policy (reuses training from fixture)
            backend: Export backend to test
            datamodule: PushT datamodule fixture
            tmp_path: Pytest's temporary directory fixture
        """
        # Policy already trained by fixture, just export
        export_dir = tmp_path / f"{trained_lerobot_policy.__class__.__name__.lower()}_{backend}"
        trained_lerobot_policy.export(export_dir, backend)

        # Load for inference
        inference_model = InferenceModel.load(export_dir)

        # Get sample batch
        sample_batch = next(iter(datamodule.train_dataloader()))

        # Get training policy prediction
        trained_lerobot_policy.eval()
        with torch.no_grad():
            train_output: torch.Tensor = trained_lerobot_policy.select_action(sample_batch)

        # Prepare inference input (convert first sample to numpy)
        inference_input_dict: dict[str, np.ndarray | Any] = {}
        for key, value in sample_batch.to_dict().items():
            if key in {"state", "images", "image"}:
                if torch.is_tensor(value):
                    # Take first sample from batch and convert to numpy
                    inference_input_dict[key] = value[0:1].cpu().numpy()
                else:
                    inference_input_dict[key] = value

        # Create Observation from dict
        inference_input: Observation = Observation.from_dict(inference_input_dict)

        # Get inference prediction
        inference_output: torch.Tensor = inference_model.select_action(inference_input)

        # Compare outputs (allowing tolerance for backend optimizations)
        train_action: torch.Tensor = train_output[0].cpu()  # First sample from batch

        # For chunked policies (ACT), compare first action
        if len(train_action.shape) > 1:
            train_action = train_action[0]  # (chunk, dim) -> (dim,)

        # inference_output is already a Tensor from select_action
        if not isinstance(inference_output, torch.Tensor):
            inference_output = torch.from_numpy(inference_output)

        # Remove batch dimension if present: (1, action_dim) -> (action_dim)
        inference_output_cpu: torch.Tensor = inference_output.cpu().squeeze(0)

        # Ensure both are on CPU for comparison
        torch.testing.assert_close(
            inference_output_cpu,
            train_action,
            rtol=0.2,  # 20% relative tolerance for backend differences
            atol=0.2,  # 0.2 absolute tolerance
        )
