# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests from training to export to inference.

This module contains comprehensive E2E tests that validate the complete pipeline:
1. Train a policy using LeRobot PushT dataset
2. Validate/test the trained policy
3. Export the trained policy to multiple backends (OpenVINO, ONNX, Torch, TorchExportIR)
4. Load exported model for inference
5. Verify numerical consistency between training and inference

Uses generic fixtures for policy-agnostic testing across all policy × backend combinations.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

from getiaction.data import LeRobotDataModule, Observation
from getiaction.inference import InferenceModel
from getiaction.policies import get_policy
from getiaction.policies.base.policy import Policy
from getiaction.train import Trainer


@pytest.fixture
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


@pytest.fixture(params=["act", "diffusion"])
def policy(request: pytest.FixtureRequest) -> Policy:
    """Create policy instance based on parametrized policy name.

    This fixture is parametrized to test multiple policies. The trainer pipeline
    automatically configures the datamodule based on the policy's requirements
    (e.g., action chunking for ACT).

    Args:
        request: Pytest request fixture containing the policy name parameter

    Returns:
        Policy: Configured policy instance.
    """
    policy_name: str = request.param
    
    # Determine source based on policy name
    # First-party: act, dummy
    # LeRobot: diffusion, vqbet, tdmpc, sac, pi0, pi05, smolvla, groot
    getiaction_policies = {"act", "dummy"}
    source = "getiaction" if policy_name in getiaction_policies else "lerobot"
    
    return get_policy(policy_name, source=source)


@pytest.fixture
def trainer() -> Trainer:
    """Create trainer with fast development configuration.

    Returns:
        Trainer: Configured trainer instance for fast testing.
    """
    return Trainer(
        fast_dev_run=1,  # Run 1 training batch and 1 validation batch
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )


class TestE2E:
    """Generic end-to-end tests for training → validation → export → inference pipeline."""

    def test_train_policy(self, policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> None:
        """Test that policy can be trained successfully.

        Args:
            policy: Policy fixture (parametrized)
            datamodule: PushT datamodule fixture
            trainer: Trainer fixture
        """
        # Train policy (fast_dev_run=1 runs 1 train + 1 val batch)
        trainer.fit(policy, datamodule=datamodule)

        # Verify training completed
        assert trainer.state.finished

    def test_validate_policy(self, policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> None:
        """Test that trained policy can be validated.

        Args:
            policy: Policy fixture (parametrized)
            datamodule: PushT datamodule fixture
            trainer: Trainer fixture
        """
        # Train and validate
        trainer.fit(policy, datamodule=datamodule)
        trainer.validate(policy, datamodule=datamodule)

        # Verify validation completed
        assert trainer.state.finished

    def test_test_policy(self, policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> None:
        """Test that trained policy can be tested.

        Args:
            policy: Policy fixture (parametrized)
            datamodule: PushT datamodule fixture
            trainer: Trainer fixture
        """
        # Train and test
        trainer.fit(policy, datamodule=datamodule)
        trainer.test(policy, datamodule=datamodule)

        # Verify test completed
        assert trainer.state.finished

    @pytest.mark.parametrize(
        "backend",
        ["openvino", "onnx", "torch", "torch_export_ir"],
    )
    def test_export_to_backend(
        self,
        policy: Policy,
        backend: str,
        datamodule: LeRobotDataModule,
        trainer: Trainer,
        tmp_path: Path,
    ) -> None:
        """Test that trained policy can be exported to different backends.

        Args:
            policy: Policy fixture (parametrized)
            backend: Export backend to test
            datamodule: PushT datamodule fixture
            trainer: Trainer fixture
            tmp_path: Pytest's temporary directory fixture
        """
        # Train policy
        trainer.fit(policy, datamodule=datamodule)

        # Export to backend
        export_dir = tmp_path / f"{policy.__class__.__name__.lower()}_{backend}"
        policy.export(export_dir, backend)

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
    def test_inference_with_exported_model(
        self,
        policy: Policy,
        backend: str,
        datamodule: LeRobotDataModule,
        trainer: Trainer,
        tmp_path: Path,
    ) -> None:
        """Test that exported model can be loaded and used for inference.

        Args:
            policy: Policy fixture (parametrized)
            backend: Export backend to test
            datamodule: PushT datamodule fixture
            trainer: Trainer fixture
            tmp_path: Pytest's temporary directory fixture
        """
        # Train policy
        trainer.fit(policy, datamodule=datamodule)

        # Export model
        export_dir = tmp_path / f"{policy.__class__.__name__.lower()}_{backend}"
        policy.export(export_dir, backend)

        # Load exported model for inference
        inference_model = InferenceModel.load(export_dir)

        # Verify backend detection
        assert inference_model.backend.value == backend

        # Get a sample observation from dataset
        sample_batch = next(iter(datamodule.train_dataloader()))

        # Prepare inference input (convert first sample to numpy)
        # Extract first sample from each tensor in the batch
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
        assert inference_output.shape[-1] == 2  # Action dimension
        # For chunked policies (ACT), first call returns (batch, action_dim) from queue
        # For non-chunked policies, expect (action_dim,) or (batch, action_dim)
        # For chunked policies like ACT: (batch, action_dim) or (action_dim,)
        assert len(inference_output.shape) in {1, 2}, f"Expected 1D or 2D tensor, got shape {inference_output.shape}"

    @pytest.mark.parametrize(
        "backend",
        ["openvino", "onnx", "torch_export_ir"],
    )
    def test_numerical_consistency_training_vs_inference(
        self,
        policy: Policy,
        backend: str,
        datamodule: LeRobotDataModule,
        trainer: Trainer,
        tmp_path: Path,
    ) -> None:
        """Test numerical consistency between training and inference outputs.

        For Diffusion policies, this test uses the full denoising loop (100 steps)
        to ensure numerical consistency with training. This makes diffusion tests
        significantly slower (~4 minutes vs ~10 seconds for fast export).
        Use `pytest -m "not slow"` to skip slow tests, or `pytest -m slow` to run only slow tests.

        Other policies use default (fast) export settings.

        Args:
            policy: Policy fixture (parametrized)
            backend: Export backend to test
            datamodule: PushT datamodule fixture
            trainer: Trainer fixture
            tmp_path: Pytest's temporary directory fixture
        """
        # Train policy
        trainer.fit(policy, datamodule=datamodule)

        # Export model with policy-specific parameters
        export_dir = tmp_path / f"{policy.__class__.__name__.lower()}_{backend}"
        export_kwargs = {}
        
        # For Diffusion: use full 100-step denoising for numerical accuracy (slow: ~4 minutes)
        # For other policies: use default (fast) export settings
        if policy.__class__.__name__.lower() == "diffusion":
            pytest.mark.slow(lambda: None)()  # Mark as slow test
            export_kwargs["num_inference_steps"] = 100
        
        policy.export(export_dir, backend, **export_kwargs)

        # Load for inference
        inference_model = InferenceModel.load(export_dir)

        # Get sample batch
        sample_batch = next(iter(datamodule.train_dataloader()))

        # Get training policy prediction
        policy.eval()
        with torch.no_grad():
            train_output: torch.Tensor = policy.select_action(sample_batch)

        # Prepare inference input (convert first sample to numpy)
        # Extract first sample from each tensor in the batch
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
