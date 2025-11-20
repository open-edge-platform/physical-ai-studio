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
from typing import Any

import numpy as np
import pytest
import torch

from getiaction.data import LeRobotDataModule, Observation
from getiaction.inference import InferenceModel
from getiaction.policies import get_policy
from getiaction.policies.base.policy import Policy
from getiaction.train import Trainer

# Policy names for parametrization
FIRST_PARTY_POLICIES = ["act"]
LEROBOT_POLICIES = ["diffusion"]

# Shared fixtures
@pytest.fixture(scope="class")
def trainer() -> Trainer:
    """Create trainer with fast development configuration (class-scoped for reuse)."""
    return Trainer(
        fast_dev_run=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator="gpu",
    )


class BaseE2ETests:
    """Base class with common E2E test methods for all policies."""

    def test_train_policy(self, trained_policy: Policy, trainer: Trainer) -> None:
        """Test that policy was trained successfully."""
        assert trainer.state.finished

    def test_validate_policy(self, trained_policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> None:
        """Test that trained policy can be validated."""
        trainer.validate(trained_policy, datamodule=datamodule)
        assert trainer.state.finished

    def test_test_policy(self, trained_policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> None:
        """Test that trained policy can be tested."""
        trainer.test(trained_policy, datamodule=datamodule)
        assert trainer.state.finished

    @pytest.mark.parametrize("backend", ["openvino", "onnx", "torch", "torch_export_ir"])
    def test_export_to_backend(self, trained_policy: Policy, backend: str, tmp_path: Path) -> None:
        """Test that trained policy can be exported to different backends."""
        export_dir = tmp_path / f"{trained_policy.__class__.__name__.lower()}_{backend}"
        trained_policy.export(export_dir, backend)

        assert export_dir.exists()
        assert (export_dir / "metadata.yaml").exists()

        if backend == "openvino":
            assert any(export_dir.glob("*.xml"))
            assert any(export_dir.glob("*.bin"))
        elif backend == "onnx":
            assert any(export_dir.glob("*.onnx"))
        elif backend == "torch":
            assert any(export_dir.glob("*.pt"))
        elif backend == "torch_export_ir":
            assert any(export_dir.glob("*.pt2"))

    @pytest.mark.parametrize("backend", ["openvino", "onnx", "torch_export_ir"])
    def test_inference_with_exported_model(
        self,
        trained_policy: Policy,
        backend: str,
        datamodule: LeRobotDataModule,
        tmp_path: Path,
    ) -> None:
        """Test that exported model can be loaded and used for inference."""
        export_dir = tmp_path / f"{trained_policy.__class__.__name__.lower()}_{backend}"
        trained_policy.export(export_dir, backend)

        inference_model = InferenceModel.load(export_dir)
        assert inference_model.backend.value == backend

        sample_batch = next(iter(datamodule.train_dataloader()))

        # Convert to Observation format using FormatConverter (handles both dict and Observation)
        from getiaction.data.lerobot import FormatConverter
        import numpy as np
        batch_observation = FormatConverter.to_observation(sample_batch)

        # Take first sample and convert to numpy for inference
        state_np = batch_observation.state[0:1].cpu().numpy() if batch_observation.state is not None and torch.is_tensor(batch_observation.state) else None
        images_np = None
        if batch_observation.images is not None:
            if isinstance(batch_observation.images, dict):
                images_np = {k: v[0:1].cpu().numpy() if torch.is_tensor(v) else v for k, v in batch_observation.images.items()}
            elif torch.is_tensor(batch_observation.images):
                images_np = batch_observation.images[0:1].cpu().numpy()

        inference_input = Observation(state=state_np, images=images_np)
        inference_output: torch.Tensor = inference_model.select_action(inference_input)

        assert inference_output.shape[-1] == 2
        assert len(inference_output.shape) in {1, 2, 3}, f"Expected 1-3D tensor, got {inference_output.shape}"

    @pytest.mark.parametrize("backend", ["openvino", "onnx", "torch_export_ir"])
    def test_numerical_consistency_training_vs_inference(
        self,
        trained_policy: Policy,
        backend: str,
        datamodule: LeRobotDataModule,
        tmp_path: Path,
    ) -> None:
        """Test numerical consistency between training and inference outputs."""
        export_dir = tmp_path / f"{trained_policy.__class__.__name__.lower()}_{backend}"

        # Export (num_inference_steps is a runtime parameter, not export parameter)
        trained_policy.export(export_dir, backend)

        inference_model = InferenceModel.load(export_dir)
        sample_batch = next(iter(datamodule.train_dataloader()))

        trained_policy.eval()
        with torch.no_grad():
            train_output: torch.Tensor = trained_policy.select_action(sample_batch)

        # Convert to Observation format using FormatConverter (handles both dict and Observation)
        from getiaction.data.lerobot import FormatConverter
        import numpy as np
        batch_observation = FormatConverter.to_observation(sample_batch)

        # Take first sample and convert to numpy for inference
        state_np = batch_observation.state[0:1].cpu().numpy() if batch_observation.state is not None and torch.is_tensor(batch_observation.state) else None
        images_np = None
        if batch_observation.images is not None:
            if isinstance(batch_observation.images, dict):
                images_np = {k: v[0:1].cpu().numpy() if torch.is_tensor(v) else v for k, v in batch_observation.images.items()}
            elif torch.is_tensor(batch_observation.images):
                images_np = batch_observation.images[0:1].cpu().numpy()

        inference_input = Observation(state=state_np, images=images_np)
        inference_output: torch.Tensor = inference_model.select_action(inference_input)

        # Extract first action from batch and handle chunked outputs
        train_action: torch.Tensor = train_output[0].cpu()  # [chunk_size, action_dim] or [action_dim]

        # For chunked policies, take first action from chunk
        if len(train_action.shape) > 1:
            train_action = train_action[0]  # [action_dim]

        # Convert inference output to tensor if needed
        if not isinstance(inference_output, torch.Tensor):
            inference_output = torch.from_numpy(inference_output)

        # Squeeze batch dimension and handle chunked inference output
        inference_output_cpu: torch.Tensor = inference_output.cpu().squeeze(0)
        if len(inference_output_cpu.shape) > 1:
            inference_output_cpu = inference_output_cpu[0]  # [action_dim]

        torch.testing.assert_close(
            inference_output_cpu,
            train_action,
            rtol=0.2,
            atol=0.2,
        )


@pytest.mark.parametrize("policy_name", FIRST_PARTY_POLICIES, indirect=True)
class TestFirstPartyPolicies(BaseE2ETests):
    """E2E tests for first-party policies (getiaction)."""

    @pytest.fixture(scope="class")
    def datamodule(self) -> LeRobotDataModule:
        """Create datamodule for first-party policies (getiaction format)."""
        return LeRobotDataModule(
            repo_id="lerobot/pusht",
            train_batch_size=8,
            episodes=list(range(10)),
        )

    @pytest.fixture(scope="class")
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        """Extract policy name from parametrize."""
        return request.param

    @pytest.fixture(scope="class")
    def policy(self, policy_name: str) -> Policy:
        """Create first-party policy instance."""
        return get_policy(policy_name, source="getiaction")

    @pytest.fixture(scope="class")
    def trained_policy(self, policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> Policy:
        """Train policy once and reuse across all tests."""
        trainer.fit(policy, datamodule=datamodule)
        return policy


@pytest.mark.parametrize("policy_name", LEROBOT_POLICIES, indirect=True)
class TestLeRobotPolicies(BaseE2ETests):
    """E2E tests for LeRobot policies."""

    @pytest.fixture(scope="class")
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        """Extract policy name from parametrize."""
        return request.param

    @pytest.fixture(scope="class")
    def datamodule(self, policy_name: str) -> LeRobotDataModule:
        """Create datamodule for LeRobot policies (lerobot format with delta timestamps)."""
        repo_id = "lerobot/pusht"
        fps = 10

        # Policy-specific configurations
        policy_configs = {
            "act": {
                "action_delta_indices": list(range(100)),  # chunk_size=100
            },
            "diffusion": {
                "observation_delta_indices": [-1, 0],  # n_obs_steps=2
                "action_delta_indices": list(range(-1, 15)),  # horizon=16
            },
        }

        config = {
            "repo_id": repo_id,
            "train_batch_size": 8,
            "episodes": list(range(10)),
            "data_format": "lerobot",
        }

        # Add delta timestamps if configured
        if policy_name in policy_configs:
            policy_cfg = policy_configs[policy_name]
            delta_timestamps = {}

            if "observation_delta_indices" in policy_cfg:
                obs_indices = policy_cfg["observation_delta_indices"]
                delta_timestamps["observation.image"] = [i / fps for i in obs_indices]
                delta_timestamps["observation.state"] = [i / fps for i in obs_indices]

            if "action_delta_indices" in policy_cfg:
                action_indices = policy_cfg["action_delta_indices"]
                delta_timestamps["action"] = [i / fps for i in action_indices]

            config["delta_timestamps"] = delta_timestamps

        return LeRobotDataModule(**config)  # type: ignore[arg-type]

    @pytest.fixture(scope="class")
    def policy(self, policy_name: str) -> Policy:
        """Create LeRobot policy instance."""
        # Fast config for integration tests
        policy_kwargs = {}
        if policy_name == "diffusion":
            policy_kwargs = {
                "num_train_timesteps": 10,
                "num_inference_steps": 5,
            }

        return get_policy(policy_name, source="lerobot", **policy_kwargs)

    @pytest.fixture(scope="class")
    def trained_policy(self, policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> Policy:
        """Train policy once and reuse across all tests."""
        trainer.fit(policy, datamodule=datamodule)
        return policy
