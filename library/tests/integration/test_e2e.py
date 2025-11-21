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
LEROBOT_POLICIES = ["act", "diffusion"]

# Shared fixtures
@pytest.fixture(scope="class")
def trainer() -> Trainer:
    """Create trainer with fast development configuration (class-scoped for reuse)."""
    return Trainer(
        fast_dev_run=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )


class E2ETests:
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

        # Convert to Observation format and extract first sample (preserve batch dim)
        from getiaction.data.lerobot import FormatConverter
        batch_observation = FormatConverter.to_observation(sample_batch)

        # Use Observation indexing to extract first sample
        inference_input = batch_observation[0:1].to_numpy()
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
        """Test numerical consistency between training and inference outputs.

        This is a clean base implementation for first-party policies.
        LeRobot-specific handling is in TestLeRobotPolicies override.
        """
        policy_name = trained_policy.__class__.__name__.lower()
        export_dir = tmp_path / f"{policy_name}_{backend}"

        # Get batch and convert to Observation
        from getiaction.data.lerobot import FormatConverter
        sample_batch = next(iter(datamodule.train_dataloader()))
        batch_observation = FormatConverter.to_observation(sample_batch)

        # Extract first sample using Observation indexing
        single_observation = batch_observation[0:1].to("cpu")

        # Get training output
        torch.manual_seed(42)
        trained_policy.eval()
        trained_policy.reset()
        with torch.no_grad():
            train_output: torch.Tensor = trained_policy.select_action(single_observation)

        # Export model
        trained_policy.export(export_dir, backend)

        # Load exported model and run inference
        inference_model = InferenceModel.load(export_dir)
        inference_input = single_observation.to_numpy()

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        inference_model.reset()
        inference_output: torch.Tensor = inference_model.select_action(inference_input)

        # Extract first action and handle chunked outputs
        train_action: torch.Tensor = train_output[0].cpu()
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
class TestFirstPartyPolicies(E2ETests):
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
class TestLeRobotPolicies(E2ETests):
    """E2E tests for LeRobot policies."""

    # Skip torch_export_ir: LeRobot calls .eval() during forward pass, unsupported by torch.export
    @pytest.mark.parametrize("backend", ["openvino", "onnx", "torch"])
    def test_export_to_backend(self, trained_policy: Policy, backend: str, tmp_path: Path) -> None:
        """Override to skip torch_export_ir (LeRobot calls .eval() in forward, incompatible with torch.export)."""
        return super().test_export_to_backend(trained_policy, backend, tmp_path)

    @pytest.mark.parametrize("backend", ["openvino", "onnx"])
    def test_inference_with_exported_model(
        self, trained_policy: Policy, backend: str, datamodule: LeRobotDataModule, tmp_path: Path
    ) -> None:
        """Override to skip torch_export_ir (no export = no inference test)."""
        return super().test_inference_with_exported_model(trained_policy, backend, datamodule, tmp_path)

    @pytest.mark.parametrize("backend", ["openvino", "onnx"])
    def test_numerical_consistency_training_vs_inference(
        self,
        trained_policy: Policy,
        backend: str,
        datamodule: LeRobotDataModule,
        tmp_path: Path,
    ) -> None:
        """Override with LeRobot-specific handling: temporal dimensions, preprocessor/postprocessor, scheduler."""
        policy_name = trained_policy.__class__.__name__.lower()

        # Skip Diffusion policies for ONNX/OpenVINO (random noise baked into exported graph)
        if policy_name == "diffusion" and backend in {"openvino", "onnx"}:
            pytest.skip(f"Diffusion numerical consistency not supported for {backend}: random noise baked into graph")

        export_dir = tmp_path / f"{policy_name}_{backend}"

        # Get batch and convert to Observation
        from getiaction.data.lerobot import FormatConverter
        sample_batch = next(iter(datamodule.train_dataloader()))
        batch_observation = FormatConverter.to_observation(sample_batch).to("cpu")

        # Extract first sample, handling temporal dimension for LeRobot policies
        # LeRobot Diffusion: observation shape [B, T, ...], need [1, T, ...] not [1, ...]
        state_single = None
        if batch_observation.state is not None and torch.is_tensor(batch_observation.state):
            if batch_observation.state.ndim == 3:  # [batch, n_obs_steps, state_dim]
                state_single = batch_observation.state[0:1, -1, ...]  # [1, state_dim]
            else:
                state_single = batch_observation.state[0:1]
        else:
            state_single = batch_observation.state

        images_single = None
        if batch_observation.images is not None:
            if isinstance(batch_observation.images, dict):
                images_single = {}
                for k, v in batch_observation.images.items():
                    if v.ndim == 5:  # [batch, n_obs_steps, C, H, W]
                        images_single[k] = v[0:1, -1, ...]  # [1, C, H, W]
                    else:
                        images_single[k] = v[0:1]
            else:
                if batch_observation.images.ndim == 5:
                    images_single = batch_observation.images[0:1, -1, ...]
                else:
                    images_single = batch_observation.images[0:1]

        single_observation = Observation(state=state_single, images=images_single)

        # Move preprocessor/postprocessor to CPU (LeRobot requirement)
        if hasattr(trained_policy, "_move_processor_steps_to_device"):
            trained_policy._move_processor_steps_to_device("cpu")  # type: ignore[attr-defined]

        # Move model to CPU for export
        trained_policy = trained_policy.cpu()

        # For Diffusion: move scheduler to CPU and set num_inference_steps=1 to match export
        original_num_inference_steps = None
        if hasattr(trained_policy, "_lerobot_policy") and hasattr(trained_policy._lerobot_policy, "diffusion"):  # type: ignore[attr-defined]
            original_num_inference_steps = trained_policy._lerobot_policy.diffusion.num_inference_steps  # type: ignore[attr-defined]
            trained_policy._lerobot_policy.diffusion.num_inference_steps = 1  # type: ignore[attr-defined]

            # Move scheduler tensors to CPU
            scheduler = trained_policy._lerobot_policy.diffusion.noise_scheduler  # type: ignore[attr-defined]
            for attr_name in dir(scheduler):
                if not attr_name.startswith("_"):
                    attr = getattr(scheduler, attr_name, None)
                    if isinstance(attr, torch.Tensor):
                        setattr(scheduler, attr_name, attr.cpu())

        # Get training output
        torch.manual_seed(42)
        trained_policy.eval()
        trained_policy.reset()
        with torch.no_grad():
            train_output: torch.Tensor = trained_policy.select_action(single_observation)

        # Restore num_inference_steps if changed
        if original_num_inference_steps is not None:
            trained_policy._lerobot_policy.diffusion.num_inference_steps = original_num_inference_steps  # type: ignore[attr-defined]

        # Export model
        trained_policy.export(export_dir, backend)

        # Load exported model and run inference
        inference_model = InferenceModel.load(export_dir)
        inference_input = single_observation.to_numpy()

        # Debug OpenVINO inputs/outputs
        import numpy as np  # noqa: PLC0415
        if backend == "openvino":
            print(f"\n=== DEBUG: OpenVINO Inference (policy={policy_name}) ===")
            if hasattr(inference_input, "state") and inference_input.state is not None:
                state_val = inference_input.state
                if isinstance(state_val, np.ndarray):
                    print(f"state: shape={state_val.shape}, min={np.min(state_val):.3f}, max={np.max(state_val):.3f}")
            if hasattr(inference_input, "images") and inference_input.images is not None:
                if isinstance(inference_input.images, dict):
                    for k, v in inference_input.images.items():
                        if isinstance(v, np.ndarray):
                            print(f"images[{k}]: shape={v.shape}, min={np.min(v):.3f}, max={np.max(v):.3f}")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        inference_model.reset()
        inference_output: torch.Tensor = inference_model.select_action(inference_input)

        if backend == "openvino":
            print(f"inference_output: shape={inference_output.shape}")
            if isinstance(inference_output, np.ndarray) and np.isnan(inference_output).any():
                print(f"  WARNING: Output contains {np.isnan(inference_output).sum()} NaN values!")
            elif torch.is_tensor(inference_output) and torch.isnan(inference_output).any():
                print(f"  WARNING: Output contains {torch.isnan(inference_output).sum()} NaN values!")

        # Extract first action and handle chunked outputs
        train_action: torch.Tensor = train_output[0].cpu()
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

    # Override trained_policy with function scope for numerical consistency tests
    # This avoids fixture corruption when modifying policy device/state
    @pytest.fixture(scope="function")
    def trained_policy(self, policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> Policy:
        """Train a fresh policy for each test (function-scoped for numerical consistency)."""
        trainer.fit(policy, datamodule=datamodule)
        return policy

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
