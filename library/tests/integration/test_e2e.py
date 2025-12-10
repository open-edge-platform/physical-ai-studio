# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests from training to export to inference.

This module contains comprehensive E2E tests that validate the complete pipeline:
1. Train a policy using LeRobot PushT dataset
2. Validate/test the trained policy
3. Export the trained policy to multiple backends (OpenVINO, ONNX, Torch, TorchExportIR) - optional
4. Load exported model for inference - optional
5. Verify numerical consistency between training and inference - optional

Design:
- CoreE2ETests: Required tests (train/val/test) - all policies must support
- ExportE2ETests: Optional tests (export/inference) - only run if policy supports export
- Policies declare export support by inheriting from Export mixin
"""

from pathlib import Path

import pytest
import torch

from getiaction.data import LeRobotDataModule, Observation
from getiaction.export import Export
from getiaction.inference import InferenceModel
from getiaction.policies import get_policy
from getiaction.policies.base.policy import Policy
from getiaction.train import Trainer

# Export backend constants
# TODO: Add "torch" when torch export backend PR is merged
# EXPORT_BACKENDS = ["openvino", "onnx", "torch", "torch_export_ir"]
EXPORT_BACKENDS = ["openvino", "onnx", "torch_export_ir"]

# Policy names for parametrization
# Default: policies without export support (core tests only)
FIRST_PARTY_POLICIES = ["pi0"]
# Policies with export support (core + export tests)
FIRST_PARTY_POLICIES_WITH_EXPORT = ["act"]
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


def _supports_export(policy: Policy) -> bool:
    """Check if policy supports export functionality."""
    return isinstance(policy, Export) and hasattr(policy, "export")


# ============================================================================ #
# Core E2E Tests (Required for all policies)                                  #
# ============================================================================ #


class CoreE2ETests:
    """Base class with required E2E test methods for all policies.

    All policies must support training, validation, and testing.
    """

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


# ============================================================================ #
# Export E2E Tests (Optional - only for policies with export support)          #
# ============================================================================ #


class ExportE2ETests:
    """Mixin with optional export tests.

    These tests only run for policies that support export (inherit from Export mixin).
    Tests automatically skip if policy doesn't support export.
    """

    @pytest.mark.parametrize("backend", EXPORT_BACKENDS)
    def test_export_to_backend(self, trained_policy: Policy, backend: str, tmp_path: Path) -> None:
        """Test that trained policy can be exported to different backends."""
        if not _supports_export(trained_policy):
            pytest.skip("Policy doesn't support export")
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

    @pytest.mark.parametrize("backend", EXPORT_BACKENDS)
    def test_inference_with_exported_model(
        self,
        trained_policy: Policy,
        backend: str,
        datamodule: LeRobotDataModule,
        tmp_path: Path,
    ) -> None:
        """Test that exported model can be loaded and used for inference."""
        if not _supports_export(trained_policy):
            pytest.skip("Policy doesn't support export")
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
        inference_output = inference_model.select_action(inference_input)

        assert inference_output.shape[-1] == 2
        assert len(inference_output.shape) in {1, 2, 3}, f"Expected 1-3D tensor, got {inference_output.shape}"

    @pytest.mark.parametrize("backend", EXPORT_BACKENDS)
    def test_numerical_consistency_training_vs_inference(
        self,
        trained_policy: Policy,
        backend: str,
        datamodule: LeRobotDataModule,
        tmp_path: Path,
    ) -> None:
        """Test numerical consistency between training and inference outputs."""
        if not _supports_export(trained_policy):
            pytest.skip("Policy doesn't support export")
        """Test numerical consistency between training and inference outputs.

        This is a clean base implementation for first-party policies.
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
            train_output = trained_policy.select_action(single_observation)

        # Export model
        trained_policy.export(export_dir, backend)

        # Load exported model and run inference
        inference_model = InferenceModel.load(export_dir)
        inference_input = single_observation.to_numpy()

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        inference_model.reset()
        inference_output = inference_model.select_action(inference_input)

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


# ============================================================================ #
# Combined E2E Tests (Core + Export)                                           #
# ============================================================================ #


class E2ETests(CoreE2ETests, ExportE2ETests):
    """Combined E2E tests: core (required) + export (optional).

    Policies that support export inherit from this.
    Policies without export can inherit from CoreE2ETests only.
    """


# ============================================================================ #
# First-Party Policies (with export support)                                    #
# ============================================================================ #


@pytest.mark.parametrize("policy_name", FIRST_PARTY_POLICIES_WITH_EXPORT, indirect=True)
class TestFirstPartyPolicies(E2ETests):
    """E2E tests for first-party policies (getiaction) with export support."""

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


# ============================================================================ #
# LeRobot Policies (core tests only - no export support)                      #
# ============================================================================ #


@pytest.mark.parametrize("policy_name", LEROBOT_POLICIES, indirect=True)
class TestLeRobotPolicies(CoreE2ETests):
    """E2E tests for LeRobot policies (core tests only, export not supported)."""

    @pytest.fixture(scope="class")
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        """Extract policy name from parametrize."""
        return request.param

    @pytest.fixture(scope="function")  # Function scope to avoid state pollution
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

    @pytest.fixture(scope="function")  # Function scope to avoid state pollution between tests
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

    @pytest.fixture(scope="function")
    def trained_policy(self, policy_name: str, datamodule: LeRobotDataModule, trainer: Trainer) -> Policy:
        """Train a fresh policy for each test (function-scoped for numerical consistency)."""
        # Create a NEW policy instance for each test to avoid state pollution
        policy_kwargs = {}
        if policy_name == "diffusion":
            policy_kwargs = {"num_inference_steps": 1}

        policy = get_policy(policy_name, source="lerobot", **policy_kwargs)
        trainer.fit(policy, datamodule=datamodule)
        return policy


# ============================================================================ #
# Pi0 Policies (core tests only - no export support yet)                       #
# ============================================================================ #


@pytest.mark.slow
@pytest.mark.requires_download
@pytest.mark.parametrize("policy_name", FIRST_PARTY_POLICIES, indirect=True)
class TestPi0Policies(CoreE2ETests):
    """E2E tests for Pi0/Pi0.5 policies (core tests only, export not yet supported).

    Pi0 uses flow matching with PaliGemma + Gemma backbone. Export support
    requires additional work due to the Euler integration loop.

    Note: These tests require downloading PaliGemma-3B (~10GB) even with "gemma_300m" variant.
    The variant name refers to action expert size, not the VLM backbone.
    Run with: pytest -m "slow and requires_download" to include these tests.
    Skip with: pytest -m "not requires_download" to exclude them.

    To add export support:
    1. Make Pi0 inherit from Export mixin
    2. Change this class to inherit from E2ETests instead of CoreE2ETests
    3. Export tests will automatically run (no skip decorators needed)
    """

    @pytest.fixture(scope="class")
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        """Extract policy name from parametrize."""
        return request.param

    @pytest.fixture(scope="class")
    def datamodule(self) -> LeRobotDataModule:
        """Create datamodule for Pi0 policies."""
        return LeRobotDataModule(
            repo_id="lerobot/pusht",
            train_batch_size=2,
            episodes=list(range(5)),
            data_format="lerobot",
            delta_timestamps={
                "action": [i / 10.0 for i in range(50)],  # chunk_size=50
            },
        )

    @pytest.fixture(scope="class")
    def policy(self, policy_name: str) -> Policy:
        """Create Pi0 policy with lightweight configuration."""
        return get_policy(
            policy_name,
            source="getiaction",
            paligemma_variant="gemma_300m",
            action_expert_variant="gemma_300m",
            chunk_size=50,
            n_action_steps=10,
            num_inference_steps=2,
            tune_paligemma=False,
            tune_action_expert=False,
            tune_vision_encoder=False,
            gradient_checkpointing=False,
            use_bf16=False,
        )

    @pytest.fixture(scope="class")
    def trained_policy(self, policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> Policy:
        """Train policy once and reuse across all tests."""
        trainer.fit(policy, datamodule=datamodule)
        return policy
