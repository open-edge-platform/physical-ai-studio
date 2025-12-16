# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for first-party policies.

Tests validate the complete pipeline:
1. Train a policy
2. Validate/test the trained policy
3. Export to multiple backends (TestE2E only)
4. Load exported model for inference (TestE2E only)
5. Verify numerical consistency (TestE2E only)
"""

from pathlib import Path

import pytest
import torch

from getiaction.data import LeRobotDataModule
from getiaction.inference import InferenceModel
from getiaction.policies import get_policy
from getiaction.policies.base.policy import Policy
from getiaction.train import Trainer

# Export backend constants
EXPORT_BACKENDS = ["openvino", "onnx", "torch", "torch_export_ir"]

# Policy names for parametrization
FIRST_PARTY_POLICIES = ["pi0"]
FIRST_PARTY_POLICIES_WITH_EXPORT = ["act"]


@pytest.fixture(scope="class")
def trainer() -> Trainer:
    """Create trainer with fast development configuration."""
    return Trainer(
        fast_dev_run=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )


class CoreE2ETests:
    """Base class with core E2E tests (train/validate/test)."""

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


class ExportE2ETests:
    """Base class with export E2E tests (export/inference/consistency)."""

    @pytest.mark.parametrize("backend", EXPORT_BACKENDS)
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

    @pytest.mark.parametrize("backend", EXPORT_BACKENDS)
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

        from getiaction.data.lerobot import FormatConverter

        batch_observation = FormatConverter.to_observation(sample_batch)
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
        policy_name = trained_policy.__class__.__name__.lower()
        export_dir = tmp_path / f"{policy_name}_{backend}"

        from getiaction.data.lerobot import FormatConverter

        sample_batch = next(iter(datamodule.train_dataloader()))
        batch_observation = FormatConverter.to_observation(sample_batch)
        single_observation = batch_observation[0:1].to("cpu")

        # Get training output
        torch.manual_seed(42)
        trained_policy.eval()
        trained_policy.reset()
        with torch.no_grad():
            train_output = trained_policy.select_action(single_observation)

        # Export and load model
        trained_policy.export(export_dir, backend)
        inference_model = InferenceModel.load(export_dir)

        # Run inference
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        inference_model.reset()
        inference_output = inference_model.select_action(single_observation.to_numpy())

        # Compare outputs
        train_action: torch.Tensor = train_output[0].cpu()
        if len(train_action.shape) > 1:
            train_action = train_action[0]

        if not isinstance(inference_output, torch.Tensor):
            inference_output = torch.from_numpy(inference_output)

        inference_output_cpu: torch.Tensor = inference_output.cpu().squeeze(0)
        if len(inference_output_cpu.shape) > 1:
            inference_output_cpu = inference_output_cpu[0]

        torch.testing.assert_close(inference_output_cpu, train_action, rtol=0.2, atol=0.2)


@pytest.mark.parametrize("policy_name", FIRST_PARTY_POLICIES, indirect=True)
class TestE2ECore(CoreE2ETests):
    """E2E core tests for policies without export support (Pi0, etc.)."""

    @pytest.fixture(scope="class")
    def datamodule(self) -> LeRobotDataModule:
        """Create datamodule with image observations for Pi0."""
        return LeRobotDataModule(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            train_batch_size=2,
            episodes=list(range(2)),
        )


@pytest.mark.parametrize("policy_name", FIRST_PARTY_POLICIES_WITH_EXPORT, indirect=True)
class TestE2E(CoreE2ETests, ExportE2ETests):
    """E2E tests for policies with export support (ACT, etc.)."""

    @pytest.fixture(scope="class")
    def datamodule(self) -> LeRobotDataModule:
        """Create datamodule for first-party policies."""
        return LeRobotDataModule(
            repo_id="lerobot/pusht",
            train_batch_size=8,
            episodes=list(range(10)),
        )
