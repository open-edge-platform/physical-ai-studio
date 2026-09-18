# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for first-party policies.

Tests validate the complete pipeline:
1. Train a policy
2. Validate/test the trained policy
3. Export to multiple backends
4. Load exported model for inference
5. Verify numerical consistency
"""

from pathlib import Path

import pytest
import torch

from physicalai.data import LeRobotDataModule
from physicalai.inference import InferenceModel
from physicalai.policies import get_policy
from physicalai.policies.base.policy import Policy
from physicalai.train import Trainer

DEPLOYMENT_EXPORT_BACKENDS = ["openvino", "onnx", "executorch"]
FIRST_PARTY_POLICIES_WITH_EXPORT = ["act", "smolvla", "pi05"]


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
        if policy_name == "pi05":
            return get_policy(
                policy_name,
                source="physicalai",
                freeze_vision_encoder=True,
                train_expert_only=True,
            )
        return get_policy(policy_name, source="physicalai")

    @pytest.fixture(scope="class")
    def trained_policy(self, policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> Policy:
        """Train policy once and reuse across all tests."""
        trainer.fit(policy, datamodule=datamodule)
        return policy

    @pytest.fixture(scope="class")
    def initialized_policy(self, policy_name: str, datamodule: LeRobotDataModule) -> Policy:
        """Create first-party policy instance for export/inference."""
        return get_policy(policy_name, source="physicalai", dataset_stats=datamodule.train_dataset.stats).eval()

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


@pytest.mark.slow
class ExportE2ETests:
    """Base class with export E2E tests (export/inference/consistency)."""

    @pytest.mark.parametrize("backend", DEPLOYMENT_EXPORT_BACKENDS)
    def test_export_to_backend(self, initialized_policy: Policy, backend: str, tmp_path: Path) -> None:
        """Test that trained policy can be exported to different backends."""
        export_dir = tmp_path / f"{initialized_policy.__class__.__name__.lower()}_{backend}"

        if backend not in initialized_policy.get_supported_export_backends():
            pytest.skip(f"{initialized_policy.__class__.__name__} does not support export to {backend}")

        initialized_policy.export(export_dir, backend)

        assert export_dir.exists()
        assert (export_dir / "manifest.json").exists()

        if backend == "openvino":
            assert any(export_dir.glob("*.xml"))
            assert any(export_dir.glob("*.bin"))
        elif backend == "onnx":
            assert any(export_dir.glob("*.onnx"))
        elif backend == "torch":
            assert any(export_dir.glob("*.pt"))
        elif backend == "executorch":
            assert any(export_dir.glob("*.pte"))

    @pytest.mark.parametrize("backend", DEPLOYMENT_EXPORT_BACKENDS)
    def test_inference_with_exported_model(
        self,
        initialized_policy: Policy,
        backend: str,
        datamodule: LeRobotDataModule,
        tmp_path: Path,
    ) -> None:
        """Test that exported model can be loaded and used for inference."""
        if backend not in initialized_policy.get_supported_export_backends():
            pytest.skip(f"{initialized_policy.__class__.__name__} does not support export to {backend}")

        export_dir = tmp_path / f"{initialized_policy.__class__.__name__.lower()}_{backend}"
        initialized_policy.export(export_dir, backend)

        inference_model = InferenceModel(export_dir)
        assert inference_model.backend == backend

        sample_batch = next(iter(datamodule.train_dataloader()))

        from physicalai.data.lerobot import FormatConverter

        batch_observation = FormatConverter.to_observation(sample_batch)
        inference_input = batch_observation[0:1].to_numpy().to_dict(flatten=False)
        inference_output = inference_model.select_action(inference_input)

        assert inference_output.shape[-1] == 2
        assert len(inference_output.shape) in {1, 2, 3}, f"Expected 1-3D tensor, got {inference_output.shape}"

    @pytest.mark.parametrize("backend", DEPLOYMENT_EXPORT_BACKENDS)
    def test_numerical_consistency_training_vs_inference(
        self,
        initialized_policy: Policy,
        backend: str,
        datamodule: LeRobotDataModule,
        tmp_path: Path,
    ) -> None:
        """Test numerical consistency between training and inference outputs."""
        policy_name = initialized_policy.__class__.__name__.lower()
        if backend not in initialized_policy.get_supported_export_backends():
            pytest.skip(f"{policy_name} does not support export to {backend}")

        export_dir = tmp_path / f"{policy_name}_{backend}"

        from physicalai.data.lerobot import FormatConverter

        sample_batch = next(iter(datamodule.train_dataloader()))
        batch_observation = FormatConverter.to_observation(sample_batch)
        single_observation = batch_observation[0:1].to("cpu")

        torch.manual_seed(42)
        initialized_policy.eval()
        with torch.no_grad():
            train_action = initialized_policy.predict_action_chunk(single_observation)
        if isinstance(train_action, tuple):
            train_action = train_action[0]
        train_action = train_action.squeeze(0)
        if len(train_action.shape) > 1:
            train_action = train_action[0]

        initialized_policy.export(export_dir, backend)
        inference_model = InferenceModel(export_dir)

        torch.manual_seed(42)
        inference_input = single_observation.to_numpy().to_dict(flatten=False)
        inference_output = inference_model.select_action(inference_input)
        inference_output = torch.as_tensor(inference_output)
        inference_output_cpu: torch.Tensor = inference_output.cpu().squeeze(0)
        if len(inference_output_cpu.shape) > 1:
            inference_output_cpu = inference_output_cpu[0]

        torch.testing.assert_close(inference_output_cpu.to(train_action.dtype), train_action, rtol=0.2, atol=0.2)


@pytest.mark.parametrize("policy_name", FIRST_PARTY_POLICIES_WITH_EXPORT, indirect=True)
class TestE2E(CoreE2ETests, ExportE2ETests):
    """E2E tests for first-party policies with export support."""

    @pytest.fixture(scope="class")
    def datamodule(self) -> LeRobotDataModule:
        """Create datamodule for first-party policies."""
        return LeRobotDataModule(
            repo_id="lerobot/pusht",
            train_batch_size=8,
            episodes=list(range(10)),
        )
