# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: file-ignore[assert,private-member-access,magic-value-comparison]

"""End-to-end integration tests for Cosmos3 policy.

Validates the complete training, inference, and rollout integration:
1. Training a policy end-to-end with Trainer(fast_dev_run=1) and LeRobotDataModule.
2. Action chunk prediction and queuing via select_action and predict_action_chunk.
3. Episode reset behavior clearing action queues and conditioning state.
4. Embodiment action space handling (PushT 2D identity and DROID 8D joint_pos with gripper flip).
5. Safe checkpoint and adapter saving/loading exclusively with .safetensors format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import load_file

from physicalai.data import LeRobotDataModule
from physicalai.data.observation import IMAGES, STATE
from physicalai.policies import Cosmos3
from physicalai.policies.cosmos3.surgery import HEAD_KEYS
from physicalai.train import Trainer
from tests.unit.policies.test_cosmos3 import _create_mock_cosmos3_pipeline

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
class TestCosmos3E2E:
    """Integration test suite for Cosmos 3 policy training and execution."""

    @pytest.fixture
    @staticmethod
    def pusht_datamodule() -> LeRobotDataModule:
        """Create a lightweight single-episode Push-T datamodule.

        Returns:
            The configured LeRobotDataModule.
        """
        return LeRobotDataModule(
            repo_id="lerobot/pusht",
            train_batch_size=1,
            episodes=[0],
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

    @pytest.fixture
    @staticmethod
    def trainer() -> Trainer:
        """Create a single-step CPU development trainer.

        Returns:
            The configured fast_dev_run Trainer.
        """
        return Trainer(
            fast_dev_run=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )

    @staticmethod
    def test_cosmos3_fit_pusht_e2e(trainer: Trainer, pusht_datamodule: LeRobotDataModule) -> None:
        """Test training Cosmos3 policy for one step with real LeRobotDataModule.

        Args:
            trainer: Trainer fixture.
            pusht_datamodule: Datamodule fixture.
        """
        pipe = _create_mock_cosmos3_pipeline()
        pipe._prepare_action_video_conditioning.return_value = (
            torch.zeros(1, 3, 5, 224, 224),
            torch.tensor([1, 3, 224, 224]),
            224,
            224,
        )
        pipe._remove_action_video_padding_from_latent.return_value = torch.zeros(1, 16, 2, 14, 14)
        pipe._encode_video.return_value = torch.zeros(1, 16, 2, 14, 14)

        policy = Cosmos3(embodiment="pusht", chunk_size=4, n_action_steps=4, pipeline=pipe)
        assert policy.model is not None
        policy.model._get_or_build_pack = MagicMock(return_value={})

        with patch("physicalai.policies.cosmos3.model.flow_matching_step") as mock_step:
            mock_step.return_value = (torch.tensor(1.0, requires_grad=True), torch.tensor(0.5), torch.tensor(0.05))
            trainer.fit(policy, datamodule=pusht_datamodule)

        assert trainer.state.finished

    @staticmethod
    def test_cosmos3_predict_action_chunk_e2e(pusht_datamodule: LeRobotDataModule) -> None:
        """Test action chunk prediction from real datamodule observation batch.

        Args:
            pusht_datamodule: Datamodule fixture.
        """
        pipe = _create_mock_cosmos3_pipeline()
        # Mock pipeline output: 5 tokens (1 state + 4 actions), action dim 64
        pipe.return_value = MagicMock(action=[torch.full((5, 64), 0.5)])
        policy = Cosmos3(embodiment="pusht", chunk_size=4, n_action_steps=4, pipeline=pipe)

        pusht_datamodule.setup("fit")
        batch = next(iter(pusht_datamodule.train_dataloader()))

        chunk = policy.predict_action_chunk(batch)
        assert chunk.shape == (1, 4, 2)
        assert isinstance(chunk, torch.Tensor)

    @staticmethod
    def test_cosmos3_select_action_chunking_and_reset_e2e(pusht_datamodule: LeRobotDataModule) -> None:
        """Test action queue chunking and episode reset behavior.

        Args:
            pusht_datamodule: Datamodule fixture.
        """
        pipe = _create_mock_cosmos3_pipeline()
        pipe.return_value = MagicMock(action=[torch.full((5, 64), 0.5)])
        policy = Cosmos3(embodiment="pusht", chunk_size=4, n_action_steps=4, pipeline=pipe)

        pusht_datamodule.setup("fit")
        batch = next(iter(pusht_datamodule.train_dataloader()))

        # Step 1: predicts chunk of 4, queues actions, returns first (B=1, D=2)
        action1 = policy.select_action(batch)
        assert action1.shape == (1, 2)
        assert len(policy._action_queue) == 3

        # Step 2: returns next action from queue without invoking pipeline again
        action2 = policy.select_action(batch)
        assert action2.shape == (1, 2)
        assert len(policy._action_queue) == 2
        torch.testing.assert_close(action1, action2)

        # Episode boundary: reset clears queue and conditioning state
        pipe.current_state = torch.tensor([1.0, 2.0])
        policy.reset()
        assert len(policy._action_queue) == 0
        assert pipe.current_state is None

    @staticmethod
    def test_cosmos3_droid_joint_pos_e2e() -> None:
        """Test DROID embodiment with 8D joint_pos action space and gripper inversion."""
        pipe = _create_mock_cosmos3_pipeline()
        # Model returns 9 tokens (1 prepended state token + 8 chunk actions) of width 64
        pipe.return_value = MagicMock(action=[torch.zeros(9, 64)])
        policy = Cosmos3(embodiment="droid_lerobot", chunk_size=8, n_action_steps=8, pipeline=pipe)

        # Synthetic DROID observation: 7 arm joints + 1 gripper = 8D
        batch = {
            IMAGES: torch.zeros(1, 3, 224, 224),
            STATE: torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]]),
        }
        chunk = policy.predict_action_chunk(batch)
        assert chunk.shape == (1, 8, 8)

        # Gripper was flipped when conditioned: 1.0 -> 0.0 in model space
        assert pipe.current_state is not None
        torch.testing.assert_close(pipe.current_state[-1], torch.tensor(0.0))

    @staticmethod
    def test_cosmos3_safetensors_checkpoint_lifecycle_e2e(tmp_path: Path) -> None:
        """Test saving adapter/head to .safetensors and restoring onto a fresh policy.

        Args:
            tmp_path: Pytest temporary path fixture.
        """
        pipe = _create_mock_cosmos3_pipeline()
        policy = Cosmos3(embodiment="pusht", chunk_size=4, pipeline=pipe)
        assert policy.model is not None

        # Configure normalization affine and mock head
        policy.model.norm_offset = torch.tensor([0.2, -0.4])
        policy.model.norm_scale = torch.tensor([1.2, 2.4])
        setattr(policy.model.transformer, HEAD_KEYS[0], torch.nn.Linear(2, 2))

        out_dir = tmp_path / "checkpoint_adapter"
        policy.save_pretrained_adapter(out_dir)

        # Verify files are saved as .safetensors and .json metadata
        assert (out_dir / "pusht_head.safetensors").is_file()
        assert (out_dir / "pusht_head.json").is_file()

        head_tensors = load_file(str(out_dir / "pusht_head.safetensors"))
        torch.testing.assert_close(head_tensors["norm_offset"], torch.tensor([0.2, -0.4]))

        # Restore onto a new policy instance
        new_pipe = _create_mock_cosmos3_pipeline()
        new_policy = Cosmos3(embodiment="pusht", chunk_size=4, pipeline=new_pipe)
        new_policy.load_pretrained_adapter(out_dir)

        assert new_policy.model is not None
        torch.testing.assert_close(new_policy.model.norm_offset, torch.tensor([0.2, -0.4]))
        torch.testing.assert_close(new_policy.model.norm_scale, torch.tensor([1.2, 2.4]))
