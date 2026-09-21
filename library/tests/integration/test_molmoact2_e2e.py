# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Low-memory end-to-end coverage for the first-party MolmoAct2 policy.

MolmoAct2 subclasses the shared ``CoreE2ETests`` suite so it must satisfy the
same train, validate, and test contract as every first-party policy. Its fixtures
use a tiny FP16 model because the production checkpoint is too large for the
fast integration job and would require a download for the custom tokenizer packaged with weights.
"""

from typing import TYPE_CHECKING, cast

import pytest

from physicalai.data import LeRobotDataModule
from physicalai.policies import get_physicalai_policy_class
from physicalai.policies.base.policy import Policy
from physicalai.policies.molmoact2 import MolmoAct2, MolmoAct2Config

from . import test_first_party_e2e as shared_e2e

CoreE2ETests = shared_e2e.CoreE2ETests
trainer = shared_e2e.trainer

if TYPE_CHECKING:
    from physicalai.data.dataset import Dataset


class _StubTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, prompts: list[str], **_: object) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [[3, 35] for _ in prompts],
            "attention_mask": [[1, 1] for _ in prompts],
        }


@pytest.mark.parametrize("policy_name", ["molmoact2"], indirect=True)
class TestMolmoAct2E2E(CoreE2ETests):
    """Run the shared first-party E2E contract with lightweight fixtures."""

    @pytest.fixture(scope="class")
    @staticmethod
    def datamodule() -> LeRobotDataModule:
        """Use one cached PushT episode for training and validation.

        Returns:
            The low-memory integration datamodule.
        """
        datamodule = LeRobotDataModule(
            repo_id="lerobot/pusht",
            train_batch_size=1,
            episodes=[0],
        )
        datamodule.val_eval_dataset = datamodule.train_dataset
        return datamodule

    @pytest.fixture(scope="class")
    @staticmethod
    def policy(
        policy_name: str,
        datamodule: LeRobotDataModule,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> Policy:
        """Create the registered policy with a tiny FP16 model.

        Returns:
            The initialized MolmoAct2 policy.

        Raises:
            RuntimeError: If registry resolution or eager initialization fails.
        """
        tokenizer_dir = tmp_path_factory.mktemp("molmoact2-tokenizer")
        (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        dataset = cast("Dataset", datamodule.train_dataset)
        config = MolmoAct2Config(
            input_features=list(dataset.observation_features.values()),
            output_features=list(dataset.action_features.values()),
            hidden_size=16,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=32,
            additional_vocab_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            vision_hidden_size=16,
            vision_intermediate_size=32,
            vision_num_hidden_layers=1,
            vision_num_attention_heads=1,
            vision_num_key_value_heads=1,
            vision_head_dim=16,
            image_default_input_size=(14, 14),
            image_patch_size=14,
            image_num_pos=1,
            adapter_vit_layers=(-1,),
            adapter_hidden_size=16,
            adapter_num_attention_heads=1,
            adapter_num_key_value_heads=1,
            adapter_head_dim=16,
            adapter_intermediate_size=32,
            adapter_text_hidden_size=16,
            action_expert_max_action_dim=2,
            action_expert_hidden_size=16,
            action_expert_num_layers=1,
            action_expert_num_heads=1,
            action_expert_ffn_multiple_of=8,
            action_expert_timestep_embed_dim=8,
            max_action_dim=2,
            chunk_size=2,
            n_action_steps=1,
            flow_matching_num_steps=1,
            num_flow_timesteps=1,
            tokenizer_name_or_path=str(tokenizer_dir),
            tokenizer_max_length=4,
            image_start_token_id=32,
            image_end_token_id=33,
            image_patch_id=34,
            image_placeholder_token_id=35,
            image_col_id=36,
            low_res_image_start_token_id=37,
            image_low_res_id=38,
            frame_start_token_id=39,
            frame_end_token_id=40,
            image_processor_size={"height": 14, "width": 14},
            image_processor_pooling_size=[1, 1],
            image_use_col_tokens=False,
        )
        if get_physicalai_policy_class(policy_name) is not MolmoAct2:
            msg = f"Expected the first-party registry to resolve {policy_name!r} to MolmoAct2."
            raise RuntimeError(msg)
        policy = MolmoAct2.from_config(config)
        preprocessor = policy._preprocessor  # ruff: ignore[SLF001]
        if preprocessor is None:
            msg = "Expected MolmoAct2.from_config() to initialize its preprocessor."
            raise RuntimeError(msg)
        preprocessor._tokenizers._tokenizer = _StubTokenizer()  # type: ignore[assignment]  # ruff: ignore[SLF001]
        return policy.half()
