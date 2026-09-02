# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: Pi05 + LoRA/DoRA forward-pass correctness.

These construct a real gemma_2b-sized Pi05Model and run actual forward/backward
passes, so they are marked ``@pytest.mark.slow`` and live here rather than in
``tests/unit/policies/test_pi05.py``. They cannot use a shrunk-VLM stand-in the
way ``tests/unit/policies/test_pi05.py::TestPi05LoRAIntegration`` (construction-only
tests) does, because ``PaliGemmaWithExpertModel.__init__`` (in
``physicalai.policies.pi05.model``) hardcodes
``vlm_config_hf.vision_config.projection_dim = 2048`` regardless of
``paligemma_variant``, so the VLM's text ``hidden_size`` must stay at 2048 (the
gemma_2b width) for the vision-to-text projection to line up. That forces a
~2GB vocab embedding table (``vocab_size=257152 x hidden_size=2048``), which is
too heavy to run on every PR in ``tests/unit``.

Tests for the shared, policy-agnostic LoRA/DoRA helpers themselves live in
``tests/unit/policies/test_peft.py``. Construction-only Pi05+LoRA tests (fast,
no forward pass, tiny stand-in backbone) live in
``tests/unit/policies/test_pi05.py::TestPi05LoRAIntegration``.

Run explicitly with::

    pytest -m slow tests/integration/test_pi05_lora_forward.py
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import torch
from physicalai.data import Observation
from physicalai.policies.mixins.peft import is_lora_injected, merge_lora_
from physicalai.policies.pi05 import Pi05


class TestPi05LoRAForward:
    """Forward-pass Pi05 + LoRA/DoRA integration tests."""

    @staticmethod
    def _stats() -> dict:
        return {
            "observation.state": {
                "name": "observation.state",
                "shape": (8,),
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
                "q01": [-1.0] * 8,
                "q99": [1.0] * 8,
            },
            "action": {
                "name": "action",
                "shape": (7,),
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "q01": [-1.0] * 7,
                "q99": [1.0] * 7,
            },
        }

    @pytest.mark.slow
    def test_merge_dora_before_export_preserves_predictions(self) -> None:
        """Test that merging DoRA adapters (like LoRA) preserves model predictions."""
        policy = Pi05(
            dataset_stats=self._stats(),
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype="float32",
            lora_enabled=True,
            lora_rank=4,
            lora_alpha=8,
            lora_use_dora=True,
            n_action_steps=5,
            chunk_size=5,
            gradient_checkpointing=False,
            use_random_input_noise=False,
        )
        policy.eval()

        obs = Observation(
            state=torch.randn(2, 8),
            images={"0": torch.rand(2, 3, 224, 224)},
            task=["do a thing", "do another thing"],
        )
        with torch.no_grad():
            action_before = policy(obs)

        original_model = policy.model
        merged_model = copy.deepcopy(original_model)
        merge_lora_(merged_model)
        assert not is_lora_injected(merged_model)

        policy.model = merged_model
        with torch.no_grad():
            action_after = policy(obs)
        policy.model = original_model

        torch.testing.assert_close(action_before, action_after, atol=1e-3, rtol=1e-3)
        assert is_lora_injected(policy.model)

    @pytest.mark.slow
    def test_forward_backward_with_lora(self) -> None:
        """Test a full forward+backward pass with LoRA injected produces gradients.

        Uses gemma_2b for the VLM backbone because the vision-to-text projection
        dimension (2048) is only compatible with the gemma_2b text config; the
        gemma_300m variant is reserved for construction-only tests (see
        ``tests/unit/policies/test_pi05.py::TestPi05LoRAIntegration``).
        """
        policy = Pi05(
            dataset_stats=self._stats(),
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype="bfloat16",
            lora_enabled=True,
            lora_rank=4,
            lora_alpha=8,
            n_action_steps=5,
            chunk_size=5,
            gradient_checkpointing=False,
        )
        obs = Observation(
            state=torch.randn(2, 8),
            images={"0": torch.rand(2, 3, 224, 224)},
            task=["do a thing", "do another thing"],
            action=torch.randn(2, 5, 7),
        )
        loss, loss_dict = policy(obs)
        assert torch.isfinite(loss)
        loss.backward()

        lora_params = [(n, p) for n, p in policy.named_parameters() if "lora_" in n]
        assert len(lora_params) > 0
        assert all(p.grad is not None for _, p in lora_params), "All LoRA params should receive gradients"

    @pytest.mark.slow
    def test_merge_before_export_preserves_predictions(self) -> None:
        """Test that Pi05.export's merge-before-export leaves self.model untouched.

        and produces predictions matching the pre-merge model on a disposable copy.
        """
        policy = Pi05(
            dataset_stats=self._stats(),
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype="float32",
            lora_enabled=True,
            lora_rank=4,
            lora_alpha=8,
            n_action_steps=5,
            chunk_size=5,
            gradient_checkpointing=False,
            use_random_input_noise=False,
        )
        policy.eval()

        obs = Observation(
            state=torch.randn(2, 8),
            images={"0": torch.rand(2, 3, 224, 224)},
            task=["do a thing", "do another thing"],
        )
        with torch.no_grad():
            action_before = policy(obs)

        original_model = policy.model
        merged_model = copy.deepcopy(original_model)
        merge_lora_(merged_model)
        assert not is_lora_injected(merged_model)

        policy.model = merged_model
        with torch.no_grad():
            action_after = policy(obs)
        policy.model = original_model

        torch.testing.assert_close(action_before, action_after, atol=1e-3, rtol=1e-3)
        # The live training model must be untouched (still has LoRA injected).
        assert is_lora_injected(policy.model)

    @pytest.mark.slow
    def test_checkpoint_roundtrip_preserves_lora_weights(self) -> None:
        """Test LoRA adapter weights survive a Lightning checkpoint save/load cycle."""
        policy = Pi05(
            dataset_stats=self._stats(),
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype="float32",
            lora_enabled=True,
            lora_rank=4,
            lora_alpha=8,
            n_action_steps=5,
            chunk_size=5,
            gradient_checkpointing=False,
        )

        # Perturb one LoRA param so a stale/zero-init restore would be detectable.
        with torch.no_grad():
            for _, p in policy.named_parameters():
                if p.requires_grad:
                    p.add_(1.0)
                    break

        checkpoint = {
            "state_dict": policy.state_dict(),
            "hyper_parameters": dict(policy.hparams),
            "pytorch-lightning_version": "2.0.0",
            "epoch": 0,
            "global_step": 0,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "pi05_lora.ckpt"
            # Test-local Lightning checkpoint written and read back within the same tmpdir;
            # not untrusted input. safetensors is not an option since load_from_checkpoint
            # requires a pickle checkpoint.
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            torch.save(checkpoint, ckpt_path)
            restored = Pi05.load_from_checkpoint(str(ckpt_path))

        assert restored.config.use_lora
        assert is_lora_injected(restored.model)

        orig_sd = policy.state_dict()
        restored_sd = restored.state_dict()
        assert set(orig_sd.keys()) == set(restored_sd.keys())
        for key, value in orig_sd.items():
            torch.testing.assert_close(value.float(), restored_sd[key].float(), atol=1e-5, rtol=1e-5)
