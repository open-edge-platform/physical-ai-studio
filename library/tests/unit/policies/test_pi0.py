# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Pi0 policy.

Fast, self-contained tests with no external dependencies (no HuggingFace model downloads).
"""

from __future__ import annotations

import pytest
import torch
from physicalai.config import Config
from physicalai.policies.pi0 import Pi0, Pi05, Pi0Config


class TestPi0Config:
    """Tests for Pi0Config dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Pi0Config()
        assert config.paligemma_variant == "gemma_2b"
        assert config.action_expert_variant == "gemma_300m"
        assert config.variant == "pi0"
        assert config.dtype == "bfloat16"
        assert config.tune_action_expert is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = Pi0Config(
            variant="pi05",
            chunk_size=100,
            n_action_steps=50,
            learning_rate=1e-4,
            tune_vision_encoder=True,
            paligemma_variant="gemma_2b",
        )
        assert config.variant == "pi05"
        assert config.chunk_size == 100
        assert config.n_action_steps == 50
        assert config.learning_rate == 1e-4
        assert config.tune_vision_encoder is True
        assert config.paligemma_variant == "gemma_2b"

    def test_training_config_values(self) -> None:
        """Test training-related configuration values."""
        config = Pi0Config()
        assert config.learning_rate == 2.5e-5
        assert config.weight_decay == 1e-10
        assert config.warmup_steps == 1000
        assert config.decay_steps == 30000
        assert config.decay_lr == 2.5e-6
        assert config.grad_clip_norm == 1.0

    def test_flow_matching_config_values(self) -> None:
        """Test flow matching configuration values."""
        config = Pi0Config()
        assert config.time_beta_alpha == 1.5
        assert config.time_beta_beta == 1.0
        assert config.time_scale == 0.999
        assert config.time_offset == 0.001
        assert config.time_min_period == 4e-3
        assert config.time_max_period == 4.0
        assert config.num_inference_steps == 10

    def test_n_action_steps_validation(self) -> None:
        """Test n_action_steps cannot exceed chunk_size."""
        with pytest.raises(ValueError, match="chunk_size"):
            Pi0Config(chunk_size=50, n_action_steps=100)

    def test_variant_validation(self) -> None:
        """Test variant must be pi0 or pi05."""
        with pytest.raises(ValueError, match="variant"):
            Pi0Config(variant="invalid")  # type: ignore[arg-type]

    def test_paligemma_variant_validation(self) -> None:
        """Test paligemma_variant must be gemma_2b."""
        with pytest.raises(ValueError, match="paligemma_variant"):
            Pi0Config(paligemma_variant="gemma_300m")
        with pytest.raises(ValueError, match="paligemma_variant"):
            Pi0Config(paligemma_variant="invalid")

    def test_action_expert_variant_validation(self) -> None:
        """Test action_expert_variant must be valid."""
        with pytest.raises(ValueError, match="action_expert_variant"):
            Pi0Config(action_expert_variant="invalid")

    def test_is_pi05_property(self) -> None:
        """Test is_pi05 property."""
        config_pi0 = Pi0Config(variant="pi0")
        config_pi05 = Pi0Config(variant="pi05")
        assert config_pi0.is_pi05 is False
        assert config_pi05.is_pi05 is True

    def test_inheritance_and_serialization(self) -> None:
        """Test config inherits from base Config and supports serialization."""
        config = Pi0Config(chunk_size=100, learning_rate=1e-4)
        assert isinstance(config, Config)

        config_dict = config.to_dict()
        assert config_dict["chunk_size"] == 100
        assert config_dict["learning_rate"] == 1e-4

        restored = Pi0Config.from_dict(config_dict)
        assert restored.chunk_size == 100
        assert restored.learning_rate == 1e-4

    def test_max_token_len_auto_computed(self) -> None:
        """Test max_token_len is auto-computed based on variant."""
        config_pi0 = Pi0Config(variant="pi0", max_token_len=None)
        config_pi05 = Pi0Config(variant="pi05", max_token_len=None)
        assert config_pi0.max_token_len == 48
        assert config_pi05.max_token_len == 200


class TestPi0LoRAConfig:
    """Tests for Pi0Config LoRA fields (via physicalai.policies.mixins.peft.PeftConfigMixin)."""

    def test_lora_disabled_by_default(self) -> None:
        """Test LoRA is disabled by default for Pi0/Pi0.5."""
        config = Pi0Config()
        assert config.lora_enabled is False
        assert config.use_lora is False
        assert config.lora_rank == 32

    def test_use_lora_true_when_enabled(self) -> None:
        """Test use_lora mirrors lora_enabled."""
        config = Pi0Config(lora_enabled=True)
        assert config.use_lora is True

    def test_effective_lora_alpha_resolves_to_rank(self) -> None:
        """Test effective_lora_alpha defaults to lora_rank (scaling=1.0) when alpha is None."""
        config = Pi0Config(lora_enabled=True, lora_rank=64)
        assert config.lora_alpha is None
        assert config.effective_lora_alpha == 64

    def test_effective_lora_alpha_respects_explicit_value(self) -> None:
        """Test effective_lora_alpha uses the explicit lora_alpha when set."""
        config = Pi0Config(lora_enabled=True, lora_rank=64, lora_alpha=128)
        assert config.effective_lora_alpha == 128

    def test_lora_rank_negative_rejected(self) -> None:
        """Test negative lora_rank is rejected."""
        with pytest.raises(ValueError, match="lora_rank"):
            Pi0Config(lora_rank=-1)

    def test_lora_enabled_requires_positive_rank(self) -> None:
        """Test lora_enabled=True with lora_rank=0 is rejected."""
        with pytest.raises(ValueError, match="lora_rank"):
            Pi0Config(lora_enabled=True, lora_rank=0)

    def test_lora_dropout_out_of_range_rejected(self) -> None:
        """Test lora_dropout must be in [0, 1)."""
        with pytest.raises(ValueError, match="lora_dropout"):
            Pi0Config(lora_dropout=1.0)

    def test_lora_target_modules_custom(self) -> None:
        """Test custom lora_target_modules is stored as-is."""
        config = Pi0Config(lora_enabled=True, lora_target_modules=("q_proj", "v_proj"))
        assert config.lora_target_modules == ("q_proj", "v_proj")

    def test_lora_use_dora_default_false(self) -> None:
        """Test lora_use_dora defaults to False."""
        config = Pi0Config(lora_enabled=True)
        assert config.lora_use_dora is False

    def test_lora_serialization_roundtrip(self) -> None:
        """Test LoRA fields survive to_dict/from_dict roundtrip."""
        config = Pi0Config(
            lora_enabled=True,
            lora_rank=32,
            lora_alpha=64,
            lora_dropout=0.1,
            lora_use_dora=True,
        )
        restored = Pi0Config.from_dict(config.to_dict())
        assert restored.lora_enabled is True
        assert restored.lora_rank == 32
        assert restored.lora_alpha == 64
        assert restored.lora_dropout == 0.1
        assert restored.lora_use_dora is True
        assert restored.use_lora is True

    def test_lora_fields_also_available_on_pi05_variant(self) -> None:
        """Test lora_* fields work identically when variant='pi05'."""
        config = Pi0Config(variant="pi05", lora_enabled=True, lora_rank=16)
        assert config.is_pi05 is True
        assert config.use_lora is True
        assert config.lora_rank == 16

    def test_lora_enabled_rejects_non_default_tune_paligemma(self) -> None:
        """Test lora_enabled=True with tune_paligemma=True (non-default) is rejected."""
        with pytest.raises(ValueError, match="tune_paligemma"):
            Pi0Config(lora_enabled=True, tune_paligemma=True)

    def test_lora_enabled_rejects_non_default_tune_action_expert(self) -> None:
        """Test lora_enabled=True with tune_action_expert=False (non-default) is rejected."""
        with pytest.raises(ValueError, match="tune_action_expert"):
            Pi0Config(lora_enabled=True, tune_action_expert=False)

    def test_lora_enabled_rejects_non_default_tune_vision_encoder(self) -> None:
        """Test lora_enabled=True with tune_vision_encoder=True (non-default) is rejected."""
        with pytest.raises(ValueError, match="tune_vision_encoder"):
            Pi0Config(lora_enabled=True, tune_vision_encoder=True)

    def test_lora_enabled_allows_default_tune_flags(self) -> None:
        """Test lora_enabled=True with all tune_* flags at their defaults is allowed."""
        config = Pi0Config(
            lora_enabled=True,
            tune_paligemma=False,
            tune_action_expert=True,
            tune_vision_encoder=False,
        )
        assert config.use_lora is True

    def test_tune_flags_allowed_when_lora_disabled(self) -> None:
        """Test non-default tune_* flags are fine when LoRA is disabled."""
        config = Pi0Config(lora_enabled=False, tune_paligemma=True, tune_action_expert=False)
        assert config.tune_paligemma is True
        assert config.tune_action_expert is False


class TestPi0Policy:
    """Tests for Pi0 Lightning policy wrapper."""

    def test_lazy_initialization(self) -> None:
        """Test lazy initialization doesn't create model."""
        policy = Pi0()
        assert policy.model is None

    def test_hyperparameters_saved(self) -> None:
        """Test hyperparameters are saved for checkpoint."""
        policy = Pi0(
            chunk_size=100,
            learning_rate=1e-4,
            tune_vision_encoder=True,
        )
        assert policy.hparams.chunk_size == 100
        assert policy.hparams.learning_rate == 1e-4
        assert policy.hparams.tune_vision_encoder is True
        assert "config" in policy.hparams
        assert policy.hparams["config"]["chunk_size"] == 100

    def test_config_attribute(self) -> None:
        """Test Pi0 policy has config attribute."""
        policy = Pi0(chunk_size=100, learning_rate=1e-4)

        assert policy.config is not None
        assert policy.config.chunk_size == 100
        assert policy.config.learning_rate == 1e-4

    def test_n_action_steps(self) -> None:
        """Test n_action_steps is correctly set."""
        policy = Pi0(n_action_steps=25, chunk_size=50)
        assert policy._n_action_steps == 25
        assert policy.config.n_action_steps == 25

    @pytest.mark.parametrize("method", ["forward", "predict_action_chunk"])
    def test_methods_raise_without_model(self, method: str) -> None:
        """Test methods raise ValueError if model not initialized."""
        from physicalai.data import Observation

        policy = Pi0()
        dummy_obs = Observation(state=torch.randn(1, 10))
        with pytest.raises(ValueError, match="not initialized"):
            getattr(policy, method)(dummy_obs)


class TestPi05Policy:
    """Tests for Pi05 (Pi0.5) policy alias."""

    def test_pi05_creates_pi05_variant(self) -> None:
        """Test Pi05 creates policy with variant='pi05'."""
        policy = Pi05()
        assert policy.config.variant == "pi05"
        assert policy.config.is_pi05 is True

    def test_pi05_inherits_from_pi0(self) -> None:
        """Test Pi05 inherits from Pi0."""
        policy = Pi05()
        assert isinstance(policy, Pi0)

    def test_pi05_with_custom_args(self) -> None:
        """Test Pi05 accepts custom arguments."""
        policy = Pi05(chunk_size=100, learning_rate=1e-4)
        assert policy.config.variant == "pi05"
        assert policy.config.chunk_size == 100
        assert policy.config.learning_rate == 1e-4


class TestPi0Preprocessor:
    """Tests for Pi0 preprocessor functions."""

    def test_make_pi0_preprocessors(self) -> None:
        """Test make_pi0_preprocessors returns callables."""
        from physicalai.policies.pi0.preprocessor import make_pi0_preprocessors

        preprocessor, postprocessor = make_pi0_preprocessors(
            max_state_dim=32,
            max_action_dim=32,
            chunk_size=50,
            stats=None,
            image_resolution=(224, 224),
            max_token_len=48,
        )
        assert callable(preprocessor)
        assert callable(postprocessor)

    def test_preprocessor_is_nn_module(self) -> None:
        """Test that preprocessors are nn.Module instances."""
        from physicalai.policies.pi0.preprocessor import (
            Pi0Postprocessor,
            Pi0Preprocessor,
        )
        from torch import nn

        preprocessor = Pi0Preprocessor()
        postprocessor = Pi0Postprocessor(action_dim=7)

        assert isinstance(preprocessor, nn.Module)
        assert isinstance(postprocessor, nn.Module)

    def test_preprocessor_default_values(self) -> None:
        """Test preprocessor default configuration values."""
        from physicalai.policies.pi0.preprocessor import Pi0Preprocessor

        preprocessor = Pi0Preprocessor()

        assert preprocessor.max_state_dim == 32
        assert preprocessor.max_action_dim == 32
        assert preprocessor.image_resolution == (224, 224)
        assert preprocessor.max_token_len == 48

    def test_preprocessor_custom_values(self) -> None:
        """Test preprocessor with custom configuration values."""
        from physicalai.policies.pi0.preprocessor import Pi0Preprocessor

        preprocessor = Pi0Preprocessor(
            max_state_dim=64,
            max_action_dim=16,
            image_resolution=(512, 512),
            max_token_len=64,
        )

        assert preprocessor.max_state_dim == 64
        assert preprocessor.max_action_dim == 16
        assert preprocessor.image_resolution == (512, 512)
        assert preprocessor.max_token_len == 64


class TestGetPolicy:
    """Tests for get_policy with Pi0."""

    def test_get_pi0_policy(self) -> None:
        """Test creating Pi0 policy via get_policy."""
        from physicalai.policies import get_policy

        policy = get_policy("pi0", source="physicalai")
        assert policy.__class__.__name__ == "Pi0"

    def test_get_pi05_policy(self) -> None:
        """Test creating Pi05 policy via get_policy."""
        from physicalai.policies import get_policy

        policy = get_policy("pi05", source="physicalai")
        assert policy.__class__.__name__ == "Pi05"

    def test_case_insensitive(self) -> None:
        """Test policy name is case-insensitive."""
        from physicalai.policies import get_policy

        policy = get_policy("PI0")
        assert policy.__class__.__name__ == "Pi0"


class TestPi0LoRAIntegration:
    """Full Pi0/Pi0.5 + LoRA integration tests.

    These construct a real Pi0Model, but the PaliGemma backbone is replaced with a
    tiny, randomly-initialized stand-in (see ``_patch_tiny_paligemma`` below) so no
    HuggingFace download happens and peak memory stays in the low GBs. The stand-in
    preserves ``head_dim``/``num_key_value_heads``/depth compatibility with the action
    expert so the shared-KV-cache attention path in ``PaliGemmaWithExpert.forward``
    still exercises real code, just at a much smaller scale.
    """

    @pytest.fixture(autouse=True)
    def _patch_tiny_paligemma(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace the real ~3B-param PaliGemma backbone with a tiny stand-in.

        Keeps ``vocab_size`` real (tokenizer output ids must stay in range) and keeps
        ``head_dim``/``num_key_value_heads`` consistent between the VLM backbone and the
        action expert, since ``PaliGemmaWithExpert.forward`` shares KV cache state
        between them. Also replaces the tokenizer with a tiny, fully offline stand-in --
        ``Pi0Preprocessor.tokenizer`` otherwise downloads the real (gated) PaliGemma
        tokenizer from HuggingFace, which requires network access and an auth token.
        """
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            AutoTokenizer,
            GemmaConfig,
            PaliGemmaConfig,
            PaliGemmaForConditionalGeneration,
            PreTrainedTokenizerFast,
            SiglipVisionConfig,
        )

        import physicalai.policies.pi0.components.gemma as gemma_mod

        tiny = gemma_mod._GemmaConfig(  # noqa: SLF001
            vocab_size=257152,
            width=64,
            depth=2,
            mlp_dim=128,
            num_heads=1,
            num_kv_heads=1,
            head_dim=64,
        )
        monkeypatch.setitem(gemma_mod._GEMMA_CONFIGS, "gemma_300m", tiny)
        monkeypatch.setitem(gemma_mod._GEMMA_CONFIGS, "gemma_2b", tiny)

        def _fake_from_pretrained(
            cls: type,
            model_id: str,  # noqa: ARG001
            *,
            dtype: torch.dtype | None = None,
            revision: str | None = None,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> PaliGemmaForConditionalGeneration:
            vision_config = SiglipVisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=1,
                image_size=224,
                patch_size=14,
            )
            text_config = GemmaConfig(
                vocab_size=257152,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=64,
            )
            config = PaliGemmaConfig(vision_config=vision_config, text_config=text_config, projection_dim=64)
            model = cls(config)
            return model.to(dtype) if dtype is not None else model

        monkeypatch.setattr(
            PaliGemmaForConditionalGeneration,
            "from_pretrained",
            classmethod(_fake_from_pretrained),
        )

        def _fake_tokenizer_from_pretrained(*args: object, **kwargs: object) -> PreTrainedTokenizerFast:  # noqa: ARG001
            backend = Tokenizer(WordLevel(vocab={"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")

        monkeypatch.setattr(AutoTokenizer, "from_pretrained", staticmethod(_fake_tokenizer_from_pretrained))

    @staticmethod
    def _stats() -> dict:
        return {
            "observation.state": {"mean": [0.0] * 8, "std": [1.0] * 8, "shape": (8,)},
            "action": {"mean": [0.0] * 8, "std": [1.0] * 8, "shape": (8,)},
        }

    def test_lora_injection_reduces_trainable_params(self) -> None:
        """Test enabling LoRA drastically reduces the number of trainable parameters."""
        from physicalai.policies.mixins.peft import is_lora_injected

        policy = Pi0(
            action_expert_variant="gemma_300m",
            max_action_dim=8,
            max_state_dim=8,
            chunk_size=10,
            n_action_steps=10,
            lora_enabled=True,
            lora_rank=4,
            lora_alpha=8,
            dataset_stats=self._stats(),
        )
        assert policy.config.use_lora
        assert is_lora_injected(policy.model)

        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total = sum(p.numel() for p in policy.parameters())
        assert 0 < trainable < total
        assert trainable / total < 0.01, "LoRA should train well under 1% of parameters"

    def test_lora_disabled_does_not_inject(self) -> None:
        """Test that with lora_enabled=False (the default), no adapters are injected."""
        from physicalai.policies.mixins.peft import is_lora_injected

        policy = Pi0(
            action_expert_variant="gemma_300m",
            max_action_dim=8,
            max_state_dim=8,
            chunk_size=10,
            n_action_steps=10,
            dataset_stats=self._stats(),
        )
        assert policy.config.use_lora is False
        assert not is_lora_injected(policy.model)

        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total = sum(p.numel() for p in policy.parameters())
        assert trainable == total or trainable > 0  # tune_action_expert defaults to True

    def test_lora_works_on_pi05_variant(self) -> None:
        """Test LoRA injection also works via the Pi05 convenience alias."""
        from physicalai.policies.mixins.peft import is_lora_injected

        policy = Pi05(
            action_expert_variant="gemma_300m",
            max_action_dim=8,
            max_state_dim=8,
            chunk_size=10,
            n_action_steps=10,
            lora_enabled=True,
            lora_rank=4,
            lora_alpha=8,
            dataset_stats=self._stats(),
        )
        assert policy.config.variant == "pi05"
        assert is_lora_injected(policy.model)

    def test_dora_injection(self) -> None:
        """Test lora_use_dora=True injects DoRA adapters (magnitude vector) on Pi0Model."""
        from physicalai.policies.mixins.peft import is_lora_injected

        policy = Pi0(
            action_expert_variant="gemma_300m",
            max_action_dim=8,
            max_state_dim=8,
            chunk_size=10,
            n_action_steps=10,
            lora_enabled=True,
            lora_rank=4,
            lora_alpha=8,
            lora_use_dora=True,
            dataset_stats=self._stats(),
        )
        assert is_lora_injected(policy.model)
        param_names = {n for n, _ in policy.named_parameters()}
        assert any("lora_magnitude_vector" in n for n in param_names)

    def test_merge_lora_before_export_preserves_predictions(self) -> None:
        """Test that Pi0.export's merge-before-export leaves self.model untouched.

        And produces predictions matching the pre-merge model on a disposable copy.
        """
        import copy

        from physicalai.data import Observation
        from physicalai.policies.mixins.peft import is_lora_injected, merge_lora_

        policy = Pi0(
            action_expert_variant="gemma_300m",
            max_action_dim=8,
            max_state_dim=8,
            chunk_size=5,
            n_action_steps=5,
            dtype="float32",
            lora_enabled=True,
            lora_rank=4,
            lora_alpha=8,
            gradient_checkpointing=False,
            dataset_stats=self._stats(),
        )
        policy.eval()

        obs = Observation(
            state=torch.randn(2, 8),
            images={"0": torch.rand(2, 3, 224, 224)},
            task=["do a thing", "do another thing"],
        )
        torch.manual_seed(0)
        with torch.no_grad():
            action_before = policy(obs)

        original_model = policy.model
        merged_model = policy._merged_lora_model_for_export()  # noqa: SLF001
        assert merged_model is not None
        assert not is_lora_injected(merged_model)

        policy.model = merged_model
        torch.manual_seed(0)
        with torch.no_grad():
            action_after = policy(obs)
        policy.model = original_model

        torch.testing.assert_close(action_before, action_after, atol=1e-3, rtol=1e-3)
        # The live training model must be untouched (still has LoRA injected).
        assert is_lora_injected(policy.model)

    def test_checkpoint_roundtrip_preserves_lora_weights(self) -> None:
        """Test LoRA adapter weights survive a Lightning checkpoint save/load cycle."""
        import tempfile
        from pathlib import Path

        from physicalai.policies.mixins.peft import is_lora_injected

        policy = Pi0(
            action_expert_variant="gemma_300m",
            max_action_dim=8,
            max_state_dim=8,
            chunk_size=5,
            n_action_steps=5,
            dtype="float32",
            lora_enabled=True,
            lora_rank=4,
            lora_alpha=8,
            gradient_checkpointing=False,
            dataset_stats=self._stats(),
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
            ckpt_path = Path(tmp_dir) / "pi0_lora.ckpt"
            # Test-local Lightning checkpoint written and read back within the same tmpdir;
            # not untrusted input. safetensors is not an option since load_from_checkpoint
            # requires a pickle checkpoint.
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            torch.save(checkpoint, ckpt_path)
            restored = Pi0.load_from_checkpoint(str(ckpt_path))

        assert restored.config.use_lora
        assert is_lora_injected(restored.model)

        orig_sd = policy.state_dict()
        restored_sd = restored.state_dict()
        assert set(orig_sd.keys()) == set(restored_sd.keys())
        for key, value in orig_sd.items():
            torch.testing.assert_close(value.float(), restored_sd[key].float(), atol=1e-5, rtol=1e-5)
