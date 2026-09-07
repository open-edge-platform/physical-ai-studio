# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared, policy-agnostic PEFT (LoRA/DoRA) helpers.

Uses small synthetic nn.Modules instead of a full policy model to keep these tests fast
and independent of any specific policy (Pi0, Pi05, ACT, ...). Full-model integration for
a given policy is covered by that policy's own test module (e.g.
``tests/unit/policies/test_pi05.py::TestPi05LoRAIntegration`` and
``tests/unit/policies/test_pi0.py::TestPi0LoRAIntegration``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from physicalai.policies.mixins.peft import (
    PeftConfigMixin,
    build_lora_config,
    inject_lora,
    is_lora_injected,
    log_trainable_parameters,
    merge_lora_,
)

if TYPE_CHECKING:
    from physicalai.policies.mixins.peft import PeftPolicyMixin


def _make_toy_module() -> torch.nn.Module:
    class _GemmaExpertStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 16, bias=False)
            self.v_proj = torch.nn.Linear(16, 16, bias=False)
            self.k_proj = torch.nn.Linear(16, 16, bias=False)

    class _ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gemma_expert = torch.nn.Module()
            self.gemma_expert.layer = _GemmaExpertStub()
            self.action_in_proj = torch.nn.Linear(8, 16)
            self.action_out_proj = torch.nn.Linear(16, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.action_in_proj(x)
            h = self.gemma_expert.layer.q_proj(h)
            return self.action_out_proj(h)

    return _ToyModel()


class TestPeftConfigMixin:
    """Tests for the shared PeftConfigMixin dataclass fields/validation."""

    def test_lora_disabled_by_default(self) -> None:
        """Test LoRA is disabled by default."""
        config = PeftConfigMixin()
        assert config.lora_enabled is False
        assert config.use_lora is False
        assert config.lora_rank == 32

    def test_use_lora_true_when_enabled(self) -> None:
        """Test use_lora mirrors lora_enabled."""
        config = PeftConfigMixin(lora_enabled=True)
        assert config.use_lora is True

    def test_effective_lora_alpha_resolves_to_rank(self) -> None:
        """Test effective_lora_alpha defaults to lora_rank (scaling=1.0) when alpha is None."""
        config = PeftConfigMixin(lora_enabled=True, lora_rank=64)
        assert config.lora_alpha is None
        assert config.effective_lora_alpha == 64

    def test_effective_lora_alpha_respects_explicit_value(self) -> None:
        """Test effective_lora_alpha uses the explicit lora_alpha when set."""
        config = PeftConfigMixin(lora_enabled=True, lora_rank=64, lora_alpha=128)
        assert config.effective_lora_alpha == 128

    def test_lora_rank_negative_rejected(self) -> None:
        """Test negative lora_rank is rejected."""
        with pytest.raises(ValueError, match="lora_rank"):
            PeftConfigMixin(lora_rank=-1)

    def test_lora_enabled_requires_positive_rank(self) -> None:
        """Test lora_enabled=True with lora_rank=0 is rejected."""
        with pytest.raises(ValueError, match="lora_rank"):
            PeftConfigMixin(lora_enabled=True, lora_rank=0)

    def test_lora_alpha_must_be_positive_when_enabled(self) -> None:
        """Test lora_alpha must be > 0 when LoRA is enabled."""
        with pytest.raises(ValueError, match="lora_alpha"):
            PeftConfigMixin(lora_enabled=True, lora_rank=8, lora_alpha=0)

    def test_lora_dropout_out_of_range_rejected(self) -> None:
        """Test lora_dropout must be in [0, 1)."""
        with pytest.raises(ValueError, match="lora_dropout"):
            PeftConfigMixin(lora_dropout=1.0)
        with pytest.raises(ValueError, match="lora_dropout"):
            PeftConfigMixin(lora_dropout=-0.1)

    def test_lora_adapter_dtype_validation(self) -> None:
        """Test lora_adapter_dtype must be 'float32' or 'auto'."""
        with pytest.raises(ValueError, match="Invalid lora_adapter_dtype"):
            PeftConfigMixin(lora_adapter_dtype="bfloat16")  # type: ignore[arg-type]

    def test_lora_target_modules_custom(self) -> None:
        """Test custom lora_target_modules is stored as-is."""
        config = PeftConfigMixin(lora_enabled=True, lora_target_modules=("q_proj", "v_proj"))
        assert config.lora_target_modules == ("q_proj", "v_proj")

    def test_lora_use_dora_default_false(self) -> None:
        """Test lora_use_dora defaults to False."""
        config = PeftConfigMixin(lora_enabled=True)
        assert config.lora_use_dora is False

    def test_lora_use_dora_enabled(self) -> None:
        """Test lora_use_dora can be enabled."""
        config = PeftConfigMixin(lora_enabled=True, lora_use_dora=True)
        assert config.lora_use_dora is True


class TestPeftHelpers:
    """Tests for the shared LoRA helpers in physicalai.policies.mixins.peft.functions."""

    def test_build_lora_config_basic_fields(self) -> None:
        """Test build_lora_config returns a usable LoraConfig object."""
        config = build_lora_config(rank=4, alpha=8, dropout=0.1, target_modules=["q_proj"])
        assert config.r == 4
        assert config.lora_alpha == 8

    def test_build_lora_config_use_dora_defaults_false(self) -> None:
        """Test build_lora_config defaults use_dora to False."""
        config = build_lora_config(rank=4, alpha=8, dropout=0.1, target_modules=["q_proj"])
        assert config.use_dora is False

    def test_build_lora_config_use_dora_enabled(self) -> None:
        """Test build_lora_config forwards use_dora=True to LoraConfig."""
        config = build_lora_config(rank=4, alpha=8, dropout=0.1, target_modules=["q_proj"], use_dora=True)
        assert config.use_dora is True

    def test_inject_lora_freezes_base_and_adds_adapters(self) -> None:
        """Test inject_lora freezes base params and adds trainable adapter params."""
        model = _make_toy_module()
        lora_config = build_lora_config(
            rank=4,
            alpha=8,
            dropout=0.0,
            target_modules=r"(gemma_expert\..*\.(q|v)_proj|(action_in_proj|action_out_proj))",
        )
        inject_lora(model, lora_config)

        assert is_lora_injected(model)

        base_params = [(n, p) for n, p in model.named_parameters() if "lora_" not in n]
        lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]

        assert len(lora_params) > 0
        assert all(not p.requires_grad for _, p in base_params)
        assert all(p.requires_grad for _, p in lora_params)

        # k_proj was not targeted, so it should remain untouched (not a PEFT tuner layer)
        assert not is_lora_injected(model.gemma_expert.layer.k_proj)
        # q_proj was targeted, so it should now be a PEFT tuner layer
        assert is_lora_injected(model.gemma_expert.layer.q_proj)

    def test_inject_lora_adapter_dtype_override(self) -> None:
        """Test inject_lora casts adapter params to the requested dtype."""
        model = _make_toy_module().to(dtype=torch.bfloat16)
        lora_config = build_lora_config(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj"])
        inject_lora(model, lora_config, adapter_dtype=torch.float32)

        lora_dtypes = {p.dtype for n, p in model.named_parameters() if "lora_" in n}
        assert lora_dtypes == {torch.float32}
        # Base (non-adapter) params keep their original bf16 dtype.
        base_dtypes = {p.dtype for n, p in model.named_parameters() if "lora_" not in n}
        assert base_dtypes == {torch.bfloat16}

    def test_inject_lora_adapter_dtype_none_inherits_base_dtype(self) -> None:
        """Test inject_lora with adapter_dtype=None lets adapters inherit the base dtype."""
        model = _make_toy_module().to(dtype=torch.bfloat16)
        lora_config = build_lora_config(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj"])
        inject_lora(model, lora_config, adapter_dtype=None)

        lora_dtypes = {p.dtype for n, p in model.named_parameters() if "lora_" in n}
        assert torch.bfloat16 in lora_dtypes

    def test_inject_lora_raises_if_no_match(self) -> None:
        """Test inject_lora raises RuntimeError if target_modules matches nothing."""
        model = _make_toy_module()
        lora_config = build_lora_config(rank=4, alpha=8, dropout=0.0, target_modules=["nonexistent_proj"])
        with pytest.raises(RuntimeError, match="zero target modules"):
            inject_lora(model, lora_config)

    def test_is_lora_injected_false_before_injection(self) -> None:
        """Test is_lora_injected returns False for a plain model."""
        model = _make_toy_module()
        assert not is_lora_injected(model)

    def test_log_trainable_parameters_does_not_raise(self) -> None:
        """Test log_trainable_parameters runs without error on both plain and LoRA models."""
        model = _make_toy_module()
        log_trainable_parameters(model)  # plain model, all params trainable

        lora_config = build_lora_config(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj"])
        inject_lora(model, lora_config)
        log_trainable_parameters(model)

    def test_merge_lora_removes_tuner_layers(self) -> None:
        """Test merge_lora_ replaces tuner layers with plain base layers."""
        model = _make_toy_module()
        lora_config = build_lora_config(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj"])
        inject_lora(model, lora_config)
        assert is_lora_injected(model)

        merge_lora_(model)

        assert not is_lora_injected(model)
        assert not is_lora_injected(model.gemma_expert.layer.q_proj)

    def test_merge_lora_preserves_forward_output(self) -> None:
        """Test that merging LoRA adapters does not change the forward output."""
        torch.manual_seed(0)
        model = _make_toy_module()
        lora_config = build_lora_config(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"])
        inject_lora(model, lora_config)
        model.eval()

        x = torch.randn(3, 8)
        with torch.no_grad():
            out_before = model(x)

        merge_lora_(model)
        with torch.no_grad():
            out_after = model(x)

        torch.testing.assert_close(out_before, out_after, atol=1e-5, rtol=1e-5)

    def test_dora_injection_freezes_base_and_adds_adapters(self) -> None:
        """Test inject_lora with use_dora=True adds a magnitude vector alongside A/B."""
        model = _make_toy_module()
        lora_config = build_lora_config(
            rank=4,
            alpha=8,
            dropout=0.0,
            target_modules=r"(gemma_expert\..*\.(q|v)_proj|(action_in_proj|action_out_proj))",
            use_dora=True,
        )
        inject_lora(model, lora_config)

        assert is_lora_injected(model)
        param_names = {n for n, _ in model.named_parameters()}
        assert any("lora_magnitude_vector" in n for n in param_names), (
            "DoRA should add a lora_magnitude_vector parameter"
        )

        base_params = [(n, p) for n, p in model.named_parameters() if "lora_" not in n]
        lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
        assert all(not p.requires_grad for _, p in base_params)
        assert all(p.requires_grad for _, p in lora_params)

    def test_merge_dora_preserves_forward_output(self) -> None:
        """Test that merging DoRA adapters does not change the forward output."""
        torch.manual_seed(0)
        model = _make_toy_module()
        lora_config = build_lora_config(
            rank=4,
            alpha=8,
            dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            use_dora=True,
        )
        inject_lora(model, lora_config)
        model.eval()

        x = torch.randn(3, 8)
        with torch.no_grad():
            out_before = model(x)

        merge_lora_(model)

        assert not is_lora_injected(model)
        with torch.no_grad():
            out_after = model(x)

        torch.testing.assert_close(out_before, out_after, atol=1e-4, rtol=1e-4)


class _ToyPeftModel(torch.nn.Module):
    """Toy model implementing the PeftModelMixin contract for host tests."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.v_proj = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_proj(self.q_proj(x))

    @classmethod
    def get_default_peft_targets(cls) -> str:
        del cls
        return r"(q|v)_proj"


class _ExportRecorder:
    """Records the model instance handed to export() by a host policy."""

    def __init__(self) -> None:
        self.exported_model: torch.nn.Module | None = None


def _make_host_policy_class(recorder: _ExportRecorder) -> type:
    from physicalai.policies.mixins.peft import PeftPolicyMixin

    class _ExportableStub:
        """Stand-in for ExportablePolicyMixin: terminal export() reading self.model."""

        def export(self, output_path: object, backend_arg: object, input_sample: object = None, **kw: object) -> None:
            recorder.exported_model = self.model  # type: ignore[attr-defined]
            del output_path, backend_arg, input_sample, kw

    class _HostPolicy(PeftPolicyMixin, _ExportableStub):
        def __init__(self, *, use_lora: bool) -> None:
            self.config = type("Cfg", (), {"use_lora": use_lora})()
            self.model: torch.nn.Module | None = _ToyPeftModel()

    return _HostPolicy


class TestPeftPolicyMixinExport:
    """Tests for PeftPolicyMixin.export()'s cooperative super() hoist."""

    def test_export_without_lora_passes_through_unmodified_model(self) -> None:
        """Test export() delegates directly to super().export() when LoRA is not enabled."""
        recorder = _ExportRecorder()
        host_cls = _make_host_policy_class(recorder)
        host = host_cls(use_lora=False)
        original_model = host.model

        host.export("out", "torch")

        assert recorder.exported_model is original_model
        assert host.model is original_model

    def test_export_with_injected_lora_swaps_in_merged_model_and_restores(self) -> None:
        """Test export() swaps self.model to a merged copy for the call, then restores it."""
        from physicalai.policies.mixins.peft import build_lora_config, inject_lora

        recorder = _ExportRecorder()
        host_cls = _make_host_policy_class(recorder)
        host = host_cls(use_lora=True)
        original_model = host.model

        lora_config = build_lora_config(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"])
        inject_lora(host.model, lora_config)

        host.export("out", "torch")

        assert recorder.exported_model is not None
        assert recorder.exported_model is not original_model, "export() should use a disposable merged copy"
        assert not is_lora_injected(recorder.exported_model)
        assert host.model is original_model, "the live training model must be restored after export()"

    def test_export_with_lora_enabled_but_not_injected_passes_through(self) -> None:
        """Test export() falls back to the live model if LoRA is enabled but never injected."""
        recorder = _ExportRecorder()
        host_cls = _make_host_policy_class(recorder)
        host = host_cls(use_lora=True)
        original_model = host.model

        host.export("out", "torch")

        assert recorder.exported_model is original_model
        assert host.model is original_model


class TestPeftPolicyMixinOnFitStart:
    """Tests for PeftPolicyMixin.on_fit_start()'s missing-injection guard."""

    def _make_host(self, *, use_lora: bool, inject: bool) -> PeftPolicyMixin:
        from physicalai.policies.mixins.peft import PeftPolicyMixin, build_lora_config, inject_lora

        class _Host(PeftPolicyMixin):
            def __init__(self) -> None:
                self.config = type("Cfg", (), {"use_lora": use_lora})()
                self.model: torch.nn.Module | None = _ToyPeftModel()

        host = _Host()
        if inject:
            lora_config = build_lora_config(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"])
            inject_lora(host.model, lora_config)
        return host

    def test_raises_when_lora_enabled_but_not_injected(self) -> None:
        """Test on_fit_start raises if use_lora is True but no adapters were injected."""
        host = self._make_host(use_lora=True, inject=False)
        with pytest.raises(RuntimeError, match="_inject_lora"):
            host.on_fit_start()

    def test_passes_when_lora_enabled_and_injected(self) -> None:
        """Test on_fit_start is a no-op if adapters were correctly injected."""
        host = self._make_host(use_lora=True, inject=True)
        host.on_fit_start()  # should not raise

    def test_passes_when_lora_disabled(self) -> None:
        """Test on_fit_start is a no-op if LoRA is disabled entirely."""
        host = self._make_host(use_lora=False, inject=False)
        host.on_fit_start()  # should not raise

