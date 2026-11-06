# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Public package exports and the intentionally empty export capability."""

from physicalai import policies
from physicalai.export import ExportablePolicyMixin
from physicalai.policies import patch_policy
from physicalai.policies.patch_policy.config import PatchPolicyConfig
from physicalai.policies.patch_policy.export import PatchPolicyExportMixin
from physicalai.policies.patch_policy.model import PatchPolicyModel
from physicalai.policies.patch_policy.policy import PatchPolicy


def test_export_mixin_declares_no_supported_backends() -> None:
    assert issubclass(PatchPolicyExportMixin, ExportablePolicyMixin)
    assert PatchPolicyExportMixin.get_supported_export_backends() == []


def test_policy_inherits_empty_export_capability_without_building_model() -> None:
    policy = PatchPolicy(encoder_pretrained=False, hidden_dim=4, num_layers=1, num_heads=1)
    assert isinstance(policy, PatchPolicyExportMixin)
    assert policy.model is None
    assert policy.get_supported_export_backends() == []


def test_backend_list_is_not_shared_mutable_state() -> None:
    backends = PatchPolicyExportMixin.get_supported_export_backends()
    backends.append("onnx")
    assert PatchPolicyExportMixin.get_supported_export_backends() == []
    assert PatchPolicy.get_supported_export_backends() == []


def test_package_exports_policy_config_and_model_classes() -> None:
    assert patch_policy.__all__ == ["PatchPolicy", "PatchPolicyConfig", "PatchPolicyModel"]
    assert patch_policy.PatchPolicy is PatchPolicy
    assert patch_policy.PatchPolicyConfig is PatchPolicyConfig
    assert patch_policy.PatchPolicyModel is PatchPolicyModel
    assert policies.PatchPolicy is PatchPolicy
    assert policies.PatchPolicyConfig is PatchPolicyConfig
    assert policies.PatchPolicyModel is PatchPolicyModel
    assert policies.get_physicalai_policy_class("patch_policy") is PatchPolicy
