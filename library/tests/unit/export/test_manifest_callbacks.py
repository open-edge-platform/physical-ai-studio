# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for callback components in exported manifests."""

from types import SimpleNamespace

import pytest
import torch

from physicalai.export.backends import ONNXExportParameters, OpenVINOExportParameters
from physicalai.export.mixin_policy import ExportBackend, ExportablePolicyMixin
from physicalai.inference.manifest import ComponentSpec, Manifest, ModelSpec
from physicalai.policies.rldx1.export import Rldx1ExportMixin


_CALLBACKS_SCHEMA_SKIP_REASON = "Runtime ModelSpec.callbacks is unavailable; waiting for the separate Runtime dependency update"


class _Policy(ExportablePolicyMixin):
    """Minimal policy used to exercise manifest creation."""

    model = torch.nn.Identity()

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        return [ExportBackend.ONNX]


@pytest.mark.skipif("callbacks" not in ModelSpec.model_fields, reason=_CALLBACKS_SCHEMA_SKIP_REASON)
def test_create_manifest_preserves_callback_order(tmp_path) -> None:
    """Callback components are stored under model.callbacks in declaration order."""
    callbacks = [
        ComponentSpec(type="first_callback", value=1),
        ComponentSpec(type="second_callback", value=2),
    ]

    _Policy().create_manifest(
        tmp_path,
        ExportBackend.ONNX,
        runner=ComponentSpec(type="single_pass"),
        callbacks=callbacks,
    )

    manifest = Manifest.load(tmp_path / "manifest.json")
    assert [callback.type for callback in manifest.model.callbacks] == ["first_callback", "second_callback"]
    assert manifest.model_dump()["model"]["callbacks"] == [callback.model_dump() for callback in callbacks]


@pytest.mark.skipif("callbacks" not in ModelSpec.model_fields, reason=_CALLBACKS_SCHEMA_SKIP_REASON)
def test_export_parameters_default_to_no_callbacks(tmp_path) -> None:
    """Existing exporters retain an empty callback list by default."""
    policy = _Policy()
    policy.create_manifest(
        tmp_path,
        ExportBackend.ONNX,
        runner=ComponentSpec(type="single_pass"),
    )

    manifest = Manifest.load(tmp_path / "manifest.json")
    assert manifest.model.callbacks == []
    assert ONNXExportParameters().callbacks_specs == []
    assert OpenVINOExportParameters().callbacks_specs == []


def test_rldx1_onnx_and_openvino_callbacks_use_config_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """RLDX ONNX and OpenVINO parameters declare the runtime VTC callback."""
    policy = object.__new__(Rldx1ExportMixin)
    policy.config = SimpleNamespace(
        action_horizon=2,
        compress_to_fp16=False,
        max_state_dim=8,
        n_cog_tokens=0,
        num_views=1,
        tokenizer_max_length=32,
        video_length=5,
        video_stride=3,
    )
    policy._dataset_stats = {
        "observation.state": {"type": "STATE", "shape": [8]},
        "observation.images.front": {"type": "VISUAL", "shape": [3, 16, 16]},
        "action": {"type": "ACTION", "shape": [4]},
    }
    policy._preprocessor = SimpleNamespace(tokenizer=object())
    policy.model = None
    policy._camera_names = ["front"]

    monkeypatch.setattr(
        "physicalai.policies.rldx1.export.build_rldx1_token_composer_params",
        lambda **_: {},
    )

    args = policy.extra_export_args
    for backend in (ExportBackend.ONNX, ExportBackend.OPENVINO):
        callbacks = args[backend.value].callbacks_specs
        assert [callback.type for callback in callbacks] == ["rldx1_vtc"]
        assert callbacks[0].model_dump()["video_length"] == 5
        assert callbacks[0].model_dump()["video_stride"] == 3
    assert args[ExportBackend.TORCH.value].callbacks_specs == []
