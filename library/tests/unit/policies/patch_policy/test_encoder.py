# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU encoder contracts without pretrained downloads."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from physicalai.policies.patch_policy.config import EncoderOutput
from physicalai.policies.patch_policy.components import encoder as encoder_module
from physicalai.policies.patch_policy.components.encoder import (
    TimmFeatureEncoder,
    get_default_encoder_config,
    resolve_encoder_artifacts,
)


class _TokenBackbone(nn.Module):
    """Expose one class token and two patches for each input image."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.num_features = 2
        self.num_prefix_tokens = 1
        self.patch_embed = SimpleNamespace(num_patches=2)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        tokens = images.new_tensor([[10.0, 20.0], [1.0, 2.0], [3.0, 4.0]])
        return tokens[None] + images[:, :1, :1, :1].reshape(-1, 1, 1) * self.weight


def _make_token_encoder(monkeypatch: pytest.MonkeyPatch, output: EncoderOutput = "patches") -> TimmFeatureEncoder:
    create_model = Mock(return_value=_TokenBackbone())
    monkeypatch.setattr(encoder_module, "_import_timm", lambda: SimpleNamespace(create_model=create_model))
    encoder = TimmFeatureEncoder("dinov2", output, image_size=(2, 4))
    create_model.assert_called_once_with(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        img_size=(2, 4),
    )
    return encoder


def test_defaults_are_local_and_return_independent_metadata() -> None:
    assert get_default_encoder_config("dinov2") is None
    assert get_default_encoder_config("resnet18") is None
    config = get_default_encoder_config("webssl")
    assert config["hidden_size"] == 1024
    assert config["num_hidden_layers"] == 24
    assert config["patch_size"] == 14
    config["out_features"].append("changed")
    assert get_default_encoder_config("webssl")["out_features"] == ["stage24"]
    with pytest.raises(ValueError, match="Unsupported encoder_name"):
        get_default_encoder_config("unknown")


def test_backbone_stays_frozen_in_training_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = _make_token_encoder(monkeypatch)
    assert encoder.train() is encoder
    assert encoder.training
    assert not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    output = encoder(torch.ones(1, 3, 2, 4, requires_grad=True))
    assert not output.requires_grad
    torch.testing.assert_close(encoder.mean.flatten(), torch.tensor([0.485, 0.456, 0.406]))
    torch.testing.assert_close(encoder.std.flatten(), torch.tensor([0.229, 0.224, 0.225]))


@pytest.mark.parametrize(
    ("output", "expected"),
    [("patches", [[1.0, 2.0], [3.0, 4.0]]), ("cls", [[10.0, 20.0]]), ("average", [[2.0, 3.0]])],
)
def test_outputs_preserve_batch_time_and_camera_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    output: EncoderOutput,
    expected: list[list[float]],
) -> None:
    encoder = _make_token_encoder(monkeypatch, output)
    image_values = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    images = image_values[..., None, None, None].expand(1, 2, 2, 3, 2, 4)
    tokens = encoder(images)
    expected_tokens = torch.tensor(expected) + image_values[..., None, None]
    torch.testing.assert_close(tokens, expected_tokens)
    assert encoder.output_dim == 2
    assert encoder.num_tokens == len(expected)


def test_nonpretrained_artifacts_do_not_call_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    download = Mock(side_effect=AssertionError("No download expected"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(encoder_module, "_import_timm", download)
    assert resolve_encoder_artifacts("dinov2", pretrained=False) == (None, None)
    assert resolve_encoder_artifacts("resnet18", pretrained=False) == (None, None)
    config, weights = resolve_encoder_artifacts("webssl", pretrained=False)
    assert config == get_default_encoder_config("webssl")
    assert weights is None
    download.assert_not_called()


def test_webssl_provider_uses_pinned_revision_and_loads_cpu_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import huggingface_hub
    from safetensors.torch import save_file

    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.tensor([1.0, 2.0])}, path)
    download = Mock(return_value=str(path))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    config, weights = resolve_encoder_artifacts("webssl", pretrained=True)
    download.assert_called_once_with(
        "facebook/webssl-dino300m-full2b-224",
        "model.safetensors",
        revision="8529cdb3fb75014932af3b896455fc21c386168e",
    )
    assert config == get_default_encoder_config("webssl")
    torch.testing.assert_close(weights["weight"], torch.tensor([1.0, 2.0]))
    assert weights["weight"].device.type == "cpu"


def test_resnet_provider_keys_are_remapped_without_changing_input(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchvision.models

    model = nn.Module()
    model.conv1 = nn.Conv2d(3, 2, 1, bias=False)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(2, 1)
    monkeypatch.setattr(torchvision.models, "resnet18", Mock(return_value=model))
    encoder = TimmFeatureEncoder("resnet18", "average", image_size=(2, 2))
    weights = {"conv1.weight": torch.full((2, 3, 1, 1), 0.5), "fc.weight": torch.tensor([[1.0, 2.0]])}
    original_keys = list(weights)
    encoder.load_pretrained_weights(weights)
    torch.testing.assert_close(encoder.backbone[0].weight, weights["conv1.weight"])
    assert list(weights) == original_keys


def test_webssl_builds_from_local_metadata_without_provider_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub
    from transformers import Dinov2Config, Dinov2Model

    download = Mock(side_effect=AssertionError("Offline construction must not download"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(Dinov2Config, "from_pretrained", download)
    monkeypatch.setattr(Dinov2Model, "from_pretrained", download)
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "image_size": 4,
        "patch_size": 2,
    }

    encoder = TimmFeatureEncoder("webssl", "patches", encoder_config=config, image_size=(4, 4))
    tokens = encoder(torch.zeros(1, 2, 3, 4, 4))

    assert tokens.shape == (1, 2, 4, 8)
    assert encoder.output_dim == 8
    assert encoder.num_tokens == 4
    download.assert_not_called()
