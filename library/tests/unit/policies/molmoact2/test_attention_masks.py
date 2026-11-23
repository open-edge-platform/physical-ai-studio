# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Padding regressions for the reduced-precision softmax failure in PR #1415."""

from types import SimpleNamespace

import pytest
import torch

from physicalai.policies.molmoact2.components.action_expert import ActionExpert
from physicalai.policies.molmoact2.components.backbone import MolmoAct2Backbone
from physicalai.policies.molmoact2.components.vision import MolmoAct2VisionBackbone


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("image_tokens", [False, True])
def test_right_padded_text_has_no_fully_masked_rows(dtype: torch.dtype, image_tokens: bool) -> None:
    lengths = torch.tensor([1, 3, 7])
    valid = torch.arange(7)[None, :] < lengths[:, None]
    token_types = torch.zeros(3, 7, dtype=torch.long)
    if image_tokens:
        token_types[1:, 1:3] = 1
    bias = MolmoAct2Backbone._build_attention_bias(torch.zeros(3, 7, 8, dtype=dtype), valid, token_types)

    assert (bias == 0).any(dim=-1).all()
    assert not (bias[:, 0] == 0)[~valid[:, None, :].expand(-1, 7, -1)].any()
    # Emulate the backend's overflow of a finite mask fill to -inf. Plain
    # softmax intentionally avoids Torch SDPA's all-masked-row special case.
    overflowed = bias.float().masked_fill(bias < 0, float("-inf"))
    probabilities = torch.softmax(torch.randn(3, 2, 7, 7) + overflowed, dim=-1)
    assert torch.isfinite(probabilities).all()
    # With no image-token exception, real-token rows retain ordinary causality.
    if not image_tokens:
        expected = valid[:, None, :] & torch.ones(7, 7, dtype=torch.bool).tril()
        torch.testing.assert_close(bias[:, 0] == 0, expected)


@pytest.mark.parametrize("causal", [False, True])
def test_action_expert_masks_have_visible_keys_after_overflow(causal: bool) -> None:
    expert = SimpleNamespace(causal_attn=causal, blocks=[])
    encoder_mask = torch.tensor([[True, False, False], [True, True, False]])
    horizon_padding = torch.tensor([[False, True, True, True], [False, False, True, True]])
    cross_mask, self_mask, _, _ = ActionExpert.prepare_context_metadata(
        expert,
        encoder_attention_mask=encoder_mask,
        seq_len=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        action_horizon_is_pad=horizon_padding,
    )
    for mask in (cross_mask, self_mask):
        assert (mask == 0).any(dim=-1).all()
        overflowed = mask.masked_fill(mask < 0, float("-inf"))
        probabilities = torch.softmax(torch.randn(2, 2, 4, mask.shape[-1]) + overflowed, dim=-1)
        assert torch.isfinite(probabilities).all()


def test_pooled_image_padding_discards_internal_nan_rows() -> None:
    internal_outputs: list[torch.Tensor] = []

    def attention(query: torch.Tensor, values: torch.Tensor, *, attn_mask: torch.Tensor) -> torch.Tensor:
        # Model a backend without SDPA's fully-masked-row special handling.
        probabilities = torch.softmax(
            torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(~attn_mask, float("-inf")), dim=-1
        )
        output = probabilities[:, 0] @ values
        internal_outputs.append(output)
        return output

    vision = SimpleNamespace(
        encode_image=lambda images: images,
        image_feature_dropout=torch.nn.Identity(),
        image_pooling_2d=attention,
        image_projector=torch.nn.Identity(),
        pooling_attention_mask=True,
        dtype=torch.float32,
    )
    indices = torch.tensor([[[0, 1], [-1, -1]], [[2, 3], [-1, -1]]])
    output = MolmoAct2VisionBackbone.forward(vision, torch.randn(2, 1, 4, 8), indices)

    assert torch.isnan(internal_outputs[0]).any()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, 1], torch.zeros(2, 8))


@pytest.mark.parametrize("precision", ["f32", "bf16", "f16"])
def test_right_padded_text_attention_is_finite_on_openvino_cpu(precision: str) -> None:
    ov = pytest.importorskip("openvino")
    core = ov.Core()
    capabilities = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    capability = {"bf16": "BF16", "f16": "FP16"}.get(precision)
    if capability is not None and capability not in capabilities:
        pytest.skip(f"CPU does not support {precision} inference")

    class TextAttention(torch.nn.Module):
        def forward(
            self,
            embeddings: torch.Tensor,
            valid: torch.Tensor,
            scores: torch.Tensor,
            values: torch.Tensor,
        ) -> torch.Tensor:
            bias = MolmoAct2Backbone._build_attention_bias(embeddings, valid, None)
            # Use explicit softmax, not Torch SDPA's all-masked-row fallback.
            return torch.softmax(scores + bias, dim=-1) @ values

    inputs = (
        torch.zeros(3, 7, 8),
        torch.arange(7)[None, :] < torch.tensor([1, 3, 7])[:, None],
        torch.randn(3, 2, 7, 7),
        torch.randn(3, 2, 7, 8),
    )
    model = TextAttention().eval()
    converted = ov.convert_model(model, example_input=inputs)
    compiled = core.compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": precision})
    actual = torch.from_numpy(compiled([value.numpy() for value in inputs])[0])

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, model(*inputs), rtol=0.05, atol=0.05)
