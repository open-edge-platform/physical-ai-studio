# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types

import pytest
import torch

from physicalai.policies.xr0.export_openvino import (
    export_add_deepstack_embeds,
    export_build_additive_causal_mask,
    export_fast_pos_embed_interpolate,
    export_rmsnorm_forward,
    export_rot_pos_emb,
    export_scatter_visual_embeds,
    export_vision_attn_forward,
    install_export_rmsnorm,
)


import numpy as np
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from physicalai.policies.xr0.preprocessor import build_pixel_grid
from physicalai.policies.xr0.export_openvino import patchify_image_grid

# Qwen3-VL geometry (Qwen3-VL reuses the Qwen2-VL image processor with patch_size=16).
TEMPORAL_PATCH_SIZE = 2
PATCH_SIZE = 16
MERGE_SIZE = 2
CHANNELS = 3


def test_numpy_grid_plus_patchify_matches_image_processor() -> None:
    """``build_pixel_grid`` + baked patchify equals the real image processor output.

    Instantiates the actual Qwen2-VL image processor Qwen3-VL uses (offline, no
    download) and checks that the NumPy grid built by ``build_pixel_grid``, once
    patchified by the baked graph op, reproduces the processor's ``pixel_values``
    (and that the geometry matches ``image_grid_thw``). This pins the native NumPy
    image path against the HuggingFace processor it replaces.
    """
    image_processor = Qwen2VLImageProcessor(
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        merge_size=MERGE_SIZE,
    )
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (256, 256, CHANNELS), dtype=np.uint8) for _ in range(2)]

    # Reference: the real processor (images are already patch-aligned -> do_resize=False).
    processed = image_processor(images=images, do_resize=False, return_tensors="np")
    grid_thw = [[int(dim) for dim in row] for row in processed["image_grid_thw"].tolist()]

    grid = build_pixel_grid(
        images,
        image_processor.image_mean,
        image_processor.image_std,
        image_processor.rescale_factor,
    )
    flat = patchify_image_grid(
        torch.from_numpy(grid),
        grid_thw,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        patch_size=PATCH_SIZE,
        merge_size=MERGE_SIZE,
    ).numpy()

    assert grid.shape == (2, CHANNELS, 256, 256)
    assert flat.shape == processed["pixel_values"].shape
    np.testing.assert_allclose(flat, processed["pixel_values"], rtol=0, atol=1e-5)


class TestExportPatchParity:
    """Numerical parity of the export-friendly VLM ops against stock Qwen3-VL.

    ``XR0Qwen3VL._ensure_export_patch`` swaps a handful of stock Qwen3-VL ops for
    OpenVINO-friendly reimplementations (see the ``export_*`` module-level
    functions). These tests check each replacement independently, comparing it to
    the stock ``transformers`` op on small reference tensors. They deliberately
    build tiny weight-free / small-module fixtures instead of loading the 4B model
    so they stay fast and download-free; the export patch is numerically identical
    to stock, so the outputs must match to floating-point tolerance.
    """

    def test_rot_pos_emb_matches_stock(self) -> None:
        """``export_rot_pos_emb`` matches stock ``rot_pos_emb`` (weight-free)."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionModel,
            Qwen3VLVisionRotaryEmbedding,
        )

        torch.manual_seed(0)
        merge_size = 2
        # rotary emb is a deterministic ``inv_freq`` buffer -> no learned weights.
        visual = types.SimpleNamespace(
            spatial_merge_size=merge_size,
            rotary_pos_emb=Qwen3VLVisionRotaryEmbedding(8),
        )
        # Two images (multi-frame on the second) to exercise the repeat path.
        grid_thw = torch.tensor([[1, 4, 6], [2, 2, 8]])

        stock = Qwen3VLVisionModel.rot_pos_emb(visual, grid_thw)
        exported = export_rot_pos_emb(visual, grid_thw.tolist())

        assert exported.shape == stock.shape
        assert torch.equal(exported, stock)

    def test_fast_pos_embed_interpolate_matches_stock(self) -> None:
        """``export_fast_pos_embed_interpolate`` matches the stock interpolation."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        torch.manual_seed(0)
        num_grid_per_side = 6
        hidden = 16
        merge_size = 2
        pos_embed = torch.nn.Embedding(num_grid_per_side * num_grid_per_side, hidden)
        visual = types.SimpleNamespace(
            num_grid_per_side=num_grid_per_side,
            pos_embed=pos_embed,
            config=types.SimpleNamespace(spatial_merge_size=merge_size),
        )
        grid_thw = torch.tensor([[1, 4, 6], [1, 2, 8]])

        stock = Qwen3VLVisionModel.fast_pos_embed_interpolate(visual, grid_thw)
        exported = export_fast_pos_embed_interpolate(visual, grid_thw.tolist())

        assert exported.shape == stock.shape
        assert torch.allclose(exported, stock, atol=1e-6)

    def test_vision_attn_forward_matches_stock(self) -> None:
        """``export_vision_attn_forward`` matches stock attention (SDPA path)."""
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention

        torch.manual_seed(0)
        config = Qwen3VLVisionConfig(hidden_size=32, num_heads=4)
        config._attn_implementation = "sdpa"
        attn = Qwen3VLVisionAttention(config).eval()
        head_dim = config.hidden_size // config.num_heads

        # Two attention windows of 6 and 4 tokens -> seq_len 10.
        split_sizes = [6, 4]
        seq_len = sum(split_sizes)
        cu_seqlens = torch.tensor([0, 6, 10])
        hidden_states = torch.randn(seq_len, config.hidden_size)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)
        position_embeddings = (cos, sin)

        with torch.no_grad():
            stock = attn.forward(
                hidden_states,
                cu_seqlens,
                position_embeddings=position_embeddings,
            )
            exported = export_vision_attn_forward(
                attn,
                split_sizes,
                hidden_states,
                position_embeddings,
            )

        assert exported.shape == stock.shape
        assert torch.allclose(exported, stock, atol=1e-5)

    def test_scatter_visual_embeds_matches_masked_scatter(self) -> None:
        """``export_scatter_visual_embeds`` matches stock ``masked_scatter`` merge."""
        torch.manual_seed(0)
        seq_len, hidden, num_visual = 8, 4, 3
        inputs_embeds = torch.randn(1, seq_len, hidden)
        image_token_indices = torch.tensor([2, 4, 5])
        image_embeds = torch.randn(num_visual, hidden)

        # Stock merge: broadcast a boolean mask and ``masked_scatter``.
        image_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        image_mask[0, image_token_indices] = True
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        stock = inputs_embeds.masked_scatter(image_mask, image_embeds)

        exported = export_scatter_visual_embeds(inputs_embeds, image_token_indices, image_embeds)

        assert exported.shape == stock.shape
        assert torch.equal(exported, stock)

    def test_add_deepstack_embeds_matches_stock(self) -> None:
        """``export_add_deepstack_embeds`` matches stock ``_deepstack_process``."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        torch.manual_seed(0)
        seq_len, hidden, num_visual = 8, 4, 3
        hidden_states = torch.randn(1, seq_len, hidden)
        image_token_indices = torch.tensor([1, 3, 6])
        visual_embeds = torch.randn(num_visual, hidden)

        # Stock deepstack uses a boolean mask over the flattened (batch, seq) grid.
        visual_pos_masks = torch.zeros(1, seq_len, dtype=torch.bool)
        visual_pos_masks[0, image_token_indices] = True
        stock = Qwen3VLTextModel._deepstack_process(
            types.SimpleNamespace(),
            hidden_states,
            visual_pos_masks,
            visual_embeds,
        )

        exported = export_add_deepstack_embeds(hidden_states, image_token_indices, visual_embeds)

        assert exported.shape == stock.shape
        assert torch.allclose(exported, stock, atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "attention_mask",
        [
            [[1, 1, 1, 1, 1]],  # no padding -> pure causal
            [[1, 1, 1, 0, 0]],  # right padding
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],  # batched, mixed padding
        ],
    )
    def test_build_additive_causal_mask_matches_stock(
        self,
        attention_mask: list[list[int]],
        dtype: torch.dtype,
    ) -> None:
        """``export_build_additive_causal_mask`` matches stock ``eager_mask``."""
        from transformers.masking_utils import eager_mask

        mask = torch.tensor(attention_mask, dtype=torch.long)
        batch, seq_len = mask.shape

        stock = eager_mask(
            batch_size=batch,
            q_length=seq_len,
            kv_length=seq_len,
            attention_mask=mask.to(torch.bool),
            dtype=dtype,
        )
        exported = export_build_additive_causal_mask(mask, dtype)

        assert exported.shape == (batch, 1, seq_len, seq_len)
        assert exported.shape == stock.shape
        assert exported.dtype == stock.dtype
        assert torch.equal(exported, stock)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(2, 16), (2, 5, 16), (1, 3, 4, 16)])
    def test_export_rmsnorm_matches_stock(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        """``export_rmsnorm_forward`` matches stock ``Qwen3VLTextRMSNorm``."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

        torch.manual_seed(0)
        hidden = shape[-1]
        norm = Qwen3VLTextRMSNorm(hidden).eval()
        with torch.no_grad():
            # Randomize the weight so the weight-scaling path is exercised.
            norm.weight.copy_(torch.randn(hidden))
        x = torch.randn(*shape, dtype=dtype)
        x_before = x.clone()

        with torch.no_grad():
            # Feed the exact same tensor to both: the export forward must not
            # mutate its input (it reduces over ``dim() - 1`` on an internal
            # float32 copy), so it stays a faithful drop-in even for a float32
            # input where ``.to(float32)`` would otherwise alias ``x``.
            stock = norm(x)
            exported = export_rmsnorm_forward(norm, x)

        assert exported.shape == stock.shape
        assert exported.dtype == stock.dtype
        assert torch.equal(exported, stock)
        # The input must be left untouched (guards the in-place aliasing bug).
        assert torch.equal(x, x_before)


class TestBakeIngraphExport:
    """``XR0._bake_ingraph_export`` pre-export hook wiring (no model download).

    The hook is registered as the OpenVINO ``pre_export_hooks`` entry. It toggles
    the model's ``export_state_passthrough`` from ``action_mode`` and forwards the
    padded export sample to ``prepare_ingraph_export`` (which bakes the vision
    geometry + OpenVINO-friendly RMSNorm on the real 4B model). These tests mock
    that heavy machinery and check only the hook's own wiring.
    """

    @pytest.mark.parametrize(
        ("action_mode", "expected_passthrough"),
        [("absolute", False), ("delta", True)],
    )
    def test_toggles_passthrough_and_forwards_padded_sample(
        self,
        action_mode: str,
        expected_passthrough: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sets ``export_state_passthrough`` from ``action_mode`` and calls prepare."""
        from physicalai.policies.xr0 import XR0

        policy = XR0(action_mode=action_mode)
        # Sentinel model with a settable pass-through flag; avoids the 4B build.
        policy.model = types.SimpleNamespace(export_state_passthrough=None)  # type: ignore[assignment]

        sample = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
        captured: dict[str, object] = {}
        monkeypatch.setattr(policy, "_build_padded_export_sample", lambda: sample)
        monkeypatch.setattr(
            policy,
            "prepare_ingraph_export",
            lambda processed: captured.__setitem__("processed", processed),
        )

        policy._bake_ingraph_export()

        assert policy.model.export_state_passthrough is expected_passthrough
        # The padded sample is forwarded verbatim to ``prepare_ingraph_export``.
        assert captured["processed"] is sample

    def test_noop_passthrough_when_model_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Skips the pass-through toggle (guarded) when no model is built yet."""
        from physicalai.policies.xr0 import XR0

        policy = XR0(action_mode="delta")
        assert policy.model is None

        called: dict[str, object] = {}
        monkeypatch.setattr(policy, "_build_padded_export_sample", lambda: {"x": torch.zeros(1)})
        monkeypatch.setattr(
            policy,
            "prepare_ingraph_export",
            lambda processed: called.__setitem__("processed", processed),
        )

        # Must not raise on the missing model; the guard skips the toggle.
        policy._bake_ingraph_export()

        assert "processed" in called

    def test_install_export_rmsnorm_patches_all_and_is_idempotent(self) -> None:
        """Patches every RMSNorm forward, skips non-RMSNorm, and is idempotent."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

        model = torch.nn.Sequential(
            Qwen3VLTextRMSNorm(16),
            torch.nn.Linear(16, 16),  # non-RMSNorm -> must be skipped.
            torch.nn.Sequential(Qwen3VLTextRMSNorm(16)),  # nested RMSNorm -> covered by the tree walk.
        )

        # First call patches both RMSNorm modules; the Linear is left alone.
        assert install_export_rmsnorm(model) == 2
        # Second call is a no-op: already-patched modules are skipped.
        assert install_export_rmsnorm(model) == 0

    def test_install_export_rmsnorm_swaps_forward_behavior(self) -> None:
        """After install, the RMSNorm forward returns ``export_rmsnorm_forward`` output."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

        torch.manual_seed(0)
        norm = Qwen3VLTextRMSNorm(16).eval()
        with torch.no_grad():
            norm.weight.copy_(torch.randn(16))
        model = torch.nn.Sequential(norm)
        x = torch.randn(2, 16)

        # Reference from the standalone export forward before the swap.
        expected = export_rmsnorm_forward(norm, x.clone())

        assert install_export_rmsnorm(model) == 1
        with torch.no_grad():
            got = norm(x.clone())

        assert torch.equal(got, expected)

