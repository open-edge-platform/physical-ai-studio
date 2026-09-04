# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the assembled XR0 VLA model (``model``).

Self-contained CPU/fp32 tests on a tiny injected Qwen3-VL shim (no downloads).
They cover the framework ``Model`` contract and the ``_run`` train / eval / export
paths, pinned against small reference values.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from physicalai.policies.xr0.export_openvino import patchify_image_grid
from physicalai.policies.xr0.model import XR0Model
from physicalai.policies.xr0.qwen3_vlm import XR0Qwen3VL

IMAGE_TOKEN_ID = 151
VIDEO_TOKEN_ID = 152
VISION_START_TOKEN_ID = 150
IMAGE_GRID = (2, 4, 4)
SPATIAL_MERGE = 2
N_IMAGE_TOKENS = (IMAGE_GRID[0] * IMAGE_GRID[1] * IMAGE_GRID[2]) // SPATIAL_MERGE**2

# In-graph export parity uses a still image (grid_t == 1) so ``patchify_image_grid``
# can reproduce the exact flat ``pixel_values`` the vision tower consumes.
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
EXPORT_GRID = (1, 4, 4)
N_EXPORT_TOKENS = (EXPORT_GRID[0] * EXPORT_GRID[1] * EXPORT_GRID[2]) // SPATIAL_MERGE**2


# Tiny model dims. The DiT head_dim / kv_heads / layer count must match the VLM
# so the DiT can consume the VLM KV-cache.
STATE_LEN = 1
STATE_DIM = 8
ACTION_LEN = 4
ACTION_DIM = 8
DIT_HIDDEN = 64
DIT_HEAD_DIM = 16
DIT_KV_HEADS = 2
DIT_LAYERS = 2
NUM_STEPS = 3

# Reference first action token of the denoised chunk for the seeded tiny model.
REFERENCE_ACTION = torch.tensor([0.4187, 2.1285, -1.0810, 0.1513, 0.3321, -0.3464, 1.1061, 0.6040])


def _config() -> Qwen3VLConfig:
    """Tiny Qwen3-VL config whose head_dim / kv_heads match the DiT."""
    vision = Qwen3VLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        depth=2,
        out_hidden_size=64,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=SPATIAL_MERGE,
        in_channels=3,
        deepstack_visual_indexes=[0],
    )
    text = Qwen3VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=DIT_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=DIT_KV_HEADS,
        head_dim=DIT_HEAD_DIM,
        vocab_size=200,
        rope_scaling={"type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": False},
    )
    return Qwen3VLConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
    )


def _build_model(local_window: int = 4) -> XR0Model:
    """Build the assembled model on a tiny injected VLM (no download)."""
    torch.manual_seed(0)
    vlm = XR0Qwen3VL(_config())
    return XR0Model(
        vlm=vlm,
        state_shape=(STATE_LEN, STATE_DIM),
        action_shape=(ACTION_LEN, ACTION_DIM),
        dit_num_layers=DIT_LAYERS,
        dit_hidden_size=DIT_HIDDEN,
        dit_head_dim=DIT_HEAD_DIM,
        dit_kv_heads=DIT_KV_HEADS,
        num_steps=NUM_STEPS,
        local_window=local_window,
        training_repeat=1,
        dtype=torch.float32,
    )


def _batch() -> dict:
    """Build a deterministic multimodal batch with action / state targets."""
    grid = torch.tensor([list(IMAGE_GRID)])
    num_patches = int(grid.prod(-1).item())
    patch_dim = 3 * 2 * 16 * 16
    torch.manual_seed(0)
    pixel_values = torch.randn(num_patches, patch_dim)
    input_ids = torch.tensor([[5, 6, VISION_START_TOKEN_ID, *([IMAGE_TOKEN_ID] * N_IMAGE_TOKENS), 7, 8, 9]])
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
        "action": torch.randn(1, ACTION_LEN, ACTION_DIM),
        "action_mask": torch.ones(1, ACTION_LEN, ACTION_DIM, dtype=torch.int32),
        "state": torch.randn(1, STATE_LEN, STATE_DIM),
    }


def _export_batches() -> tuple[dict, dict]:
    """Build matching eager / in-graph-export batches for the same observation.

    The eager batch carries the flat patchified ``pixel_values`` the tower
    consumes; the export batch carries the raw ``(1, C, H, W)`` grid the export
    forward patchifies itself. Both share the identical image, action, state and
    seed, so their ``_run`` outputs must match (the export op swaps are
    numerically identical).

    Returns:
        Tuple of ``(eager_batch, export_batch)``.
    """
    grid = torch.tensor([list(EXPORT_GRID)])
    height = EXPORT_GRID[1] * PATCH_SIZE
    width = EXPORT_GRID[2] * PATCH_SIZE
    torch.manual_seed(0)
    raw_image = torch.randn(1, 3, height, width)
    pixel_values = patchify_image_grid(
        raw_image,
        [list(EXPORT_GRID)],
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        patch_size=PATCH_SIZE,
        merge_size=SPATIAL_MERGE,
    )
    input_ids = torch.tensor([[5, 6, VISION_START_TOKEN_ID, *([IMAGE_TOKEN_ID] * N_EXPORT_TOKENS), 7, 8, 9]])
    attention_mask = torch.ones_like(input_ids)
    action = torch.randn(1, ACTION_LEN, ACTION_DIM)
    action_mask = torch.ones(1, ACTION_LEN, ACTION_DIM, dtype=torch.int32)
    state = torch.randn(1, STATE_LEN, STATE_DIM)

    def _with(pixels: torch.Tensor) -> dict:
        return {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "image_grid_thw": grid.clone(),
            "pixel_values": pixels,
            "action": action.clone(),
            "action_mask": action_mask.clone(),
            "state": state.clone(),
            "seed": 1234,
        }

    return _with(pixel_values), _with(raw_image)


@pytest.fixture(scope="module")
def model() -> XR0Model:
    """Build the tiny assembled model once per module (default ``local_window``).
    """
    return _build_model()


class TestDeltaIndices:
    """Framework Model delta-index properties."""

    def test_indices(self, model: XR0Model) -> None:
        assert model.reward_delta_indices is None
        assert model.observation_delta_indices is None
        assert model.action_delta_indices == list(range(ACTION_LEN))


class TestGetActionInput:
    """``get_action_input`` pops provided tensors and fills sensible defaults."""

    def test_pops_provided_tensors(self, model: XR0Model) -> None:
        batch = _batch()
        action, action_mask, state = model.get_action_input(batch)
        assert action.shape == (1, ACTION_LEN, ACTION_DIM)
        assert action_mask.shape == (1, ACTION_LEN, ACTION_DIM)
        assert state.shape == (1, STATE_LEN, STATE_DIM)
        assert action.dtype == torch.float32
        # the tensors are consumed from the batch.
        assert "action" not in batch
        assert "action_mask" not in batch
        assert "state" not in batch

    def test_defaults_action_mask_to_ones(self, model: XR0Model) -> None:
        batch = _batch()
        del batch["action_mask"]
        _, action_mask, _ = model.get_action_input(batch)
        assert action_mask.shape == (1, ACTION_LEN, ACTION_DIM)
        assert action_mask.dtype == torch.int32
        assert torch.all(action_mask == 1)

    def test_defaults_when_action_and_state_absent(self, model: XR0Model) -> None:
        batch = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
        action, action_mask, state = model.get_action_input(batch)
        assert action.shape == (1, ACTION_LEN, ACTION_DIM)
        assert state.shape == (1, STATE_LEN, STATE_DIM)
        assert torch.all(action == 0)
        assert torch.all(state == 0)
        assert torch.all(action_mask == 1)


class TestNormalizePrefixLength:
    """``_normalize_prefix_length`` coerces to a bounded int."""

    @pytest.mark.parametrize(
        ("prefix_length", "action_length", "expected"),
        [
            (None, 4, 0),
            (2, 4, 2),
            (10, 4, 4),
            (-3, 4, 0),
            (torch.tensor(2), 4, 2),
            (torch.tensor([3, 9]), 4, 3),
            (torch.tensor(10), 4, 4),
            (torch.tensor([]), 4, 0),
        ],
    )
    def test_matches_reference(
        self, prefix_length: int | torch.Tensor | None, action_length: int, expected: int
    ) -> None:
        assert XR0Model._normalize_prefix_length(prefix_length, action_length) == expected


class TestMakeLocalCausalMask:
    """``_make_local_causal_mask`` batches the ``[sink, state, action]`` mask."""

    # Full causal mask over [sink, state, action] for state_length=1,
    # action_length=4 (q_len=6); the action-action block is a plain tril.
    _FULL_CAUSAL = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.int32,
    )

    def test_default_shapes_use_cached_buffer(self, model: XR0Model) -> None:
        mask = model._make_local_causal_mask(2, STATE_LEN, ACTION_LEN, torch.device("cpu"), local=True)
        assert mask.shape == (2, 1 + STATE_LEN + ACTION_LEN, 1 + STATE_LEN + ACTION_LEN)
        # local_window=4 >= action_length, so no banding: matches the full causal.
        assert torch.equal(mask, self._FULL_CAUSAL.expand(2, -1, -1))
        # Default shapes reuse the registered buffer rather than recomputing.
        assert torch.equal(mask, model.saved_causal_mask.expand(2, -1, -1))

    def test_non_local_is_full_causal(self, model: XR0Model) -> None:
        mask = model._make_local_causal_mask(1, STATE_LEN, ACTION_LEN, torch.device("cpu"), local=False)
        assert torch.equal(mask[0], self._FULL_CAUSAL)

    def test_local_window_bands_action_block(self) -> None:
        # local_window=1 bands the action-action block to a width-1 diagonal.
        model = _build_model(local_window=1)
        mask = model._make_local_causal_mask(1, STATE_LEN, ACTION_LEN, torch.device("cpu"), local=True)
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 0, 1, 1, 0],
                [1, 1, 0, 0, 1, 1],
            ],
            dtype=torch.int32,
        )
        assert torch.equal(mask[0], expected)


class TestSampleNoise:
    """``_sample_noise`` draws rectified-flow noise across train / eval / seed."""

    @staticmethod
    def _action() -> torch.Tensor:
        return torch.zeros(1, ACTION_LEN, ACTION_DIM)

    def test_shape_and_dtype(self, model: XR0Model) -> None:
        model.eval()
        action = self._action()
        noise = model._sample_noise(action, seed=42)
        assert noise.shape == action.shape
        assert noise.dtype == action.dtype

    def test_eval_seeded_matches_reference(self, model: XR0Model) -> None:
        # Eval + int seed reproduces ``manual_seed`` around an f32 draw cast to
        # the action dtype (the export-friendly deterministic path).
        model.eval()
        action = self._action()
        noise = model._sample_noise(action, seed=42)
        torch.manual_seed(42)
        reference = torch.randn(action.shape, dtype=torch.float32).to(action.dtype)
        assert torch.equal(noise, reference)

    def test_eval_seeded_is_deterministic(self, model: XR0Model) -> None:
        model.eval()
        action = self._action()
        first = model._sample_noise(action, seed=7)
        second = model._sample_noise(action, seed=7)
        assert torch.equal(first, second)

    def test_tensor_seed_uses_first_element(self, model: XR0Model) -> None:
        # A tensor seed is coerced via ``flatten()[0]`` -> same draw as its int.
        model.eval()
        action = self._action()
        int_noise = model._sample_noise(action, seed=5)
        tensor_noise = model._sample_noise(action, torch.tensor([5, 9]))
        assert torch.equal(int_noise, tensor_noise)

    def test_seeded_restores_global_rng_state(self, model: XR0Model) -> None:
        # The seeded eval path must not perturb the global RNG stream.
        model.eval()
        action = self._action()
        torch.manual_seed(123)
        state_before = torch.get_rng_state()
        model._sample_noise(action, seed=42)
        assert torch.equal(torch.get_rng_state(), state_before)

    def test_eval_unseeded_advances_rng(self, model: XR0Model) -> None:
        # Without a seed the eval draw is not restored, so successive calls differ.
        model.eval()
        action = self._action()
        first = model._sample_noise(action, seed=None)
        second = model._sample_noise(action, seed=None)
        assert first.dtype == action.dtype
        assert not torch.equal(first, second)

    def test_training_ignores_seed(self, model: XR0Model) -> None:
        # Training draws via ``randn_like`` from the global RNG and ignores the
        # seed, keeping its RNG stream unchanged.
        model.train()
        action = self._action()
        torch.manual_seed(321)
        noise = model._sample_noise(action, seed=999)
        torch.manual_seed(321)
        reference = torch.randn_like(action)
        assert torch.equal(noise, reference)


class TestRandomMaskPrefix:
    """``_random_mask_prefix`` hides part of the action prefix from the suffix."""

    @staticmethod
    def _mask(size: int = 8) -> torch.Tensor:
        return torch.ones(1, size, size, dtype=torch.int32)

    @pytest.mark.parametrize("prefix_length", [0, 1, 2])
    def test_short_prefix_returns_same_object(self, model: XR0Model, prefix_length: int) -> None:
        # prefix_length <= _PREFIX_KEEP_LAST_K (2): nothing to mask, no clone.
        mask = self._mask()
        out = model._random_mask_prefix(mask, prefix_length=prefix_length, state_length=1)
        assert out is mask

    def test_suffix_out_of_range_returns_same_object(self, model: XR0Model) -> None:
        # suffix_start = 1 + state_length + prefix_length = 6 >= width (6): no-op.
        mask = self._mask(size=6)
        out = model._random_mask_prefix(mask, prefix_length=4, state_length=1)
        assert out is mask

    def test_prob_one_masks_all_but_last_k(self, model: XR0Model) -> None:
        # prob=1.0 -> every maskable prefix column hidden from the suffix rows;
        # the trailing _PREFIX_KEEP_LAST_K (2) prefix columns stay visible.
        # state_length=1 -> action_start=2; prefix_length=4 -> mask cols [2:4],
        # keep cols 4,5; suffix_start=6 -> only rows 6,7 change.
        model.prefix_mask_prob = 1.0
        mask = self._mask()
        out = model._random_mask_prefix(mask, prefix_length=4, state_length=1)
        expected = self._mask()
        expected[:, 6:, 2:4] = 0
        assert torch.equal(out, expected)
        # Input is cloned, not mutated in place.
        assert torch.all(mask == 1)

    def test_prob_zero_clones_without_masking(self, model: XR0Model) -> None:
        # prob=0.0 -> nothing hidden, but the returned mask is still a clone.
        model.prefix_mask_prob = 0.0
        mask = self._mask()
        out = model._random_mask_prefix(mask, prefix_length=4, state_length=1)
        assert torch.equal(out, mask)
        assert out is not mask

    def test_state_length_shifts_action_start(self, model: XR0Model) -> None:
        # state_length=2 -> action_start=3; prefix_length=3 -> mask col [3:4],
        # keep cols 4,5; suffix_start=6 -> only rows 6,7 change.
        model.prefix_mask_prob = 1.0
        mask = self._mask()
        out = model._random_mask_prefix(mask, prefix_length=3, state_length=2)
        expected = self._mask()
        expected[:, 6:, 3:4] = 0
        assert torch.equal(out, expected)


class TestRun:
    """``_run`` end-to-end: eval action chunk and training loss dict.

    The shared ``model`` fixture is reused for every run (no rebuild). Training
    draws its noise / timestep from the global RNG, so each training test builds
    the batch first (``_batch`` reseeds to 0 internally) and then seeds with
    ``_TRAIN_SEED`` right before ``_run``. Eval passes an in-batch ``seed`` so it
    is deterministic regardless of the global RNG.

    Reference values are placeholders (zeros); fill them from a trusted run.
    """

    _EVAL_SEED = 1234
    _TRAIN_SEED = 0

    # Fill from a trusted run.
    _LOSS_REF = {"loss": 0.8410, "loss_mse": 1.6821, "loss_freq": 0.0}

    @pytest.mark.parametrize(
        "expected_action",
        # Fill from a trusted run.
        [torch.tensor([[[ 0.0448, -0.6232,  0.6625, -1.5139, -0.2641,  1.2134, -0.2071,
          -0.4787],
         [ 0.2335,  0.8937, -0.7311, -0.6636, -0.2807, -1.6301,  0.5070,
          -0.6718],
         [ 1.0003, -0.0866, -0.0545, -2.2477, -0.2620,  1.2205, -0.6802,
          -0.6978],
         [ 0.1304, -1.6699,  0.4045,  0.0757, -1.5303,  0.7214, -0.5941,
           1.3126]]])],
    )
    def test_eval_returns_reference_action(self, model: XR0Model, expected_action: torch.Tensor) -> None:
        model.eval()
        batch = _batch()
        batch["seed"] = self._EVAL_SEED
        pred = model._run(batch, return_loss=False)
        assert pred.shape == (1, ACTION_LEN, ACTION_DIM)
        assert torch.allclose(pred, expected_action, atol=1e-4)

    def test_training_returns_reference_loss(self, model: XR0Model) -> None:
        model.train()
        model.freq_coefficient = 0.0
        batch = _batch()
        torch.manual_seed(self._TRAIN_SEED)
        out = model._run(batch, return_loss=True)
        assert set(out) == set(self._LOSS_REF)
        for key, expected in self._LOSS_REF.items():
            assert torch.allclose(out[key], torch.tensor(expected), atol=1e-4)

    def test_training_reuses_model_deterministically(self, model: XR0Model) -> None:
        # Same seed + fresh (deterministic) batch on the reused model -> identical
        # loss across runs, confirming no residual per-run state.
        model.train()
        model.freq_coefficient = 0.0
        batch = _batch()
        torch.manual_seed(self._TRAIN_SEED)
        first = model._run(batch, return_loss=True)["loss"]
        batch = _batch()
        torch.manual_seed(self._TRAIN_SEED)
        second = model._run(batch, return_loss=True)["loss"]
        assert torch.equal(first, second)

    @pytest.mark.parametrize(
        "expected_loss",
        [3.0285],
    )
    def test_training_freq_term_active(self, model: XR0Model, expected_loss: float) -> None:
        # freq_coefficient > 0 adds the frequency-domain term to the total loss.
        model.train()
        model.freq_coefficient = 1.0
        try:
            batch = _batch()
            torch.manual_seed(self._TRAIN_SEED)
            out = model._run(batch, return_loss=True)
            assert out["loss_freq"].item() != 0.0
            assert torch.allclose(out["loss"], torch.tensor(expected_loss), atol=1e-4)
        finally:
            model.freq_coefficient = 0.0


@pytest.fixture
def export_model() -> XR0Model:
    """A fresh eval-mode model for a single export bake (never the shared one).

    ``prepare_ingraph_export`` irreversibly monkeypatches the VLM instance, so
    the module-scoped ``model`` fixture cannot be reused; this is function-scoped
    to give every export test a clean instance.
    """
    return _build_model().eval()


@pytest.fixture(scope="module")
def eager_export_pred() -> torch.Tensor:
    """Eager ``_run`` action for the export observation, computed once.

    Weight-deterministic and read-only, so it is safe to share across the export
    tests as the reference the baked export output must reproduce.
    """
    eager_batch, _ = _export_batches()
    return _build_model().eval()._run(eager_batch, return_loss=False)


class TestRunExport:
    """``_run`` in-graph export branch: f32 output(s) matching the eager run.

    The op swaps installed by ``prepare_ingraph_export`` are numerically
    identical, so the export ``_run`` must reproduce the eager ``_run`` for the
    same observation.
    """

    @staticmethod
    def _bake_export(model: XR0Model, batch: dict) -> None:
        model.prepare_ingraph_export(
            batch["input_ids"],
            batch["attention_mask"],
            batch["image_grid_thw"],
        )

    def test_ingraph_export_matches_eager(self, export_model: XR0Model, eager_export_pred: torch.Tensor) -> None:
        # Single f32 action output, numerically equal to the eager eval output.
        _, export_batch = _export_batches()
        self._bake_export(export_model, export_batch)
        export_pred = export_model._run(export_batch, return_loss=False)

        assert isinstance(export_pred, torch.Tensor)
        assert export_pred.dtype == torch.float32
        assert export_pred.shape == (1, ACTION_LEN, ACTION_DIM)
        assert torch.allclose(export_pred, eager_export_pred.float(), atol=1e-3, rtol=1e-3)

    def test_ingraph_export_state_passthrough(
        self, export_model: XR0Model, eager_export_pred: torch.Tensor
    ) -> None:
        # delta-mode export echoes the current-frame state as a second f32 output
        # and the action still matches the eager run.
        _, export_batch = _export_batches()
        expected_state = export_batch["state"].clone()

        self._bake_export(export_model, export_batch)
        export_model.export_state_passthrough = True
        pred, state = export_model._run(export_batch, return_loss=False)

        assert pred.shape == (1, ACTION_LEN, ACTION_DIM)
        assert state.shape == (1, STATE_LEN, STATE_DIM)
        assert pred.dtype == torch.float32
        assert state.dtype == torch.float32
        # The second output is the current-frame state echoed verbatim (as f32).
        assert torch.allclose(state, expected_state.float(), atol=1e-6)
        assert torch.allclose(pred, eager_export_pred.float(), atol=1e-3, rtol=1e-3)

