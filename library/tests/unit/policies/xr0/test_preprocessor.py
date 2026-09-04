# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 preprocessor / postprocessor.

These exercise the framework-observation -> XR0-batch mapping, mirroring the
source repository's Qwen3-VL multi-view prompt + ``io`` state/action helpers.
The prompt/vision tests require the Qwen3-VL processor to be available locally
and are skipped otherwise; the normalization round-trip test does not.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from physicalai.data.observation import ACTION, STATE, TASK
from physicalai.policies.xr0.preprocessor import (
    _ASSISTANT_PRIMER,
    _MULTI_VIEW_HEADER,
    _TASK_TEMPLATE,
    ACTION_EPS,
    XR0Postprocessor,
    XR0Preprocessor,
    resize_image,
    normalize_action,
    denormalize_action,
    make_xr0_preprocessors,
    view_title,
)

STATE_DIM = 8
ACTION_DIM = 7
HORIZON = 3


def _stats() -> dict:
    return {
        "observation.state": {"name": "observation.state", "shape": (STATE_DIM,), "mean": [0.0] * STATE_DIM, "std": [1.0] * STATE_DIM},
        "action": {"name": "action", "shape": (ACTION_DIM,), "mean": [0.1] * ACTION_DIM, "std": [2.0] * ACTION_DIM},
    }


def _batch(batch_size: int = 2) -> dict:
    return {
        "images.base": torch.rand(batch_size, 3, 64, 64),
        "images.wrist_left": torch.rand(batch_size, 3, 64, 64),
        STATE: torch.rand(batch_size, STATE_DIM),
        TASK: ["pick up the cube", "open the drawer"][:batch_size],
        ACTION: torch.randn(batch_size, HORIZON, ACTION_DIM),
    }


def _processor_available() -> bool:
    try:
        XR0Preprocessor(camera_views=("base",)).processor  # noqa: B018
    except Exception:  # noqa: BLE001
        return False
    return True


requires_processor = pytest.mark.skipif(
    not _processor_available(),
    reason="Qwen3-VL processor not available locally",
)


@requires_processor
class TestVisionPrompt:
    """Prompt + vision pipeline through the real Qwen3-VL processor."""

    def test_image_grid(self) -> None:
        # image_grid builds the pre-patchify (num_images, C, H, W) grid the
        # exported graph consumes: one entry per (sample, view), sample-major.
        pre, _ = make_xr0_preprocessors(camera_views=("base", "wrist_left"), stats=_stats())
        grid = pre.image_grid(_batch(2))
        assert isinstance(grid, np.ndarray)
        assert grid.dtype == np.float32
        # 2 samples x 2 views of 64x64 (factor-aligned, within budget) RGB images
        assert grid.shape == (4, 3, 64, 64)
        assert np.isfinite(grid).all()

    def test_apply_chat_template(self) -> None:
        # The built message tokenizes into the model input keys via the processor.
        pre, _ = make_xr0_preprocessors(camera_views=("base",), stats=_stats())
        message = pre._build_message("pick up the cube", [Image.new("RGB", (32, 32))])  # noqa: SLF001
        encoded = pre.processor.apply_chat_template(
            [message],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True, "images_kwargs": {"do_resize": False}},
        )
        assert {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"} <= set(encoded)
        assert encoded["input_ids"].shape[0] == 1

    def test_forward_keys_and_shapes(self) -> None:
        # End-to-end glue: batch -> model input keys with batched leading dims.
        pre, _ = make_xr0_preprocessors(camera_views=("base", "wrist_left"), stats=_stats())
        out = pre(_batch(2))
        assert {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state", ACTION, "action_mask"} <= set(
            out
        )
        assert out["input_ids"].shape[0] == 2
        assert out["state"].shape == (2, 1, 32)
        assert out[ACTION].shape == (2, HORIZON, 32)
        assert out["action_mask"].shape == (2, HORIZON, 32)
        # only the real action dims are marked valid.
        assert out["action_mask"][..., :ACTION_DIM].all()
        assert not out["action_mask"][..., ACTION_DIM:].any()


class TestBuildMessage:
    """Multi-view chat message assembly (processor-free)."""

    def test_message_structure(self) -> None:
        pre = XR0Preprocessor(camera_views=("base", "wrist_left"))
        img = Image.new("RGB", (32, 32))
        messages = pre._build_message("pick up the cube", [img, img])  # noqa: SLF001

        assert [m["role"] for m in messages] == ["user", "assistant"]
        user_content = messages[0]["content"]
        # header, then (title, image, newline) per view, then the task text.
        assert user_content[0]["text"] == _MULTI_VIEW_HEADER
        assert user_content[1]["text"] == "# Base View\n"
        assert user_content[2]["type"] == "image"
        assert user_content[4]["text"] == "# Left-Wrist View\n"
        assert user_content[-1]["text"] == _TASK_TEMPLATE.format(instruction="pick up the cube")
        assert messages[1]["content"][0]["text"] == _ASSISTANT_PRIMER


class TestViewTitle:
    """Human-readable camera-view titles embedded in the chat prompt."""

    def test_known_views_use_reference_titles(self) -> None:
        assert view_title("base") == "Base"
        assert view_title("wrist_left") == "Left-Wrist"
        assert view_title("wrist_right") == "Right-Wrist"

    def test_hyphenated_key_is_normalized(self) -> None:
        assert view_title("wrist-left") == "Left-Wrist"

    def test_unknown_view_falls_back_to_capitalized_join(self) -> None:
        assert view_title("front_cam") == "Front Cam"

class TestNumpyActionNormalization:
    """Action normalize/denormalize round-trip."""

    def test_normalize_denormalize_roundtrip(self):
        rng = np.random.default_rng(1)
        action = rng.standard_normal((4, ACTION_DIM)).astype(np.float32)
        mean = rng.standard_normal((4, ACTION_DIM)).astype(np.float32)
        std = np.abs(rng.standard_normal((4, ACTION_DIM)).astype(np.float32)) + 0.1
        normalized = normalize_action(action, mean, std)
        roundtrip = denormalize_action(normalized, mean, std)
        np.testing.assert_allclose(roundtrip, action, atol=1e-5, rtol=1e-5)


class TestResizeImage:
    """VLM image resize keeps factor alignment within the pixel budget."""

    def test_factor_aligned_within_budget(self):
        img = Image.fromarray(np.zeros((200, 300, 3), dtype=np.uint8))
        out = resize_image(img, factor=32, min_pixels=1024, max_pixels=90000)
        w, h = out.size
        assert w % 32 == 0 and h % 32 == 0
        assert w * h <= 90000


class TestExtractViewImages:
    """Per-sample, per-view image extraction + resize (processor-free)."""

    def test_matches_reference(self) -> None:
        pre = XR0Preprocessor(camera_views=("base", "wrist_left"))
        batch = {
            "images.base": torch.zeros(1, 3, 32, 32),
            "images.wrist_left": torch.ones(1, 3, 32, 32),
        }
        images = pre._extract_view_images(batch)  # noqa: SLF001
        grid = torch.stack([torch.from_numpy(np.asarray(img)) for img in images[0]])

        # base -> 0, wrist_left -> 255 (rescaled uint8), each 32x32 RGB, in view order.
        expected = torch.stack([
            torch.zeros(32, 32, 3, dtype=torch.uint8),
            torch.full((32, 32, 3), 255, dtype=torch.uint8),
        ])
        assert len(images) == 1  # one sample
        assert grid.shape == expected.shape
        assert torch.equal(grid, expected)


class TestPrepareAction:
    """Action normalize + pad + validity mask, incl. delta corner cases."""

    @staticmethod
    def _pre(max_action_dim: int = 10, *, action_mode: str = "absolute") -> XR0Preprocessor:
        # Identity stats (mean 0 / std 1) so normalization is a near-passthrough.
        return XR0Preprocessor(
            max_action_dim=max_action_dim,
            action_mode=action_mode,
            action_mean=torch.zeros(max_action_dim),
            action_std=torch.ones(max_action_dim),
        )

    def test_absolute_pad_and_mask(self) -> None:
        pre = self._pre(max_action_dim=10)
        action = torch.randn(2, HORIZON, 4)
        out, mask = pre._prepare_action(action, torch.device("cpu"))  # noqa: SLF001
        assert out.shape == (2, HORIZON, 10)
        # real dims pass through; padding dims are zero.
        assert torch.allclose(out[..., :4], action / (1 + ACTION_EPS), atol=1e-5)
        assert torch.allclose(out[..., 4:], torch.zeros(2, HORIZON, 6))
        assert mask[..., :4].all()
        assert not mask[..., 4:].any()

    def test_does_not_mutate_input(self) -> None:
        pre = self._pre(max_action_dim=10)
        action = torch.randn(2, HORIZON, 4)
        original = action.clone()
        pre._prepare_action(action, torch.device("cpu"))  # noqa: SLF001
        assert torch.equal(action, original)

    def test_equal_dim_no_padding(self) -> None:
        pre = self._pre(max_action_dim=4)
        action = torch.randn(2, HORIZON, 4)
        out, mask = pre._prepare_action(action, torch.device("cpu"))  # noqa: SLF001
        assert out.shape == (2, HORIZON, 4)
        assert mask.all()

    def test_delta_requires_state(self) -> None:
        pre = self._pre(max_action_dim=10, action_mode="delta")
        with pytest.raises(ValueError, match="delta"):
            pre._prepare_action(torch.randn(2, HORIZON, 4), torch.device("cpu"))  # noqa: SLF001

    @pytest.mark.parametrize(
        ("state", "reference"),
        [
            # (B, D) state, state_dim (4) >= action_dim (2): whole action shifted.
            (
                torch.arange(2 * 4).reshape(2, 4).float(),
                torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]], [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]]),
            ),
            # (B, T, D) state uses its current (last) frame.
            (
                torch.arange(2 * 2 * 4).reshape(2, 2, 4).float(),
                torch.tensor([[[-4.0, -4.0, 0.0], [-2.0, -2.0, 0.0]], [[-8.0, -8.0, 0.0], [-6.0, -6.0, 0.0]]]),
            ),
            # state_dim (1) < action_dim (2): only the leading dim gets the delta.
            (
                torch.arange(2 * 1).reshape(2, 1).float(),
                torch.tensor([[[0.0, 1.0, 0.0], [2.0, 3.0, 0.0]], [[3.0, 5.0, 0.0], [5.0, 7.0, 0.0]]]),
            ),
        ],
    )
    def test_delta_matches_reference(self, state: torch.Tensor, reference: torch.Tensor) -> None:
        pre = self._pre(max_action_dim=3, action_mode="delta")
        action = torch.arange(2 * 2 * 2).reshape(2, 2, 2).float()
        out, _ = pre._prepare_action(action, torch.device("cpu"), state=state)  # noqa: SLF001
        assert torch.allclose(out, reference, atol=1e-5)

    @pytest.mark.parametrize(
        ("max_action_dim", "reference"),
        [
            # action_dim (2) < max_action_dim (3): real dims pass through, tail padded.
            (3, torch.tensor([[[0.0, 1.0, 0.0], [2.0, 3.0, 0.0]], [[4.0, 5.0, 0.0], [6.0, 7.0, 0.0]]])),
            # equal dims: no padding.
            (2, torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])),
        ],
    )
    def test_absolute_matches_reference(self, max_action_dim: int, reference: torch.Tensor) -> None:
        pre = self._pre(max_action_dim=max_action_dim, action_mode="absolute")
        action = torch.arange(2 * 2 * 2).reshape(2, 2, 2).float()
        out, mask = pre._prepare_action(action, torch.device("cpu"))  # noqa: SLF001
        assert torch.allclose(out, reference, atol=1e-5)
        # mask has the padded shape: real dims (2) are valid, padding dims are zero.
        assert mask.shape == reference.shape
        assert mask[..., :2].all()
        assert not mask[..., 2:].any()


class TestPostprocessor:
    """Action denormalize + delta re-add + unpad, against references."""

    @staticmethod
    def _post(
        max_action_dim: int = 3,
        *,
        action_mode: str = "absolute",
        action_dim: int | None = None,
    ) -> XR0Postprocessor:
        # Identity stats (mean 0 / std 1) so denormalization is a near-passthrough.
        post = XR0Postprocessor(
            max_action_dim=max_action_dim,
            action_mode=action_mode,
            action_mean=torch.zeros(max_action_dim),
            action_std=torch.ones(max_action_dim),
        )
        post.action_dim = action_dim  # unpadded slice width (None -> no slice)
        return post

    def test_missing_action_is_passthrough(self) -> None:
        post = self._post()
        batch = {STATE: torch.zeros(2, 3)}
        assert post(batch) == batch

    @pytest.mark.parametrize(
        ("action_dim", "reference"),
        [
            # no unpad: full padded width returned.
            (
                None,
                torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]]),
            ),
            # unpad to the original action_dim (2).
            (2, torch.tensor([[[0.0, 1.0], [3.0, 4.0]], [[6.0, 7.0], [9.0, 10.0]]])),
        ],
    )
    def test_absolute_matches_reference(self, action_dim: int | None, reference: torch.Tensor) -> None:
        post = self._post(action_dim=action_dim)
        action = torch.arange(2 * 2 * 3).reshape(2, 2, 3).float()
        out = post({ACTION: action})[ACTION]
        assert out.shape == reference.shape
        assert torch.allclose(out, reference, atol=1e-5)

    def test_delta_requires_state(self) -> None:
        post = self._post(action_mode="delta")
        with pytest.raises(ValueError, match="delta"):
            post({ACTION: torch.zeros(2, 2, 3)})

    @pytest.mark.parametrize(
        ("state", "reference"),
        [
            # (B, D) state, state_dim (3) == action_dim (3): whole action re-added.
            (
                torch.arange(2 * 3).reshape(2, 3).float(),
                torch.tensor([[[0.0, 2.0, 4.0], [3.0, 5.0, 7.0]], [[9.0, 11.0, 13.0], [12.0, 14.0, 16.0]]]),
            ),
            # (B, T, D) state uses its current (last) frame.
            (
                torch.arange(2 * 2 * 3).reshape(2, 2, 3).float(),
                torch.tensor([[[3.0, 5.0, 7.0], [6.0, 8.0, 10.0]], [[15.0, 17.0, 19.0], [18.0, 20.0, 22.0]]]),
            ),
            # state_dim (2) < action_dim (3): only the leading dims get the state.
            (
                torch.arange(2 * 2).reshape(2, 2).float(),
                torch.tensor([[[0.0, 2.0, 2.0], [3.0, 5.0, 5.0]], [[8.0, 10.0, 8.0], [11.0, 13.0, 11.0]]]),
            ),
        ],
    )
    def test_delta_matches_reference(self, state: torch.Tensor, reference: torch.Tensor) -> None:
        post = self._post(action_mode="delta")
        action = torch.arange(2 * 2 * 3).reshape(2, 2, 3).float()
        out = post({ACTION: action, STATE: state})[ACTION]
        assert out.shape == reference.shape
        assert torch.allclose(out, reference, atol=1e-5)


class TestMakeXr0Preprocessors:
    """Factory wiring: stats -> features -> pre/post action stats."""

    def test_absolute_derives_stats_from_features(self) -> None:
        pre, post = make_xr0_preprocessors(max_action_dim=32, stats=_stats())
        assert isinstance(pre, XR0Preprocessor)
        assert isinstance(post, XR0Postprocessor)
        assert pre.action_mode == "absolute"
        # feature action mean/std (0.1 / 2.0) fill the real dims, padding stays 0/1.
        assert torch.allclose(pre.action_mean[:ACTION_DIM], torch.full((ACTION_DIM,), 0.1))
        assert torch.allclose(pre.action_mean[ACTION_DIM:], torch.zeros(32 - ACTION_DIM))
        assert torch.allclose(pre.action_std[:ACTION_DIM], torch.full((ACTION_DIM,), 2.0))
        assert torch.allclose(pre.action_std[ACTION_DIM:], torch.ones(32 - ACTION_DIM))
        # postprocessor recovers the unpadded action_dim for the final slice.
        assert post.action_dim == ACTION_DIM
        assert torch.allclose(post.action_mean, pre.action_mean)
        assert torch.allclose(post.action_std, pre.action_std)

    def test_delta_override_stats_applied_to_pair(self) -> None:
        delta_mean = torch.full((HORIZON, 32), 0.5)
        delta_std = torch.full((HORIZON, 32), 3.0)
        pre, post = make_xr0_preprocessors(
            stats=_stats(),
            action_mode="delta",
            action_delta_mean=delta_mean,
            action_delta_std=delta_std,
        )
        assert pre.action_mode == "delta"
        assert post.action_mode == "delta"
        # explicit delta stats override the feature-derived absolute stats.
        assert torch.allclose(pre.action_mean, delta_mean)
        assert torch.allclose(pre.action_std, delta_std)
        assert torch.allclose(post.action_mean, delta_mean)
        assert torch.allclose(post.action_std, delta_std)
        # the unpadded action_dim is still recovered from features.
        assert post.action_dim == ACTION_DIM

    def test_no_stats_is_identity(self) -> None:
        pre, post = make_xr0_preprocessors(max_action_dim=32, stats=None)
        assert torch.allclose(pre.action_mean, torch.zeros(32))
        assert torch.allclose(pre.action_std, torch.ones(32))
        assert post.action_dim is None