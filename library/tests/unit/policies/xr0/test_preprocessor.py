# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 preprocessor / postprocessor.

These exercise the framework-observation -> XR0-batch mapping, mirroring the
source repository's Qwen3-VL multi-view prompt + ``io`` state/action helpers.
The prompt/vision tests require the Qwen3-VL processor to be available locally
and are skipped otherwise; the normalization round-trip test does not.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from physicalai.data.observation import ACTION, STATE, TASK, Feature, FeatureType, NormalizationParameters
from physicalai.policies.xr0.preprocessor import (
    _ASSISTANT_PRIMER,
    _MULTI_VIEW_HEADER,
    _TASK_TEMPLATE,
    ACTION_MASK,
    ACTION_EPS,
    XR0Postprocessor,
    XR0Preprocessor,
    _augment_images,
    _resize_batch,
    _to_chunk_stats,
    _normalize_action,
    _denormalize_action,
    make_xr0_preprocessors,
    _validate_view_map,
    _view_title,
)

STATE_DIM = 8
ACTION_DIM = 7
HORIZON = 3


def _stats() -> dict:
    # Action stats are per-timestep ``(HORIZON, ACTION_DIM)``; the state is a
    # single frame, so its stats stay a plain ``(STATE_DIM,)`` vector.
    return {
        "observation.state": {"name": "observation.state", "shape": (STATE_DIM,), "mean": [0.0] * STATE_DIM, "std": [1.0] * STATE_DIM},
        "action": {
            "name": "action",
            "shape": (ACTION_DIM,),
            "mean": [[0.1] * ACTION_DIM] * HORIZON,
            "std": [[2.0] * ACTION_DIM] * HORIZON,
        },
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
        XR0Preprocessor().processor  # noqa: B018
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

    def test_apply_chat_template(self) -> None:
        # The built message tokenizes into the model input keys via the processor.
        pre, _ = make_xr0_preprocessors(stats=_stats(), chunk_size=HORIZON)
        message = pre._build_message("pick up the cube", ["base"], [Image.new("RGB", (32, 32))])  # noqa: SLF001
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
        pre, _ = make_xr0_preprocessors(stats=_stats(), chunk_size=HORIZON)
        out = pre(_batch(2))
        assert {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", STATE, ACTION, ACTION_MASK} <= set(
            out
        )
        assert out["input_ids"].shape[0] == 2
        assert out[STATE].shape == (2, 1, 32)
        assert out[ACTION].shape == (2, HORIZON, 32)
        assert out[ACTION_MASK].shape == (2, HORIZON, 32)
        # only the real action dims are marked valid.
        assert out[ACTION_MASK][..., :ACTION_DIM].all()
        assert not out[ACTION_MASK][..., ACTION_DIM:].any()


class TestBuildMessage:
    """Multi-view chat message assembly (processor-free)."""

    def test_message_structure(self) -> None:
        pre = XR0Preprocessor()
        img = Image.new("RGB", (32, 32))
        messages = pre._build_message("pick up the cube", ["base", "wrist_left"], [img, img])  # noqa: SLF001

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
        assert _view_title("ego") == "Ego"
        assert _view_title("base") == "Base"
        assert _view_title("wrist_left") == "Left-Wrist"
        assert _view_title("wrist_right") == "Right-Wrist"

    def test_hyphenated_key_is_normalized(self) -> None:
        assert _view_title("wrist-left") == "Left-Wrist"

    def test_unknown_view_falls_back_to_capitalized_join(self) -> None:
        assert _view_title("front_cam") == "Front Cam"

class TestNumpyActionNormalization:
    """Action normalize/denormalize round-trip."""

    def test_normalize_denormalize_roundtrip(self):
        rng = np.random.default_rng(1)
        action = rng.standard_normal((4, ACTION_DIM)).astype(np.float32)
        mean = rng.standard_normal((4, ACTION_DIM)).astype(np.float32)
        std = np.abs(rng.standard_normal((4, ACTION_DIM)).astype(np.float32)) + 0.1
        normalized = _normalize_action(action, mean, std)
        roundtrip = _denormalize_action(normalized, mean, std)
        np.testing.assert_allclose(roundtrip, action, atol=1e-5, rtol=1e-5)


class TestResizeImage:
    """VLM image resize keeps factor alignment within the pixel budget."""

    def test_factor_aligned_within_budget(self):
        images = torch.zeros(2, 3, 200, 300, dtype=torch.uint8)
        out = _resize_batch(images, factor=32, max_pixels=90000)
        b, c, h, w = out.shape
        assert (b, c) == (2, 3)
        assert out.dtype == torch.uint8
        assert w % 32 == 0 and h % 32 == 0
        assert w * h <= 90000

    def test_channels_last_and_float_input(self):
        # (B, H, W, C) float images in [0, 1] are permuted and rescaled to uint8.
        images = torch.ones(1, 64, 64, 3)
        out = _resize_batch(images, factor=32, max_pixels=90000)
        assert out.shape == (1, 3, 64, 64)
        assert out.dtype == torch.uint8
        assert int(out.max()) == 255


def test_augment_images_seeded_reference() -> None:
    """Random RGB views produce the expected seeded crop and color jitter."""
    generator = torch.Generator().manual_seed(41)
    images = [torch.randint(0, 256, (3, 3, 3), generator=generator, dtype=torch.uint8) for _ in range(2)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)  # Enables brightness and contrast; crops differ per view.
        augmented = _augment_images(images)

    expected = [
        torch.tensor(
            [
                [[161, 150, 137], [119, 110, 99], [76, 68, 61]],
                [[22, 64, 106], [102, 110, 118], [182, 156, 128]],
                [[81, 82, 83], [91, 112, 129], [102, 140, 178]],
            ],
            dtype=torch.uint8,
        ),
        torch.tensor(
            [
                [[179, 180, 181], [160, 186, 213], [142, 193, 244]],
                [[49, 32, 14], [119, 93, 66], [187, 154, 119]],
                [[55, 91, 126], [32, 74, 119], [7, 58, 111]],
            ],
            dtype=torch.uint8,
        ),
    ]
    assert all(torch.equal(actual, reference) for actual, reference in zip(augmented, expected, strict=True))


@pytest.mark.parametrize(
    ("dim", "expected_mean", "expected_std"),
    [
        (3, [[2, 4, 0], [6, 8, 0]], [[1, 3, 1], [5, 7, 1]]),
        (2, [[2, 4], [6, 8]], [[1, 3], [5, 7]]),
        (1, [[2], [6]], [[1], [5]]),
    ],
)
def test_to_chunk_stats_pad_copy_or_truncate(dim: int, expected_mean: list[list[int]], expected_std: list[list[int]]) -> None:
    """Keep per-timestep values; missing dimensions use identity stats."""
    mean, std = _to_chunk_stats([[2, 4], [6, 8]], [[1, 3], [5, 7]], chunk_size=2, dim=dim, name="action")
    assert mean.dtype == std.dtype == torch.float32
    assert torch.equal(mean, torch.tensor(expected_mean, dtype=torch.float32))
    assert torch.equal(std, torch.tensor(expected_std, dtype=torch.float32))


def test_to_chunk_stats_accepts_1d_for_single_timestep() -> None:
    """A length-one chunk can use per-dimension vector stats."""
    mean, std = _to_chunk_stats([2, 4], torch.tensor([1, 3], dtype=torch.float64), chunk_size=1, dim=3, name="state")
    assert torch.equal(mean, torch.tensor([[2.0, 4.0, 0.0]]))
    assert torch.equal(std, torch.tensor([[1.0, 3.0, 1.0]]))


@pytest.mark.parametrize(
    ("mean", "std", "chunk_size", "message"),
    [
        ([[2, 4]], [[1]], 1, "action mean/std must have the same shape"),
        ([2, 4], [1, 3], 2, "action stats must be per-timestep"),
        ([[2, 4]], [[1, 3]], 2, "action stats must be per-timestep"),
    ],
)
def test_to_chunk_stats_rejects_invalid_shapes(
    mean: list, std: list, chunk_size: int, message: str
) -> None:
    """Reject mismatched shapes and missing or wrong temporal axes."""
    with pytest.raises(ValueError, match=message):
        _to_chunk_stats(mean, std, chunk_size=chunk_size, dim=3, name="action")


class TestExtractViewImages:
    """Per-sample, per-view image extraction + resize (processor-free)."""

    def test_matches_reference(self) -> None:
        pre = XR0Preprocessor()
        batch = {
            "images.base": torch.zeros(1, 3, 32, 32),
            "images.wrist_left": torch.ones(1, 3, 32, 32),
        }
        views, images = pre._extract_view_images(batch)  # noqa: SLF001
        grid = torch.stack(images[0])

        # base -> 0, wrist_left -> 255 (rescaled uint8), each 32x32 RGB, in view order.
        expected = torch.stack([
            torch.zeros(3, 32, 32, dtype=torch.uint8),
            torch.full((3, 32, 32), 255, dtype=torch.uint8),
        ])
        assert views == ["base", "wrist_left"]
        assert len(images) == 1  # one sample
        assert grid.shape == expected.shape
        assert torch.equal(grid, expected)

    def test_selected_in_observation_order(self) -> None:
        # Views follow the observation key insertion order, so the images stay
        # aligned with the prompt view sections.
        pre = XR0Preprocessor()
        batch = {
            "images.wrist_left": torch.ones(1, 3, 32, 32),
            "images.base": torch.zeros(1, 3, 32, 32),
        }
        views, images = pre._extract_view_images(batch)  # noqa: SLF001
        sample = images[0]
        # First image is wrist_left (255), second is base (0).
        assert views == ["wrist_left", "base"]
        assert int(sample[0].max()) == 255
        assert int(sample[1].max()) == 0

    def test_temporal_images_use_last_frame_per_sample(self) -> None:
        """Take the latest image from each sample's history."""
        frames = torch.zeros(2, 2, 3, 32, 32)
        frames[0, 0] = 1  # The first sample's earlier frame must be ignored.
        frames[1, 1] = 1
        views, images = XR0Preprocessor()._extract_view_images({"images.base": frames})  # noqa: SLF001
        assert views == ["base"]
        assert len(images) == 2
        assert torch.equal(images[0][0], torch.zeros(3, 32, 32, dtype=torch.uint8))
        assert torch.equal(images[1][0], torch.full((3, 32, 32), 255, dtype=torch.uint8))

    def test_augment_images_flag_applies_per_sample(self) -> None:
        """Only augment when requested, using each sample's returned images."""
        batch = {"images.base": torch.stack([torch.zeros(3, 32, 32), torch.ones(3, 32, 32)])}
        pre = XR0Preprocessor()
        with patch(
            "physicalai.policies.xr0.preprocessor._augment_images",
            side_effect=lambda sample: [255 - image for image in sample],
        ) as augment:
            _, plain = pre._extract_view_images(batch)  # noqa: SLF001
            augment.assert_not_called()
            _, augmented = pre._extract_view_images(batch, augment_images=True)  # noqa: SLF001
        assert augment.call_count == 2
        assert torch.equal(augmented[0][0], 255 - plain[0][0])
        assert torch.equal(augmented[1][0], 255 - plain[1][0])

    def test_augmentation_crops_after_resize_without_changing_geometry(self) -> None:
        pre = XR0Preprocessor()
        image = torch.arange(64 * 64, dtype=torch.int32).reshape(64, 64).remainder(256).to(torch.uint8)
        batch = {"images.base": image.expand(1, 3, -1, -1)}
        _, plain = pre._extract_view_images(batch)  # noqa: SLF001
        with (
            patch(
                "physicalai.policies.xr0.preprocessor.torch.rand",
                return_value=torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0]),
            ),
            patch("physicalai.policies.xr0.preprocessor.torch.randint", return_value=torch.tensor(1)) as crop_draws,
        ):
            _, augmented = pre._extract_view_images(batch, augment_images=True)  # noqa: SLF001
        assert crop_draws.call_count == 2
        assert augmented[0][0].shape == plain[0][0].shape == (3, 64, 64)
        assert augmented[0][0].dtype == torch.uint8
        assert not torch.equal(augmented[0][0], plain[0][0])

    def test_color_gates_and_factors_shared_across_views(self) -> None:
        pre = XR0Preprocessor()
        image = torch.full((1, 3, 64, 64), 128, dtype=torch.uint8)
        batch = {"images.base": image, "images.wrist_left": image}
        with patch(
            "physicalai.policies.xr0.preprocessor.torch.rand",
            return_value=torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
        ) as draws:
            _, images = pre._extract_view_images(batch, augment_images=True)  # noqa: SLF001
        draws.assert_called_once_with(6, device=image.device)
        assert torch.equal(images[0][0], images[0][1])
        assert int(images[0][0].max()) < 128


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        {},
        {"images.ego": "ego", "images.base": "base", "images.left": "wrist_left", "images.right": "wrist_right"},
    ],
)
def test_validate_view_map_returns_copy(mapping: dict[str, str] | None) -> None:
    """Accept canonical views and return an independent mapping."""
    validated = _validate_view_map(mapping)
    assert validated == (mapping or {})
    if mapping is not None:
        assert validated is not mapping


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"images.front": "front_cam"}, "canonical XR0 view names"),
        ({"images.first": "ego", "images.second": "ego"}, "must be unique"),
    ],
)
def test_validate_view_map_rejects_invalid_views(mapping: dict[str, str], message: str) -> None:
    """Reject unknown and duplicate canonical views."""
    with pytest.raises(ValueError, match=message):
        _validate_view_map(mapping)


class TestImageKeyViewMap:
    """Dataset image keys renamed onto the canonical XR0 view names."""

    @staticmethod
    def _batch() -> dict:
        # Keys and order of the local ``Put-the-yellow-ball-to-the-black-box``
        # dataset: neither name is a canonical XR0 view.
        return {
            "images.pov_black_follower_camera": torch.ones(1, 3, 32, 32),
            "images.top_camera": torch.zeros(1, 3, 32, 32),
        }

    def test_renames_and_reorders_canonically(self) -> None:
        # The map is declared in dataset order, but the prompt must follow the
        # canonical order the pretrained checkpoint was trained with.
        pre = XR0Preprocessor(
            image_key_view_map={
                "images.pov_black_follower_camera": "wrist_left",
                "images.top_camera": "ego",
            },
        )
        views, images = pre._extract_view_images(self._batch())  # noqa: SLF001
        sample = images[0]
        assert views == ["ego", "wrist_left"]
        # ``ego`` is top_camera (0) and ``wrist_left`` is the follower cam (255).
        assert int(sample[0].max()) == 0
        assert int(sample[1].max()) == 255

    def test_prompt_matches_pretrain_reference(self) -> None:
        # Byte-for-byte parity with the reference prompt of the base Pretrain
        # checkpoint (xr0/docs/data_format.md, mibot/server/runtime/client.py).
        pre = XR0Preprocessor(image_key_view_map={"images.top_camera": "ego", "images.pov_black_follower_camera": "wrist_left"})
        views, _ = pre._extract_view_images(self._batch())  # noqa: SLF001
        img = Image.new("RGB", (32, 32))
        content = pre._build_message("pick up the cube", views, [img, img])[0]["content"]  # noqa: SLF001
        assert content[1]["text"] == "# Ego View\n"
        assert content[4]["text"] == "# Left-Wrist View\n"

    @pytest.mark.parametrize("prefix", ["", "observation.images."])
    def test_alias_keys_are_rejected(self, prefix: str) -> None:
        pre = XR0Preprocessor(
            image_key_view_map={
                f"{prefix}top_camera": "ego",
                f"{prefix}pov_black_follower_camera": "wrist_left",
            },
        )
        with pytest.raises(ValueError, match="must match the batch image keys exactly"):
            pre._extract_view_images(self._batch())  # noqa: SLF001

    def test_empty_map_keeps_dataset_names(self) -> None:
        pre = XR0Preprocessor()
        views, _ = pre._extract_view_images(self._batch())  # noqa: SLF001
        assert views == ["pov_black_follower_camera", "top_camera"]

    def test_key_mismatch_raises(self) -> None:
        pre = XR0Preprocessor(image_key_view_map={"images.top_camera": "ego"})
        with pytest.raises(ValueError, match="must match the batch image keys exactly"):
            pre._extract_view_images(self._batch())  # noqa: SLF001

    def test_unknown_view_name_raises(self) -> None:
        with pytest.raises(ValueError, match="canonical XR0 view names"):
            XR0Preprocessor(image_key_view_map={"images.top_camera": "front_cam"})

    def test_duplicate_view_name_raises(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            XR0Preprocessor(image_key_view_map={"images.top_camera": "ego", "images.pov_black_follower_camera": "ego"})

    def test_factory_threads_the_map(self) -> None:
        pre, _ = make_xr0_preprocessors(
            stats=_stats(),
            chunk_size=HORIZON,
            image_key_view_map={"images.top_camera": "ego"},
        )
        assert pre.image_key_view_map == {"images.top_camera": "ego"}


class TestPrepareAction:
    """Action normalize + pad + validity mask, incl. delta corner cases."""

    @staticmethod
    def _pre(
        max_action_dim: int = 10,
        *,
        chunk_size: int = HORIZON,
        action_mode: str = "absolute",
    ) -> XR0Preprocessor:
        # Identity stats (mean 0 / std 1) so normalization is a near-passthrough.
        return XR0Preprocessor(
            max_action_dim=max_action_dim,
            chunk_size=chunk_size,
            action_mode=action_mode,
            action_mean=torch.zeros(chunk_size, max_action_dim),
            action_std=torch.ones(chunk_size, max_action_dim),
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
        pre = self._pre(max_action_dim=3, chunk_size=2, action_mode="delta")
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
        pre = self._pre(max_action_dim=max_action_dim, chunk_size=2, action_mode="absolute")
        action = torch.arange(2 * 2 * 2).reshape(2, 2, 2).float()
        out, mask = pre._prepare_action(action, torch.device("cpu"))  # noqa: SLF001
        assert torch.allclose(out, reference, atol=1e-5)
        # mask has the padded shape: real dims (2) are valid, padding dims are zero.
        assert mask.shape == reference.shape
        assert mask[..., :2].all()
        assert not mask[..., 2:].any()

    def test_chunk_length_must_match_stats(self) -> None:
        # Per-timestep stats are never broadcast over time.
        pre = self._pre(max_action_dim=3, chunk_size=2)
        with pytest.raises(ValueError, match="2 timesteps"):
            pre._prepare_action(torch.randn(2, 5, 3), torch.device("cpu"))  # noqa: SLF001


class TestPostprocessor:
    """Action denormalize + delta re-add + unpad, against references."""

    def test_init_defaults_to_identity(self) -> None:
        """Without features, use identity stats and no action-width trimming."""
        post = XR0Postprocessor(max_action_dim=3, chunk_size=2)
        assert post.action_mode == "absolute"
        assert post.action_dim is None
        assert torch.equal(post.action_mean, torch.zeros(2, 3))
        assert torch.equal(post.action_std, torch.ones(2, 3))

    @pytest.mark.parametrize(("provide_mean", "provide_std"), [(False, False), (True, False), (False, True), (True, True)])
    def test_init_uses_feature_stats_unless_both_overrides_are_given(self, provide_mean: bool, provide_std: bool) -> None:
        """Infer the action width from features even when explicit stats override them."""
        feature = Feature(
            ftype=FeatureType.ACTION,
            normalization_data=NormalizationParameters(mean=[[2, 4], [6, 8]], std=[[1, 3], [5, 7]]),
        )
        post = XR0Postprocessor(
            max_action_dim=3,
            chunk_size=2,
            action_mode="delta",
            features={"action": feature},
            action_mean=[[10, 20], [30, 40]] if provide_mean else None,
            action_std=[[11, 21], [31, 41]] if provide_std else None,
        )
        both = provide_mean and provide_std
        assert post.action_mode == "delta"
        assert post.action_dim == 2
        assert torch.equal(post.action_mean, torch.tensor([[10, 20, 0], [30, 40, 0]] if both else [[2, 4, 0], [6, 8, 0]]))
        assert torch.equal(post.action_std, torch.tensor([[11, 21, 1], [31, 41, 1]] if both else [[1, 3, 1], [5, 7, 1]]))

    @staticmethod
    def _post(
        max_action_dim: int = 3,
        *,
        chunk_size: int = 2,
        action_mode: str = "absolute",
        action_dim: int | None = None,
    ) -> XR0Postprocessor:
        # Identity stats (mean 0 / std 1) so denormalization is a near-passthrough.
        post = XR0Postprocessor(
            max_action_dim=max_action_dim,
            chunk_size=chunk_size,
            action_mode=action_mode,
            action_mean=torch.zeros(chunk_size, max_action_dim),
            action_std=torch.ones(chunk_size, max_action_dim),
        )
        post.action_dim = action_dim  # unpadded slice width (None -> no slice)
        return post

    def test_missing_action_is_passthrough(self) -> None:
        post = self._post()
        batch = {STATE: torch.zeros(2, 3)}
        assert post(batch) == batch

    def test_chunk_length_must_match_stats(self) -> None:
        """Reject predictions whose horizon differs from the normalization buffers."""
        post = self._post(chunk_size=2)
        with pytest.raises(ValueError, match="expected an action chunk of 2 timesteps.*got 3"):
            post({ACTION: torch.zeros(1, 3, 3)})

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


def test_action_stats_from_feature_or_identity() -> None:
    """Use action feature statistics, falling back to identity when missing."""
    pre = XR0Preprocessor(chunk_size=2, max_action_dim=3)
    feature = Feature(
        ftype=FeatureType.ACTION,
        normalization_data=NormalizationParameters(mean=[[2, 4], [6, 8]], std=[[1, 3], [5, 7]]),
    )
    mean, std = pre._action_stats({"action": feature})  # noqa: SLF001
    assert torch.equal(mean, torch.tensor([[2, 4], [6, 8]]))
    assert torch.equal(std, torch.tensor([[1, 3], [5, 7]]))
    mean, std = pre._action_stats(None)  # noqa: SLF001
    assert torch.equal(mean, torch.zeros(2, 3))
    assert torch.equal(std, torch.ones(2, 3))


def test_state_stats_only_when_enabled() -> None:
    """Use state feature statistics only with state normalization enabled."""
    feature = Feature(
        ftype=FeatureType.STATE,
        normalization_data=NormalizationParameters(mean=[2, 4], std=[1, 3]),
    )
    pre = XR0Preprocessor(max_state_dim=3, normalize_state=True)
    mean, std = pre._state_stats({"state": feature})  # noqa: SLF001
    assert torch.equal(mean, torch.tensor([2, 4]))
    assert torch.equal(std, torch.tensor([1, 3]))
    mean, std = pre._state_stats(None)  # noqa: SLF001
    assert torch.equal(mean, torch.zeros(1, 3))
    assert torch.equal(std, torch.ones(1, 3))
    raw = XR0Preprocessor(max_state_dim=3)
    mean, std = raw._state_stats({"state": feature})  # noqa: SLF001
    assert torch.equal(mean, torch.zeros(1, 3))
    assert torch.equal(std, torch.ones(1, 3))


@pytest.mark.parametrize(("provide_mean", "provide_std"), [(False, False), (True, False), (False, True), (True, True)])
def test_action_stats_override_requires_both(provide_mean: bool, provide_std: bool) -> None:
    """Use explicit action stats only when both mean and std are supplied."""
    feature = Feature(
        ftype=FeatureType.ACTION,
        normalization_data=NormalizationParameters(mean=[[2, 4], [6, 8]], std=[[1, 3], [5, 7]]),
    )
    pre = XR0Preprocessor(
        max_action_dim=3,
        chunk_size=2,
        features={"action": feature},
        action_mean=[[10, 20], [30, 40]] if provide_mean else None,
        action_std=[[11, 21], [31, 41]] if provide_std else None,
    )
    both = provide_mean and provide_std
    assert torch.equal(pre.action_mean, torch.tensor([[10, 20, 0], [30, 40, 0]] if both else [[2, 4, 0], [6, 8, 0]]))
    assert torch.equal(pre.action_std, torch.tensor([[11, 21, 1], [31, 41, 1]] if both else [[1, 3, 1], [5, 7, 1]]))


@pytest.mark.parametrize(("provide_mean", "provide_std"), [(False, False), (True, False), (False, True), (True, True)])
def test_state_stats_override_requires_both(provide_mean: bool, provide_std: bool) -> None:
    """Use explicit state stats only when both mean and std are supplied."""
    feature = Feature(ftype=FeatureType.STATE, normalization_data=NormalizationParameters(mean=[2, 4], std=[1, 3]))
    pre = XR0Preprocessor(
        max_state_dim=3,
        normalize_state=True,
        features={"state": feature},
        state_mean=[10, 20] if provide_mean else None,
        state_std=[11, 21] if provide_std else None,
    )
    both = provide_mean and provide_std
    assert torch.equal(pre.state_mean, torch.tensor([[10, 20, 0]] if both else [[2, 4, 0]]))
    assert torch.equal(pre.state_std, torch.tensor([[11, 21, 1]] if both else [[1, 3, 1]]))


class TestMakeXr0Preprocessors:
    """Factory wiring: stats -> features -> pre/post action stats."""

    def test_state_stats_feature_default_and_explicit(self) -> None:
        stats = _stats()
        stats["observation.state"]["mean"] = [2.0] * STATE_DIM
        stats["observation.state"]["std"] = [3.0] * STATE_DIM
        pre, _ = make_xr0_preprocessors(
            stats=stats, chunk_size=HORIZON, max_state_dim=10, normalize_state=True
        )
        assert torch.equal(pre.state_mean[0, :STATE_DIM], torch.full((STATE_DIM,), 2.0))
        assert torch.equal(pre.state_std[0, :STATE_DIM], torch.full((STATE_DIM,), 3.0))
        assert torch.equal(pre.state_mean[0, STATE_DIM:], torch.zeros(2))
        assert torch.equal(pre.state_std[0, STATE_DIM:], torch.ones(2))

        raw, _ = make_xr0_preprocessors(stats=stats, chunk_size=HORIZON, max_state_dim=10)
        assert torch.equal(raw.state_mean, torch.zeros(1, 10))
        assert torch.equal(raw.state_std, torch.ones(1, 10))

        explicit = XR0Preprocessor(
            max_state_dim=10, state_mean=[4.0] * STATE_DIM, state_std=[5.0] * STATE_DIM
        )
        assert torch.equal(explicit.state_mean[0, :STATE_DIM], torch.full((STATE_DIM,), 4.0))
        assert torch.equal(explicit.state_std[0, :STATE_DIM], torch.full((STATE_DIM,), 5.0))

    def test_absolute_derives_stats_from_features(self) -> None:
        pre, post = make_xr0_preprocessors(max_action_dim=32, stats=_stats(), chunk_size=HORIZON)
        assert isinstance(pre, XR0Preprocessor)
        assert isinstance(post, XR0Postprocessor)
        assert pre.action_mode == "absolute"
        # feature action mean/std (0.1 / 2.0) fill the real dims, padding stays 0/1.
        assert pre.action_mean.shape == (HORIZON, 32)
        assert torch.allclose(pre.action_mean[:, :ACTION_DIM], torch.full((HORIZON, ACTION_DIM), 0.1))
        assert torch.allclose(pre.action_mean[:, ACTION_DIM:], torch.zeros(HORIZON, 32 - ACTION_DIM))
        assert torch.allclose(pre.action_std[:, :ACTION_DIM], torch.full((HORIZON, ACTION_DIM), 2.0))
        assert torch.allclose(pre.action_std[:, ACTION_DIM:], torch.ones(HORIZON, 32 - ACTION_DIM))
        # postprocessor recovers the unpadded action_dim for the final slice.
        assert post.action_dim == ACTION_DIM
        assert torch.allclose(post.action_mean, pre.action_mean)
        assert torch.allclose(post.action_std, pre.action_std)

    def test_per_dim_action_stats_are_rejected(self) -> None:
        # A per-dimension action vector would have to be broadcast over time.
        stats = _stats()
        stats["action"]["mean"] = [0.1] * ACTION_DIM
        stats["action"]["std"] = [2.0] * ACTION_DIM
        with pytest.raises(ValueError, match="per-timestep"):
            make_xr0_preprocessors(max_action_dim=32, stats=stats, chunk_size=HORIZON)

    def test_per_dim_dataset_stats_with_chunk_override(self) -> None:
        stats = _stats()
        stats["action"]["mean"] = [0.1] * ACTION_DIM
        stats["action"]["std"] = [2.0] * ACTION_DIM
        mean = torch.full((HORIZON, 32), 0.5)
        std = torch.full((HORIZON, 32), 3.0)

        pre, post = make_xr0_preprocessors(
            stats=stats,
            chunk_size=HORIZON,
            action_mean=mean,
            action_std=std,
        )

        assert post.action_dim == ACTION_DIM
        assert torch.equal(pre.action_mean, mean)
        assert torch.equal(post.action_mean, mean)
        assert torch.equal(post.action_std, std)

    @pytest.mark.parametrize("action_mode", ["absolute", "delta"])
    def test_override_stats_applied_to_pair(self, action_mode: str) -> None:
        override_mean = torch.full((HORIZON, 32), 0.5)
        override_std = torch.full((HORIZON, 32), 3.0)
        pre, post = make_xr0_preprocessors(
            stats=_stats(),
            chunk_size=HORIZON,
            action_mode=action_mode,
            action_mean=override_mean,
            action_std=override_std,
        )
        assert pre.action_mode == action_mode
        assert post.action_mode == action_mode
        # explicit per-timestep stats override the ones derived from the features.
        assert torch.allclose(pre.action_mean, override_mean)
        assert torch.allclose(pre.action_std, override_std)
        assert torch.allclose(post.action_mean, override_mean)
        assert torch.allclose(post.action_std, override_std)
        # the unpadded action_dim is still recovered from features.
        assert post.action_dim == ACTION_DIM

    def test_round_trip_with_per_timestep_stats(self) -> None:
        # Each chunk position gets its own scale; pre -> post must be the identity.
        mean = torch.arange(HORIZON * 32, dtype=torch.float32).reshape(HORIZON, 32)
        std = torch.arange(1, HORIZON * 32 + 1, dtype=torch.float32).reshape(HORIZON, 32)
        pre, post = make_xr0_preprocessors(
            stats=_stats(),
            chunk_size=HORIZON,
            action_mean=mean,
            action_std=std,
        )
        action = torch.randn(2, HORIZON, ACTION_DIM)
        normalized, _ = pre._prepare_action(action, torch.device("cpu"))  # noqa: SLF001
        assert torch.allclose(post({ACTION: normalized})[ACTION], action, atol=1e-4)

    def test_no_stats_is_identity(self) -> None:
        pre, post = make_xr0_preprocessors(max_action_dim=32, stats=None, chunk_size=HORIZON)
        assert torch.allclose(pre.action_mean, torch.zeros(HORIZON, 32))
        assert torch.allclose(pre.action_std, torch.ones(HORIZON, 32))
        assert post.action_dim is None
