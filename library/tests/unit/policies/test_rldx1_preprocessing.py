# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline parity tests for the PAS-native RLDX-1 preprocessing blocks.

Stage 0 establishes the vendored :class:`StateActionProcessor` (pure numpy, no
network) as the golden reference. Stage 1 asserts that the native normalization
(``FeatureNormalizeTransform`` + :func:`clip_state_action`) reproduces that
reference for the v1 single-group state/action case, under both min/max and
q01/q99 bounds. Stage 2 asserts that :func:`pad_state_action` matches the
vendored padding / mask block.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from physicalai.data.observation import ACTION, STATE
from physicalai.policies.rldx1.preprocessing import (
    ACTION_MASK,
    build_qwen_conversation,
    build_state_action_degenerate_masks,
    build_state_action_features,
    build_state_action_norm_map,
    clip_state_action,
    formalize_language,
    pad_state_action,
    tokenize_vlm_batch,
    zero_degenerate_state_action,
)
from physicalai.policies.utils.normalization import FeatureNormalizeTransform
from tests.unit.policies.rldx1_vendored.data_types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from tests.unit.policies.rldx1_vendored.state_action_processor import StateActionProcessor
from typing import Literal

MAX_STATE_DIM = 64
MAX_ACTION_DIM = 64

STATE_DIM = 7
ACTION_DIM = 7
ACTION_HORIZON = 16


def _native_normalize(
    batch: dict[str, torch.Tensor],
    stats: dict[str, dict[str, list[float]]],
    *,
    use_percentiles: bool,
) -> dict[str, torch.Tensor]:
    """Apply the PAS-native normalization (transform + clip + degenerate zeroing).

    Mirrors the preprocessor pipeline: constant channels (``q99 == q01``) are
    zeroed to match the vendored ``normalize_values_minmax`` mask path.
    """
    features = build_state_action_features(stats)
    norm_map = build_state_action_norm_map(use_percentiles=use_percentiles)
    normalizer = FeatureNormalizeTransform(features, norm_map)
    normalizer.eval()
    masks = build_state_action_degenerate_masks(features, use_percentiles=use_percentiles)
    with torch.no_grad():
        return zero_degenerate_state_action(clip_state_action(normalizer(batch)), masks)


def _stat(dim: int, *, seed: int) -> dict[str, list[float]]:
    """Build a non-degenerate stat dict (min < max, q01 < q99) for ``dim``."""
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-3.0, -1.0, size=dim)
    hi = rng.uniform(1.0, 3.0, size=dim)
    q01 = lo + 0.1
    q99 = hi - 0.1
    return {
        "min": lo.tolist(),
        "max": hi.tolist(),
        "mean": ((lo + hi) / 2.0).tolist(),
        "std": np.ones(dim).tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }


def _vendored_normalizer(
    state_stat: dict[str, list[float]],
    action_stat: dict[str, list[float]],
    *,
    use_percentiles: bool,
) -> StateActionProcessor:
    """Build the vendored single-group StateActionProcessor (golden oracle)."""
    modality_configs = {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["state"]),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)),
            modality_keys=["action"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="state",
                ),
            ],
        ),
    }
    statistics = {
        "state": {"state": state_stat},
        "action": {"action": action_stat},
    }
    proc = StateActionProcessor(
        modality_configs=modality_configs,
        statistics=statistics,
        use_percentiles=use_percentiles,
        clip_outliers=True,
        use_relative_action=False,
    )
    proc.eval()
    return proc


@pytest.mark.parametrize("use_percentiles", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_native_normalizer_matches_vendored(use_percentiles: bool, seed: int) -> None:
    """Native normalizer matches the vendored StateActionProcessor elementwise."""
    rng = np.random.default_rng(seed)
    state_stat = _stat(STATE_DIM, seed=seed)
    action_stat = _stat(ACTION_DIM, seed=seed + 100)

    # Raw inputs span beyond the bounds so clipping is exercised.
    raw_state = rng.uniform(-4.0, 4.0, size=(1, STATE_DIM)).astype(np.float32)
    raw_action = rng.uniform(-4.0, 4.0, size=(ACTION_HORIZON, ACTION_DIM)).astype(np.float32)

    # -- golden: vendored numpy pipeline --
    vendored = _vendored_normalizer(state_stat, action_stat, use_percentiles=use_percentiles)
    gold_state = vendored.apply_state({"state": raw_state})["state"]
    gold_action = vendored.apply_action(
        {"action": raw_action},
        state={"state": raw_state},
    )["action"]

    # -- native: FeatureNormalizeTransform + clip --
    stats = {"observation.state": state_stat, "action": action_stat}
    batch = {
        STATE: torch.from_numpy(raw_state),
        ACTION: torch.from_numpy(raw_action),
    }
    out = _native_normalize(batch, stats, use_percentiles=use_percentiles)

    np.testing.assert_allclose(out[STATE].numpy(), gold_state, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(out[ACTION].numpy(), gold_action, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("use_percentiles", [False, True])
def test_native_normalizer_zeros_degenerate_channels(use_percentiles: bool) -> None:
    """Degenerate (constant) channels normalize to 0, matching the vendored path.

    Without the zeroing step PAS maps a constant channel to -1 (``2*0/1e-8 - 1``,
    clipped); the vendored ``normalize_values_minmax`` mask leaves it at 0. This
    caused the RoboCasa ``base_rotation`` state drift.
    """
    state_stat = _stat(STATE_DIM, seed=3)
    # Force channels 2 and 4 to be degenerate (min==max and q01==q99).
    for idx in (2, 4):
        for lo_key, hi_key in (("min", "max"), ("q01", "q99")):
            state_stat[lo_key][idx] = 0.0
            state_stat[hi_key][idx] = 0.0
    action_stat = _stat(ACTION_DIM, seed=103)

    raw_state = np.zeros((1, STATE_DIM), dtype=np.float32)  # constant channels sit at 0
    raw_action = np.random.default_rng(3).uniform(-4.0, 4.0, size=(ACTION_HORIZON, ACTION_DIM)).astype(np.float32)

    vendored = _vendored_normalizer(state_stat, action_stat, use_percentiles=use_percentiles)
    gold_state = vendored.apply_state({"state": raw_state})["state"]

    stats = {"observation.state": state_stat, "action": action_stat}
    batch = {STATE: torch.from_numpy(raw_state), ACTION: torch.from_numpy(raw_action)}
    out = _native_normalize(batch, stats, use_percentiles=use_percentiles)

    assert out[STATE][0, 2].item() == 0.0
    assert out[STATE][0, 4].item() == 0.0
    np.testing.assert_allclose(out[STATE].numpy(), gold_state, atol=1e-5, rtol=0.0)


def test_build_features_present_keys() -> None:
    """Feature builder maps state/action stats to STATE/ACTION features."""
    stats = {
        "observation.state": _stat(STATE_DIM, seed=0),
        "action": _stat(ACTION_DIM, seed=1),
    }
    features = build_state_action_features(stats)
    assert set(features) == {STATE, ACTION}
    assert features[STATE].shape == (STATE_DIM,)
    assert features[ACTION].shape == (ACTION_DIM,)


def test_normalize_transform_identity_without_features() -> None:
    """With no stats the normalize transform is a passthrough.

    The ``clip_state_action`` step is applied separately by the preprocessor and
    only after real normalization, so the no-feature path leaves values
    untouched.
    """
    features = build_state_action_features({})
    norm_map = build_state_action_norm_map(use_percentiles=True)
    transform = FeatureNormalizeTransform(features, norm_map)
    state = torch.randn(1, STATE_DIM)
    out = transform({STATE: state.clone()})
    torch.testing.assert_close(out[STATE], state)


def _vendored_pad_state(state_np: np.ndarray, max_state_dim: int) -> torch.Tensor:
    """Vendored state padding, copied verbatim from RLDXProcessor.__call__.

    Source: ``components/processing/processing_rldx.py`` (state concat + pad).
    """
    normalized_states = torch.from_numpy(state_np.astype(np.float32))  # (1, d)
    normalized_states = torch.cat(
        [
            normalized_states,
            torch.zeros(normalized_states.shape[0], max_state_dim - normalized_states.shape[1]),
        ],
        dim=-1,
    )
    return normalized_states  # (1, max_state_dim)


def _vendored_pad_action(
    action_np: np.ndarray,
    max_action_dim: int,
    max_action_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vendored action padding + mask, copied verbatim from RLDXProcessor.__call__.

    Source: ``components/processing/processing_rldx.py`` (action pad + mask).
    """
    normalized_actions = torch.from_numpy(action_np.astype(np.float32))  # (t, d)
    action_dim = normalized_actions.shape[1]
    normalized_actions = torch.cat(
        [
            normalized_actions,
            torch.zeros(normalized_actions.shape[0], max_action_dim - normalized_actions.shape[1]),
        ],
        dim=-1,
    )
    action_horizon = normalized_actions.shape[0]
    normalized_actions = torch.cat(
        [
            normalized_actions,
            torch.zeros(max_action_horizon - normalized_actions.shape[0], max_action_dim),
        ],
        dim=0,
    )
    action_mask = torch.ones_like(normalized_actions)
    action_mask[action_horizon:] = 0
    action_mask[:, action_dim:] = 0
    return normalized_actions, action_mask


@pytest.mark.parametrize("use_percentiles", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_padder_matches_vendored(use_percentiles: bool, batch_size: int) -> None:
    """Native padder + mask match the vendored RLDXProcessor padding block."""
    rng = np.random.default_rng(batch_size)
    state_stat = _stat(STATE_DIM, seed=0)
    action_stat = _stat(ACTION_DIM, seed=100)
    vendored = _vendored_normalizer(state_stat, action_stat, use_percentiles=use_percentiles)

    gold_states = []
    gold_actions = []
    gold_masks = []
    raw_states = []
    raw_actions = []
    for _ in range(batch_size):
        raw_state = rng.uniform(-4.0, 4.0, size=(1, STATE_DIM)).astype(np.float32)
        raw_action = rng.uniform(-4.0, 4.0, size=(ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
        raw_states.append(raw_state)
        raw_actions.append(raw_action)

        norm_state = vendored.apply_state({"state": raw_state})["state"]
        norm_action = vendored.apply_action({"action": raw_action}, state={"state": raw_state})["action"]

        gold_states.append(_vendored_pad_state(norm_state, MAX_STATE_DIM))
        action, mask = _vendored_pad_action(norm_action, MAX_ACTION_DIM, ACTION_HORIZON)
        gold_actions.append(action)
        gold_masks.append(mask)

    gold_state = torch.stack(gold_states, dim=0)  # (B, 1, max_state_dim)
    gold_action = torch.stack(gold_actions, dim=0)  # (B, max_h, max_action_dim)
    gold_mask = torch.stack(gold_masks, dim=0)

    # -- native: normalize then pad --
    stats = {"observation.state": state_stat, "action": action_stat}
    batch = {
        STATE: torch.from_numpy(np.concatenate(raw_states, axis=0)),  # (B, state_dim)
        ACTION: torch.from_numpy(np.stack(raw_actions, axis=0)),  # (B, T, action_dim)
    }
    out = _native_normalize(batch, stats, use_percentiles=use_percentiles)
    out = pad_state_action(
        out,
        max_state_dim=MAX_STATE_DIM,
        max_action_dim=MAX_ACTION_DIM,
        max_action_horizon=ACTION_HORIZON,
    )

    torch.testing.assert_close(out[STATE], gold_state, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(out[ACTION], gold_action, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(out[ACTION_MASK], gold_mask, atol=0.0, rtol=0.0)


def test_padder_inference_no_action() -> None:
    """Without an action the padder only pads state, no mask is added."""
    out = pad_state_action(
        {STATE: torch.randn(2, STATE_DIM)},
        max_state_dim=MAX_STATE_DIM,
        max_action_dim=MAX_ACTION_DIM,
        max_action_horizon=ACTION_HORIZON,
    )
    assert out[STATE].shape == (2, 1, MAX_STATE_DIM)
    assert ACTION not in out
    assert ACTION_MASK not in out


# ---------------------------------------------------------------------------- #
# Stage 3: image geometry                                                      #
# ---------------------------------------------------------------------------- #

_IMAGE_MAX_AREA = 65536
_IMAGE_RESIZE_M = 32

# (h, w) covering: square no-op, non-square downscale, odd sizes, large frames.
_IMAGE_SHAPES = [
    (256, 256),
    (224, 224),
    (480, 640),
    (240, 320),
    (200, 200),
    (333, 211),
    (720, 1280),
]


@pytest.mark.parametrize(("height", "width"), _IMAGE_SHAPES)
def test_aspect_area_resize_crop_shape(height: int, width: int) -> None:
    """Torchvision geometry produces an m-aligned, area-budgeted crop."""
    from physicalai.policies.rldx1.augmentations import AspectAreaResizeAndCrop

    transform = AspectAreaResizeAndCrop(target_area=_IMAGE_MAX_AREA, m_alignment=_IMAGE_RESIZE_M)
    rng = np.random.default_rng(height * 1000 + width)
    image = torch.from_numpy(rng.integers(0, 256, size=(3, height, width), dtype=np.uint8))

    out = transform(image)

    assert out.dtype == torch.uint8
    assert out.shape[0] == 3
    assert out.shape[1] % _IMAGE_RESIZE_M == 0
    assert out.shape[2] % _IMAGE_RESIZE_M == 0


def test_aspect_area_resize_crop_min_area_floor() -> None:
    """A frame below ``min_area`` is upscaled toward the floor, not ``target_area``."""
    from physicalai.policies.rldx1.augmentations import AspectAreaResizeAndCrop

    transform = AspectAreaResizeAndCrop(target_area=65536, m_alignment=32, min_area=50176)
    image = torch.zeros(3, 96, 96, dtype=torch.uint8)

    out = transform(image)

    assert out.shape[1:] == (224, 224)


# ---------------------------------------------------------------------------- #
# Stage 4: VLM tokenization                                                    #
# ---------------------------------------------------------------------------- #

_VLM_MODEL_NAME = "RLWRLD/RLDX-1-VLM"
_VLM_MODEL_TYPE: Literal["qwen3_vl", "vtc_qwen3_vl"] = "vtc_qwen3_vl"
_VLM_LOADING_KWARGS = {"trust_remote_code": False, "use_fast": True}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Pick up the RED block!", "pick up the red block"),
        ("Move left/right, then STOP.", "move leftright then stop"),
        ("  Keep   spacing  ", "  keep   spacing  "),
    ],
)
def test_formalize_language(raw: str, expected: str) -> None:
    """formalize_language lowercases and strips punctuation, keeps whitespace."""
    assert formalize_language(raw) == expected


def test_build_qwen_conversation_text_first() -> None:
    """Default (text-first) conversation places text before image blocks."""
    conv = build_qwen_conversation(["img0", "img1"], "pick the block")
    assert len(conv) == 1
    content = conv[0]["content"]
    assert content[0] == {"type": "text", "text": "pick the block"}
    assert content[1:] == [
        {"type": "image", "image": "img0"},
        {"type": "image", "image": "img1"},
    ]


def test_build_qwen_conversation_image_first() -> None:
    """image_first places image blocks before the text block."""
    conv = build_qwen_conversation(["img0"], "go", image_first=True)
    content = conv[0]["content"]
    assert content[0] == {"type": "image", "image": "img0"}
    assert content[-1] == {"type": "text", "text": "go"}


@pytest.fixture(scope="module")
def vlm_processor():  # noqa: ANN201
    """Load the cached Qwen processor, skipping when unavailable offline."""
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(_VLM_MODEL_NAME, **_VLM_LOADING_KWARGS)
    except Exception as exc:  # noqa: BLE001 - any load failure -> skip
        pytest.skip(f"Qwen processor unavailable offline: {exc}")
    processor.tokenizer.padding_side = "left"
    return processor


def test_tokenize_vlm_batch_matches_vendored(vlm_processor) -> None:  # noqa: ANN001
    """Native VLM tokenization matches the vendored RLDXDataCollator output."""
    from PIL import Image

    from tests.unit.policies.rldx1_vendored.processing_rldx import RLDXDataCollator

    processor = vlm_processor
    rng = np.random.default_rng(0)

    samples = []
    conversations = []
    for b in range(2):
        images = [
            Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
            for _ in range(2)
        ]
        language = formalize_language("Pick the red block." if b == 0 else "Push the blue cube.")
        conv = build_qwen_conversation(images, language)
        chat_text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        samples.append(
            {
                "text": chat_text,
                "images": images,
                "conversation": conv,
                "image_wise_encoding": True,
                "num_views": 2,
                "num_frames": 1,
            },
        )
        conversations.append(conv)

    collator = RLDXDataCollator(
        model_name=_VLM_MODEL_NAME,
        model_type=_VLM_MODEL_TYPE,
        transformers_loading_kwargs=_VLM_LOADING_KWARGS,
    )
    gold = collator._collate_vlm_content(samples)  # noqa: SLF001 - parity oracle
    native = tokenize_vlm_batch(processor, conversations)

    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        torch.testing.assert_close(native[key], gold[key], atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------- #
# Wiring: native Rldx1Preprocessor.forward vs vendored                         #
# ---------------------------------------------------------------------------- #

OBSERVATION_STATE = "observation.state"


def _make_obs_batch(*, batch_size: int, with_action: bool, seed: int) -> dict[str, object]:
    """Build a flat observation batch (2 camera views) for the preprocessor."""
    rng = np.random.default_rng(seed)
    batch: dict[str, object] = {
        OBSERVATION_STATE: torch.from_numpy(
            rng.uniform(-4.0, 4.0, size=(batch_size, STATE_DIM)).astype(np.float32),
        ),
        "task": [f"Pick object {i}." for i in range(batch_size)],
    }
    for view in range(2):
        batch[f"observation.images.cam{view}"] = torch.from_numpy(
            rng.integers(0, 256, size=(batch_size, 3, 64, 64), dtype=np.uint8),
        )
    if with_action:
        batch[ACTION] = torch.from_numpy(
            rng.uniform(-4.0, 4.0, size=(batch_size, ACTION_HORIZON, ACTION_DIM)).astype(np.float32),
        )
    return batch


@pytest.fixture(scope="module")
def rldx1_preprocessor():  # noqa: ANN201
    """Build a preprocessor and trigger the VLM load, skipping if uncached."""
    from physicalai.policies.rldx1.preprocessor import Rldx1Preprocessor

    stats = {
        OBSERVATION_STATE: _stat(STATE_DIM, seed=0),
        "action": _stat(ACTION_DIM, seed=1),
    }
    pre = Rldx1Preprocessor(
        stats=stats,
        max_state_dim=MAX_STATE_DIM,
        max_action_dim=MAX_ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        use_percentiles=True,
    )
    pre.eval()
    try:
        _ = pre._vlm_processor  # noqa: SLF001 - force the native HF processor load
    except Exception as exc:  # noqa: BLE001 - any load failure -> skip
        pytest.skip(f"Qwen processor unavailable offline: {exc}")
    return pre


@pytest.mark.parametrize("with_action", [True, False])
def test_native_forward_contract(rldx1_preprocessor, with_action: bool) -> None:  # noqa: ANN001
    """Native forward emits the documented RLDX input keys and shapes.

    Each sub-component (normalizer, padder, image geometry, VLM tokenization) is
    parity-tested against the vendored reference in the stage tests above; this
    asserts the end-to-end native ``forward`` wires them into the contract that
    :meth:`RLDX.forward` / :meth:`RLDX.get_action` consume.
    """
    pre = rldx1_preprocessor
    batch_size = 2
    batch = _make_obs_batch(batch_size=batch_size, with_action=with_action, seed=7)

    out = pre.forward(batch)

    required = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "image_wise_encoding",
        "num_views",
        "num_frames",
        STATE,
        "embodiment_id",
    }
    assert required <= set(out)

    assert out[STATE].shape == (batch_size, 1, MAX_STATE_DIM)
    assert out[STATE].dtype == torch.float32
    for key in ("image_wise_encoding", "num_views", "num_frames", "embodiment_id"):
        assert out[key].shape == (batch_size,), key
    assert out["input_ids"].shape[0] == batch_size
    assert out["attention_mask"].shape == out["input_ids"].shape

    if with_action:
        assert out[ACTION].shape == (batch_size, ACTION_HORIZON, MAX_ACTION_DIM)
        assert out[ACTION_MASK].shape == out[ACTION].shape
    else:
        assert ACTION not in out
        assert ACTION_MASK not in out


@pytest.mark.parametrize("use_percentiles", [True, False])
def test_denormalize_action_matches_vendored(use_percentiles: bool) -> None:
    """Native action denormalization matches the vendored ``StateActionProcessor``."""
    from physicalai.policies.rldx1.preprocessor import make_rldx1_transforms

    state_stat = _stat(STATE_DIM, seed=4)
    action_stat = _stat(ACTION_DIM, seed=5)
    stats = {
        OBSERVATION_STATE: state_stat,
        "action": action_stat,
    }
    _, post = make_rldx1_transforms(
        stats=stats,
        max_state_dim=MAX_STATE_DIM,
        max_action_dim=MAX_ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        use_percentiles=use_percentiles,
    )

    rng = np.random.default_rng(7)
    # Predicted, padded action chunk in the normalized [-1, 1] space (with a few
    # out-of-range values to exercise the clip parity).
    pred = rng.uniform(-1.3, 1.3, size=(2, ACTION_HORIZON, MAX_ACTION_DIM)).astype(np.float32)

    native = post(torch.from_numpy(pred))

    # Vendored oracle: slice the padded prediction to the unpadded action width
    # (mirrors RLDXProcessor.decode_action), then invert via the numpy processor.
    vendored = _vendored_normalizer(state_stat, action_stat, use_percentiles=use_percentiles)
    sliced = pred[..., :ACTION_HORIZON, :ACTION_DIM]
    gold = vendored.unapply_action({"action": sliced})["action"]

    assert native.shape == (2, ACTION_HORIZON, ACTION_DIM)
    torch.testing.assert_close(native, torch.from_numpy(np.asarray(gold)).to(native.dtype), atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------- #
# VTC multi-frame: delta timestamps + frame stacking parity                    #
# ---------------------------------------------------------------------------- #

_VIDEO_LENGTH = 4
_VIDEO_STRIDE = 2
_NUM_VIEWS = 2


def test_rldx1_observation_delta_indices_video_window() -> None:
    """``observation_delta_indices`` is the VTC window [-6, -4, -2, 0] for L=4, S=2."""
    from physicalai.policies.rldx1.model import compute_action_delta_indices, compute_video_window_offsets

    assert compute_video_window_offsets(_VIDEO_LENGTH, _VIDEO_STRIDE) == [-6, -4, -2, 0]
    assert compute_action_delta_indices(ACTION_HORIZON) == list(range(ACTION_HORIZON))


@pytest.mark.parametrize(
    ("video_length", "video_stride", "expected"),
    [
        (4, 2, [-6, -4, -2, 0]),
        (4, 1, [-3, -2, -1, 0]),
        (1, 2, [0]),
        (3, 3, [-6, -3, 0]),
    ],
)
def test_rldx1_observation_delta_indices_parametrized(
    video_length: int,
    video_stride: int,
    expected: list[int],
) -> None:
    """The video-window offsets scale with ``video_length`` / ``video_stride``."""
    from physicalai.policies.rldx1.model import compute_video_window_offsets

    assert compute_video_window_offsets(video_length, video_stride) == expected


@pytest.mark.parametrize(
    ("shape", "expected_frames"),
    [
        ((3, 8, 8), 1),  # (C, H, W)
        ((8, 8, 3), 1),  # (H, W, C)
        ((_VIDEO_LENGTH, 3, 8, 8), _VIDEO_LENGTH),  # (T, C, H, W)
        ((_VIDEO_LENGTH, 8, 8, 3), _VIDEO_LENGTH),  # (T, H, W, C)
    ],
)
def test_split_frames_single_and_multi(shape: tuple[int, ...], expected_frames: int) -> None:
    """``_split_frames`` returns one array for 3-D and T arrays for 4-D inputs."""
    from physicalai.policies.rldx1.preprocessor import Rldx1Preprocessor

    arr = np.zeros(shape, dtype=np.uint8)
    frames = Rldx1Preprocessor._split_frames(arr)  # noqa: SLF001 - unit under test
    assert len(frames) == expected_frames


def _make_multiframe_obs(*, num_frames: int, num_views: int, seed: int) -> tuple[dict[str, object], dict[str, list]]:
    """Build matched native (CHW) and oracle (HWC) multi-frame image batches.

    Returns:
        ``(native_batch, oracle_images)`` -- the flat Observation dict (images
        ``(1, T, C, H, W)``) and the upstream ``images`` dict (view -> list of
        ``T`` HWC uint8 frames), carrying pixel-identical content.
    """
    rng = np.random.default_rng(seed)
    native_batch: dict[str, object] = {
        OBSERVATION_STATE: torch.zeros(1, STATE_DIM, dtype=torch.float32),
        "task": ["Pick the RED block!"],
    }
    oracle_images: dict[str, list] = {}
    for view in range(num_views):
        # (T, H, W, C) uint8 raw content, shared by both pipelines.
        raw = rng.integers(0, 256, size=(num_frames, 64, 64, 3), dtype=np.uint8)
        oracle_images[f"cam{view}"] = [raw[t] for t in range(num_frames)]
        # Native consumes (B=1, T, C, H, W).
        chw = np.transpose(raw, (0, 3, 1, 2))[None]
        native_batch[f"observation.images.cam{view}"] = torch.from_numpy(chw)
    return native_batch, oracle_images


def _vendored_vtc_processor(model_name: str):  # noqa: ANN202
    """Construct the vendored RLDXProcessor (VTC) for the VLM-path oracle."""
    from tests.unit.policies.rldx1_vendored.data_types import (
        ActionConfig,
        ActionFormat,
        ActionRepresentation,
        ActionType,
        ModalityConfig,
    )
    from tests.unit.policies.rldx1_vendored.processing_rldx import RLDXProcessor

    emb = "new_embodiment"
    views = [f"cam{i}" for i in range(_NUM_VIEWS)]
    modality_configs = {
        emb: {
            "state": ModalityConfig(delta_indices=[0], modality_keys=["state"]),
            "action": ModalityConfig(
                delta_indices=list(range(ACTION_HORIZON)),
                modality_keys=["action"],
                action_configs=[
                    ActionConfig(
                        rep=ActionRepresentation.ABSOLUTE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                        state_key="state",
                    ),
                ],
            ),
            "video": ModalityConfig(
                delta_indices=[(i - (_VIDEO_LENGTH - 1)) * _VIDEO_STRIDE for i in range(_VIDEO_LENGTH)],
                modality_keys=views,
            ),
        },
    }
    statistics = {
        emb: {
            "state": {"state": _stat(STATE_DIM, seed=0)},
            "action": {"action": _stat(ACTION_DIM, seed=1)},
        },
    }
    processor = RLDXProcessor(
        modality_configs=modality_configs,
        statistics=statistics,
        model_name=model_name,
        model_type="vtc_qwen3_vl",
        use_percentiles=False,
        transformers_loading_kwargs=_VLM_LOADING_KWARGS,
    )
    processor.eval()
    return processor


def test_multiframe_forward_matches_vendored() -> None:
    """Native multi-frame VLM inputs match the vendored upstream stacking.

    Runs the native :meth:`Rldx1Preprocessor.forward` (eval) on a 4-frame /
    2-view observation and the vendored upstream ``_get_vlm_inputs`` + collator
    on pixel-identical frames, then asserts identical ``pixel_values``,
    ``image_grid_thw`` and ``num_frames`` -- the frame-major / view-inner
    ordering and the reported frame count.
    """
    from physicalai.policies.rldx1.preprocessing import formalize_language as native_formalize
    from physicalai.policies.rldx1.preprocessor import Rldx1Preprocessor

    try:
        processor = _vendored_vtc_processor(_VLM_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 - any load failure -> skip offline
        pytest.skip(f"Qwen processor unavailable offline: {exc}")

    native_batch, oracle_images = _make_multiframe_obs(
        num_frames=_VIDEO_LENGTH,
        num_views=_NUM_VIEWS,
        seed=3,
    )

    # -- oracle: upstream frame stacking + collation (eval geometry) --
    language = native_formalize("Pick the RED block!")
    vlm = processor._get_vlm_inputs(  # noqa: SLF001 - parity oracle
        [f"cam{i}" for i in range(_NUM_VIEWS)],
        oracle_images,
        processor.eval_image_transform,
        language,
        memory_length=1,
    )
    assert vlm["vlm_content"]["num_frames"] == _VIDEO_LENGTH
    gold = processor.collator._collate_vlm_content([vlm["vlm_content"]])  # noqa: SLF001

    # -- native: multi-frame preprocessor forward (eval) --
    stats = {OBSERVATION_STATE: _stat(STATE_DIM, seed=0), "action": _stat(ACTION_DIM, seed=1)}
    pre = Rldx1Preprocessor(
        stats=stats,
        max_state_dim=MAX_STATE_DIM,
        max_action_dim=MAX_ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        use_percentiles=False,
        model_name=_VLM_MODEL_NAME,
    )
    pre.eval()
    out = pre.forward(native_batch)

    assert int(out["num_frames"][0]) == _VIDEO_LENGTH
    assert int(out["num_views"][0]) == _NUM_VIEWS
    torch.testing.assert_close(out["image_grid_thw"], gold["image_grid_thw"], atol=0, rtol=0)
    torch.testing.assert_close(out["pixel_values"], gold["pixel_values"], atol=0.0, rtol=0.0)
