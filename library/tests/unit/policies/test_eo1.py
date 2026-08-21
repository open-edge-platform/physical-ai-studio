# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the EO-1 policy.

Fast, self-contained tests with no external dependencies (no HuggingFace model downloads), so the
Qwen2.5-VL backbone is never built here. The ported flow-matching primitives, the conversation
template and the pre/postprocessing pipeline are exercised directly on synthetic tensors.
"""

from __future__ import annotations

import pytest
import torch
from physicalai.config import Config
from physicalai.policies import get_policy
from physicalai.policies.eo1 import EO1, EO1Config
from physicalai.policies.eo1.components.flow_matching import (
    create_sinusoidal_pos_embedding,
    euler_integrate,
    pad_vector,
    sample_noise,
    sample_time_beta,
)
from physicalai.policies.eo1.components.qwen_interface import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_STATE_TOKEN,
    SYSTEM_MESSAGE,
    TASK_VLA_TOKEN,
    EO1QwenInterface,
    to_uint8_image,
)
from physicalai.policies.eo1.model import EO1ActionProjector, EO1Model
from physicalai.policies.eo1.preprocessor import make_eo1_preprocessors
from physicalai.policies.eo1.pretrained_utils import fix_state_dict_keys

ACTION_DIM = 3
STATE_DIM = 4
CHUNK_SIZE = 4


def tiny_config(**overrides: object) -> EO1Config:
    """Build a config small enough to exercise the ported pieces in a unit test.

    Args:
        **overrides: Fields to override on top of the tiny defaults.

    Returns:
        An EO1Config sized for CPU tests.
    """
    kwargs: dict[str, object] = {
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": CHUNK_SIZE,
        "action_dim": ACTION_DIM,
        "state_dim": STATE_DIM,
        "max_action_dim": 8,
        "max_state_dim": 8,
        "num_denoise_steps": 2,
    }
    kwargs.update(overrides)
    return EO1Config(**kwargs)  # type: ignore[arg-type]


def tiny_stats() -> dict[str, dict[str, object]]:
    """Build dataset statistics for a 3-dim action and 4-dim state.

    Returns:
        Stats dict in the format produced by ``Dataset.stats``.
    """
    return {
        "observation.state": {
            "name": "state",
            "type": "STATE",
            "shape": (STATE_DIM,),
            "mean": [0.0] * STATE_DIM,
            "std": [2.0] * STATE_DIM,
        },
        "action": {
            "name": "action",
            "type": "ACTION",
            "shape": (ACTION_DIM,),
            "mean": [1.0] * ACTION_DIM,
            "std": [2.0] * ACTION_DIM,
        },
    }


def offline_interface(config: EO1Config) -> EO1QwenInterface:
    """Build an interface without loading the Qwen processor.

    ``build_messages`` only reads ``config.chunk_size``, so the prompt layout can be tested without
    reaching the Hub. Everything that needs the tokenizer is out of scope here.

    Args:
        config: Policy configuration.

    Returns:
        An interface with only its config populated.
    """
    interface = EO1QwenInterface.__new__(EO1QwenInterface)
    interface.config = config
    return interface


# ============================================================================ #
# Configuration Tests                                                          #
# ============================================================================ #


class TestEO1Config:
    """Tests for the EO1Config dataclass."""

    def test_default_config(self) -> None:
        """Test the published EO-1 defaults are what the dataclass carries."""
        config = EO1Config()
        assert config.vlm_base == "Qwen/Qwen2.5-VL-3B-Instruct"
        assert config.chunk_size == 8
        assert config.n_action_steps == 8
        assert config.max_state_dim == 32
        assert config.max_action_dim == 32
        assert config.num_denoise_steps == 10
        assert config.dtype == "auto"
        assert config.force_fp32_autocast is True
        assert config.vlm_config is None

    def test_custom_config(self) -> None:
        """Test constructor overrides land on the dataclass."""
        config = tiny_config(attn_implementation="sdpa", gradient_checkpointing=True)
        assert config.chunk_size == CHUNK_SIZE
        assert config.attn_implementation == "sdpa"
        assert config.gradient_checkpointing is True

    def test_n_action_steps_validation(self) -> None:
        """Test executing more steps than the chunk holds is rejected."""
        with pytest.raises(ValueError, match="chunk size is the upper bound"):
            tiny_config(n_action_steps=CHUNK_SIZE + 1)

    def test_unknown_dtype_is_rejected(self) -> None:
        """Test a dtype the backbone loader cannot honor fails at construction."""
        with pytest.raises(ValueError, match="Unknown dtype"):
            tiny_config(dtype="float16")

    def test_action_dim_beyond_padding_is_rejected(self) -> None:
        """Test an action wider than the flow head's padded width fails loudly."""
        with pytest.raises(ValueError, match="exceeds `max_action_dim`"):
            tiny_config(action_dim=16, max_action_dim=8)

    def test_state_dim_beyond_padding_is_rejected(self) -> None:
        """Test a state wider than the state projection's padded width fails loudly."""
        with pytest.raises(ValueError, match="exceeds `max_state_dim`"):
            tiny_config(state_dim=16, max_state_dim=8)

    def test_inheritance_and_serialization(self) -> None:
        """Test the config is a Studio Config and survives a dict round-trip."""
        config = tiny_config()
        assert isinstance(config, Config)
        assert EO1Config.from_dict(config.to_dict()) == config

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Test a published LeRobot config.json parses despite its extra keys."""
        data = {**tiny_config().to_dict(), "normalization_mapping": {"STATE": "MEAN_STD"}}
        config = EO1Config.from_dict(data, strict=False)
        assert config.chunk_size == CHUNK_SIZE


class TestDeltaIndices:
    """Tests for the timestamp windows the config hands to the dataloader."""

    def test_observation_delta_indices(self) -> None:
        """Test EO-1 asks for the current frame only."""
        assert tiny_config().observation_delta_indices is None

    def test_action_delta_indices(self) -> None:
        """Test one action index is requested per chunk step."""
        assert tiny_config().action_delta_indices == list(range(CHUNK_SIZE))


# ============================================================================ #
# Flow-Matching Primitives                                                     #
# ============================================================================ #


class TestFlowMatching:
    """Tests for the ported openpi flow-matching helpers."""

    def test_pad_vector_pads_and_passes_through(self) -> None:
        """Test narrow vectors are zero-padded and wide ones left alone."""
        padded = pad_vector(torch.ones(2, 3), 5)
        assert padded.shape == (2, 5)
        assert torch.equal(padded[:, 3:], torch.zeros(2, 2))

        wide = torch.ones(2, 7)
        assert pad_vector(wide, 5) is wide

    def test_sinusoidal_embedding_shape(self) -> None:
        """Test the timestep embedding matches the requested width."""
        time = torch.linspace(0.0, 1.0, 3)
        embedding = create_sinusoidal_pos_embedding(time, 8, 4e-3, 4.0, torch.device("cpu"))
        assert embedding.shape == (3, 8)

    def test_sinusoidal_embedding_rejects_odd_dimension(self) -> None:
        """Test an odd width cannot be split into sine and cosine halves."""
        with pytest.raises(ValueError, match="divisible by 2"):
            create_sinusoidal_pos_embedding(torch.zeros(2), 7, 4e-3, 4.0, torch.device("cpu"))

    def test_sinusoidal_embedding_rejects_bad_rank(self) -> None:
        """Test only per-sample and per-step timesteps are accepted."""
        with pytest.raises(ValueError, match="must have shape"):
            create_sinusoidal_pos_embedding(torch.zeros(2, 3, 4), 8, 4e-3, 4.0, torch.device("cpu"))

    def test_sample_time_beta_is_in_range(self) -> None:
        """Test sampled timesteps stay inside the configured scale and offset."""
        time = sample_time_beta(64, torch.device("cpu"), alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
        assert time.shape == (64,)
        assert time.dtype == torch.float32
        assert bool(((time >= 0.001) & (time <= 1.0)).all())

    def test_sample_noise_shape_and_dtype(self) -> None:
        """Test the flow-matching noise draw is float32 of the requested shape."""
        noise = sample_noise((2, CHUNK_SIZE, 8), torch.device("cpu"))
        assert noise.shape == (2, CHUNK_SIZE, 8)
        assert noise.dtype == torch.float32

    def test_euler_integrate_recovers_a_constant_field(self) -> None:
        """Test integrating a constant velocity from t=1 to t=0 shifts by that velocity."""
        velocity = torch.full((2, 3), 0.5)
        result = euler_integrate(lambda x_t, _t: velocity.to(x_t.dtype), torch.zeros(2, 3), 10)
        # dt = -1/10 for 10 steps, so x_0 = x_1 - velocity.
        assert torch.allclose(result, -velocity, atol=1e-5)

    def test_euler_integrate_sees_descending_timesteps(self) -> None:
        """Test the loop hands the velocity field times running from 1.0 down to just above 0."""
        seen: list[float] = []

        def denoise(x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            seen.append(float(time[0]))
            return torch.zeros_like(x_t)

        euler_integrate(denoise, torch.zeros(1, 2), 4)
        assert seen == pytest.approx([1.0, 0.75, 0.5, 0.25])


class TestActionProjector:
    """Tests for the flow head's output projector."""

    def test_layer_layout_and_shape(self) -> None:
        """Test the projector alternates linear layers and activations, ending on a linear."""
        projector = EO1ActionProjector(16, 8, num_layers=2, activation_layer="linear")
        assert len(projector) == 3
        assert projector.dtype == torch.float32
        assert projector(torch.randn(2, CHUNK_SIZE, 16)).shape == (2, CHUNK_SIZE, 8)

    def test_single_layer(self) -> None:
        """Test a one-layer projector is a bare linear map."""
        projector = EO1ActionProjector(16, 8, num_layers=1)
        assert len(projector) == 1
        assert projector(torch.randn(2, 16)).shape == (2, 8)


# ============================================================================ #
# Conversation Template                                                        #
# ============================================================================ #


class TestConversationTemplate:
    """Tests for the EO-1 prompt layout and image conversion."""

    def test_messages_carry_the_placeholders(self) -> None:
        """Test each sample gets one state placeholder and chunk_size action placeholders."""
        interface = offline_interface(tiny_config())
        messages = interface.build_messages(
            [[torch.rand(3, 8, 8), torch.rand(3, 8, 8)]],
            ["pick up the cube"],
        )
        assert len(messages) == 1
        system, user, assistant = messages[0]

        assert system["content"][0]["text"] == SYSTEM_MESSAGE
        assert [entry["type"] for entry in user["content"]] == ["image", "image", "text"]

        prompt = user["content"][-1]["text"]
        assert prompt.count(DEFAULT_STATE_TOKEN) == 1
        assert "pick up the cube" in prompt
        assert prompt.endswith(TASK_VLA_TOKEN)

        answer = assistant["content"][0]["text"]
        assert answer.count(DEFAULT_ACTION_TOKEN) == CHUNK_SIZE
        assert answer.startswith(ACTION_START_TOKEN)
        assert answer.endswith(ACTION_END_TOKEN)

    def test_messages_are_built_per_sample(self) -> None:
        """Test the batch is unrolled into one conversation per sample."""
        interface = offline_interface(tiny_config())
        messages = interface.build_messages([[torch.rand(3, 8, 8)], [torch.rand(3, 8, 8)]], ["a", "b"])
        assert len(messages) == 2
        assert "a" in messages[0][1]["content"][-1]["text"]
        assert "b" in messages[1][1]["content"][-1]["text"]

    def test_mismatched_batch_sizes_are_rejected(self) -> None:
        """Test a task list that does not line up with the images fails rather than truncating."""
        interface = offline_interface(tiny_config())
        with pytest.raises(ValueError, match="argument 2 is shorter"):
            interface.build_messages([[torch.rand(3, 8, 8)], [torch.rand(3, 8, 8)]], ["only one"])

    def test_to_uint8_image_quantizes_floats(self) -> None:
        """Test float images in [0, 1] become uint8 in [0, 255]."""
        image = torch.tensor([0.0, 0.5, 1.0]).reshape(3, 1, 1)
        converted = to_uint8_image(image)
        assert converted.dtype == torch.uint8
        assert converted.flatten().tolist() == [0, 128, 255]

    def test_to_uint8_image_expands_grayscale(self) -> None:
        """Test a single channel is repeated to three for the RGB processor."""
        assert to_uint8_image(torch.rand(1, 4, 4)).shape == (3, 4, 4)
        assert to_uint8_image(torch.rand(2, 1, 4, 4)).shape == (2, 3, 4, 4)

    def test_to_uint8_image_is_idempotent(self) -> None:
        """Test an already-quantized image passes through unchanged."""
        once = to_uint8_image(torch.rand(3, 4, 4))
        assert torch.equal(to_uint8_image(once), once)


# ============================================================================ #
# Pre/Postprocessing                                                           #
# ============================================================================ #


class TestPreprocessors:
    """Tests for the pre/postprocessor pair."""

    def test_normalization_round_trip(self) -> None:
        """Test actions survive normalize -> denormalize unchanged."""
        pre, post = make_eo1_preprocessors(tiny_config(), tiny_stats())
        action = torch.tensor([[[0.5, -1.5, 2.0], [0.0, 1.0, -2.0]]])
        batch = {"state": torch.zeros(1, STATE_DIM), "action": action.clone()}

        processed = pre(batch)
        assert torch.allclose(processed["action"], (action - 1.0) / 2.0)

        restored = post({"action": processed["action"]})["action"]
        assert torch.allclose(restored, action, atol=1e-5)

    def test_state_is_normalized_with_mean_std(self) -> None:
        """Test the state uses mean/std normalization."""
        pre, _ = make_eo1_preprocessors(tiny_config(), tiny_stats())
        processed = pre({"state": torch.full((1, STATE_DIM), 4.0)})
        assert torch.allclose(processed["state"], torch.full((1, STATE_DIM), 2.0), atol=1e-5)

    def test_images_are_passed_through(self) -> None:
        """Test visual observations reach the model untouched; the Qwen interface converts them."""
        pre, _ = make_eo1_preprocessors(tiny_config(), tiny_stats())
        image = torch.rand(1, 3, 8, 8)
        processed = pre({"state": torch.zeros(1, STATE_DIM), "images.top": image})
        assert torch.equal(processed["images.top"], image)

    def test_postprocessor_ignores_batches_without_actions(self) -> None:
        """Test the postprocessor is a no-op when the model produced nothing to denormalize."""
        _, post = make_eo1_preprocessors(tiny_config(), tiny_stats())
        result = post({"state": torch.zeros(1, STATE_DIM)})
        assert list(result) == ["state"]
        assert torch.equal(result["state"], torch.zeros(1, STATE_DIM))

    def test_missing_stats_for_normalization_raise(self) -> None:
        """Test MEAN_STD actions without mean/std statistics fail with a clear message."""
        stats = tiny_stats()
        del stats["action"]["std"]
        with pytest.raises(ValueError, match="no std statistics, which MEAN_STD"):
            make_eo1_preprocessors(tiny_config(), stats)

    def test_unknown_normalization_is_rejected(self) -> None:
        """Test a typo in the normalization name fails at construction, not at runtime."""
        with pytest.raises(ValueError, match="Unknown normalization"):
            make_eo1_preprocessors(tiny_config(action_normalization="MEANSTD"), tiny_stats())


class TestActionChunkValidation:
    """Tests for the guard that keeps the action horizon aligned with the prompt placeholders."""

    @staticmethod
    def _validate(actions: torch.Tensor) -> None:
        """Call the model's guard without building a backbone."""
        stub = type("_Stub", (), {"config": tiny_config()})()
        EO1Model._validate_action_chunk(stub, actions)  # type: ignore[arg-type]  # noqa: SLF001

    def test_matching_chunk_is_accepted(self) -> None:
        """Test a correctly chunked action passes."""
        self._validate(torch.zeros(2, CHUNK_SIZE, ACTION_DIM))

    def test_unchunked_action_is_rejected(self) -> None:
        """Test a dataset without action delta timestamps fails with an actionable message."""
        with pytest.raises(ValueError, match="action_delta_indices"):
            self._validate(torch.zeros(2, ACTION_DIM))

    def test_wrong_horizon_is_rejected(self) -> None:
        """Test a chunk shorter than the prompt's placeholder count is caught before tokenizing."""
        with pytest.raises(ValueError, match=f"chunk_size={CHUNK_SIZE}"):
            self._validate(torch.zeros(2, CHUNK_SIZE - 1, ACTION_DIM))


class TestPretrainedUtils:
    """Tests for the published-checkpoint adapters."""

    def test_fix_state_dict_keys_strips_one_model_prefix(self) -> None:
        """Test only the LeRobot policy's own ``model.`` wrapper is removed."""
        state_dict = {
            "model.state_proj.weight": torch.zeros(1),
            "model.vlm_backbone.model.language_model.layers.0.weight": torch.zeros(1),
            "already_bare.weight": torch.zeros(1),
        }
        fixed = fix_state_dict_keys(state_dict)
        assert set(fixed) == {
            "state_proj.weight",
            "vlm_backbone.model.language_model.layers.0.weight",
            "already_bare.weight",
        }


# ============================================================================ #
# Policy                                                                       #
# ============================================================================ #


class TestEO1Policy:
    """Tests for the Lightning wrapper that do not build the backbone."""

    def test_lazy_construction_defers_model(self) -> None:
        """Test constructing the policy does not download or build the backbone."""
        policy = EO1(chunk_size=CHUNK_SIZE, n_action_steps=CHUNK_SIZE)
        assert policy.model is None
        assert policy.config.chunk_size == CHUNK_SIZE

    def test_config_is_saved_to_hparams(self) -> None:
        """Test the resolved config lands in the checkpoint hyperparameters."""
        policy = EO1(chunk_size=CHUNK_SIZE, n_action_steps=2, num_denoise_steps=3)
        assert policy.hparams["chunk_size"] == CHUNK_SIZE
        assert policy.hparams["config"]["num_denoise_steps"] == 3

    def test_predict_before_setup_raises(self) -> None:
        """Test inference without an initialized model fails with a clear message."""
        policy = EO1()
        with pytest.raises(ValueError, match="Model is not initialized"):
            policy.predict_action_chunk(None)  # type: ignore[arg-type]

    def test_get_policy_factory(self) -> None:
        """Test the policy is reachable through the name-based factory."""
        assert isinstance(get_policy("eo1"), EO1)

    def test_action_queue_uses_n_action_steps(self) -> None:
        """Test the inherited action queue is sized from n_action_steps."""
        policy = EO1(chunk_size=CHUNK_SIZE, n_action_steps=2)
        chunk = torch.arange(CHUNK_SIZE * ACTION_DIM, dtype=torch.float32).reshape(1, CHUNK_SIZE, ACTION_DIM)
        first = policy._queue_actions(chunk)  # noqa: SLF001
        assert torch.equal(first, chunk[:, 0])
        assert len(policy._action_queue) == 1  # noqa: SLF001


class TestExport:
    """Tests for the ExportablePolicyMixin surface.

    Uses a lightweight stub instead of constructing the full model, so the Qwen2.5-VL backbone is
    never downloaded.
    """

    @staticmethod
    def _stub(dataset_stats: dict | None, **config_overrides: object) -> EO1:
        """Build a stand-in exposing only what the export properties read."""

        class _Stub:
            def __init__(self) -> None:
                self._dataset_stats = dataset_stats
                self.model = torch.nn.Linear(1, 1)
                self.config = tiny_config(**config_overrides)

        return _Stub()  # type: ignore[return-value]

    @staticmethod
    def _stats() -> dict:
        """Dataset stats in the shape the policy stores them."""
        return {
            "observation.state": {"name": "observation.state", "shape": (STATE_DIM,), "type": "STATE"},
            # Real datasets carry the dotted feature path in `name`; the schema must strip the
            # `observation.`/`images.` prefixes rather than nest them (`images.images.top`).
            "observation.images.top": {"name": "images.top", "shape": (3, 480, 640), "type": "VISUAL"},
            "observation.images.wrist": {"name": "images.wrist", "shape": (3, 480, 640), "type": "VISUAL"},
            "action": {"name": "action", "shape": (ACTION_DIM,), "type": "ACTION"},
        }

    def test_only_torch_backend_is_supported(self) -> None:
        """Test tracing backends are not advertised."""
        from physicalai.export import ExportBackend

        assert EO1.get_supported_export_backends() == [ExportBackend.TORCH]

    def test_schemas_are_none_before_setup(self) -> None:
        """Test the schemas stay None until the model and stats exist."""
        policy = EO1()
        assert policy.inputs_schema is None
        assert policy.outputs_schema is None

    def test_inputs_schema_covers_state_images_and_task(self) -> None:
        """Test every observation modality is described once."""
        from physicalai.data.observation import IMAGES, STATE, TASK

        schema = EO1.inputs_schema.fget(self._stub(self._stats()))  # type: ignore[attr-defined]
        names = [feature.name for feature in schema]
        assert names == [STATE, f"{IMAGES}.top", f"{IMAGES}.wrist", TASK]

    def test_inputs_schema_reports_the_dataset_resolution(self) -> None:
        """Test the visual features keep the dataset resolution; Qwen rescales internally."""
        schema = EO1.inputs_schema.fget(self._stub(self._stats()))  # type: ignore[attr-defined]
        visual = [feature for feature in schema if "images" in feature.name]
        assert all(feature.shape == (3, 480, 640) for feature in visual)

    def test_outputs_schema_is_the_action_chunk(self) -> None:
        """Test the output feature carries the full chunk horizon."""
        from physicalai.data.observation import ACTION

        schema = EO1.outputs_schema.fget(self._stub(self._stats()))  # type: ignore[attr-defined]
        assert len(schema) == 1
        assert schema[0].name == ACTION
        assert schema[0].shape == (CHUNK_SIZE, ACTION_DIM)

    def test_sample_input_is_built_from_the_schema(self) -> None:
        """Test the inherited sample_input turns the schema into traceable tensors."""
        from physicalai.data.observation import STATE, TASK

        stub = self._stub(self._stats())
        stub.inputs_schema = EO1.inputs_schema.fget(stub)  # type: ignore[attr-defined]
        sample = EO1.sample_input.fget(stub)  # type: ignore[attr-defined]
        assert sample[STATE].shape == (1, STATE_DIM)
        assert isinstance(sample[TASK], str)

    def test_extra_export_args_trims_the_chunk_when_needed(self) -> None:
        """Test the torch manifest carries a trimmer only when the horizons differ."""
        trimmed = EO1.extra_export_args.fget(  # type: ignore[attr-defined]
            self._stub(self._stats(), n_action_steps=CHUNK_SIZE - 1),
        )
        assert [spec.type for spec in trimmed["torch"].postprocessors_specs] == ["action_chunk_trimmer"]

        untrimmed = EO1.extra_export_args.fget(self._stub(self._stats()))  # type: ignore[attr-defined]
        assert untrimmed["torch"].postprocessors_specs == []
        assert [spec.type for spec in untrimmed["torch"].preprocessors_specs] == ["to_float_tensor"]

    def test_extra_export_args_requires_dataset_stats(self) -> None:
        """Test export fails loudly when normalization statistics are missing."""
        with pytest.raises(ValueError, match="Dataset stats are required"):
            EO1.extra_export_args.fget(self._stub(None))  # type: ignore[attr-defined]
