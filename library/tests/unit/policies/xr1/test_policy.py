# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the XR1 Lightning policy."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from jsonargparse import Namespace
from physicalai.cli import export as cli_export
from physicalai.cli.export import build_parser
from physicalai.data import FeatureType, Observation
from physicalai.export import ExportBackend
from physicalai.inference import InferenceModel
from physicalai.policies import get_policy
from physicalai.policies.xr1 import XR1, XR1Config
from physicalai.policies.xr1.preprocessor import XR1Preprocessor
from physicalai.policies.xr1.vla import XR1Model

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.policies.xr1.vlm import XR1Qwen3VL

STATE_DIM = 5
ACTION_DIM = 6
CHUNK_SIZE = 4

TINY_KWARGS: dict[str, Any] = {
    "vlm_pretrained": False,
    "dtype": "float32",
    "chunk_size": CHUNK_SIZE,
    "n_action_steps": CHUNK_SIZE,
    "max_state_dim": 8,
    "max_action_dim": 8,
    "dit_num_layers": 4,
    "dit_hidden_size": 256,
    "dit_head_dim": 32,
    "dit_kv_heads": 2,
    "num_inference_steps": 2,
    "training_repeat": 1,
    "image_resolution": (64, 64),
    "camera_views": ("top",),
    "gradient_checkpointing": False,
}


@pytest.fixture
def dataset_stats() -> dict[str, dict[str, Any]]:
    """Return dataset statistics in the shape ``Dataset.stats`` produces.

    ``type`` is a plain string, which is what LeRobot writes and therefore what ends
    up in a checkpoint's hyperparameters. Storing the enum instead would make the
    checkpoint unloadable under ``torch.load(weights_only=True)``.

    Returns:
        Statistics for one state feature, one camera and the action.
    """
    return {
        "observation.state": {
            "name": "state",
            "type": str(FeatureType.STATE),
            "shape": (STATE_DIM,),
            "mean": [0.0] * STATE_DIM,
            "std": [1.0] * STATE_DIM,
        },
        "observation.images.top": {
            "name": "top",
            "type": str(FeatureType.VISUAL),
            "shape": (3, 96, 96),
            "mean": [0.0] * 3,
            "std": [1.0] * 3,
        },
        "action": {
            "name": "action",
            "type": str(FeatureType.ACTION),
            "shape": (ACTION_DIM,),
            "mean": [0.0] * ACTION_DIM,
            "std": [1.0] * ACTION_DIM,
        },
    }


@pytest.fixture
def offline_backbone(monkeypatch: pytest.MonkeyPatch, tiny_vlm: XR1Qwen3VL) -> XR1Qwen3VL:
    """Make model construction use the tiny random backbone instead of the Hub.

    Args:
        monkeypatch: Pytest patcher.
        tiny_vlm: The small backbone.

    Returns:
        The backbone that will be injected.
    """
    monkeypatch.setattr(XR1Model, "_build_vlm", staticmethod(lambda _config: tiny_vlm))
    return tiny_vlm


@pytest.fixture
def policy(
    offline_backbone: XR1Qwen3VL,
    dataset_stats: dict[str, dict[str, Any]],
    stub_processor: Any,
) -> XR1:
    """Build an eagerly initialized policy with a stubbed processor.

    Args:
        offline_backbone: Ensures no Hub download happens.
        dataset_stats: Statistics driving normalization and schemas.
        stub_processor: Processor stand-in.

    Returns:
        The policy, ready for forward and inference calls.
    """
    del offline_backbone
    policy = XR1(dataset_stats=dataset_stats, **TINY_KWARGS)
    policy._preprocessor._processor = stub_processor  # noqa: SLF001 - avoids a tokenizer download
    return policy


def make_observation() -> Observation:
    """Build a two-sample observation batch.

    Returns:
        An observation with state, action, one camera and instructions.
    """
    return Observation(
        state=torch.randn(2, STATE_DIM),
        action=torch.randn(2, CHUNK_SIZE, ACTION_DIM),
        images={"top": torch.rand(2, 3, 96, 96)},
        task=["transfer the cube", "transfer the cube"],
    )


class TestRegistration:
    """The policy must be reachable the way the other families are."""

    def test_get_policy_returns_xr1(self) -> None:
        """``get_policy("xr1")`` constructs without touching the Hub."""
        assert isinstance(get_policy("xr1", source="physicalai", **TINY_KWARGS), XR1)

    def test_exported_from_package_root(self) -> None:
        """The class is importable from ``physicalai.policies``."""
        from physicalai.policies import XR1 as exported

        assert exported is XR1


class TestLazyInitialization:
    """The model is built from the datamodule, not at construction."""

    def test_model_absent_without_stats(self) -> None:
        """Constructing without statistics leaves the model unbuilt."""
        policy = XR1(**TINY_KWARGS)

        assert policy.model is None

    def test_forward_explains_missing_initialization(self) -> None:
        """The error tells the user how to fix it."""
        policy = XR1(**TINY_KWARGS)
        policy.train()

        with pytest.raises(ValueError, match="dataset_stats"):
            policy(make_observation())

    def test_eager_initialization_builds_everything(self, policy: XR1) -> None:
        """Passing statistics builds the model and both processors."""
        assert policy.model is not None
        assert policy._preprocessor is not None  # noqa: SLF001 - initialization is the behavior
        assert policy._postprocessor is not None  # noqa: SLF001


class TestConfigPlumbing:
    """Constructor arguments, config and hparams must agree."""

    def test_config_reflects_arguments(self) -> None:
        """Every constructor argument lands in the config."""
        policy = XR1(**{**TINY_KWARGS, "freeze_vlm": True, "num_inference_steps": 7})

        assert policy.config.freeze_vlm is True
        assert policy.config.num_inference_steps == 7

    def test_defaults_match_the_config(self) -> None:
        """The constructor restates every default, so the two can silently drift.

        jsonargparse needs the arguments spelled out for the CLI, which duplicates
        every default from ``XR1Config``. This is the only thing stopping a config
        default from being changed in one place and not the other.
        """
        policy = XR1(vlm_pretrained=False)

        assert dataclasses.asdict(policy.config) == dataclasses.asdict(XR1Config(vlm_pretrained=False))

    def test_slot_maps_reach_the_config(self) -> None:
        """Slot maps arrive from YAML as lists and must end up as tuples."""
        policy = XR1(**{**TINY_KWARGS, "state_slot_map": [0, 2, 4, 5, 6]})

        assert policy.config.state_slot_map == (0, 2, 4, 5, 6)

    def test_hparams_carry_resolved_config(self) -> None:
        """Checkpoints must round-trip the resolved config, not the raw arguments."""
        policy = XR1(**TINY_KWARGS)

        assert policy.hparams["config"]["dit_num_layers"] == TINY_KWARGS["dit_num_layers"]

    def test_action_queue_uses_n_action_steps(self) -> None:
        """The base class action queue is sized from n_action_steps."""
        policy = XR1(**{**TINY_KWARGS, "chunk_size": 8, "n_action_steps": 3})

        assert policy._n_action_steps == 3  # noqa: SLF001 - base-class queue size

    def test_features_from_stats(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Statistics are turned back into a feature schema."""
        features = XR1.features_from_stats(dataset_stats)

        assert set(features) == {"state", "top", "action"}
        assert features["action"].shape == (ACTION_DIM,)


class TestTrainingAndInference:
    """End-to-end behavior through the Lightning surface."""

    def test_forward_returns_loss_in_training(self, policy: XR1) -> None:
        """Training mode returns a loss and its metrics."""
        policy.train()

        loss, metrics = policy(make_observation())

        assert torch.isfinite(loss)
        assert "loss" in metrics

    def test_training_step(self, policy: XR1) -> None:
        """The Lightning step returns a differentiable loss."""
        policy.train()

        loss = policy.training_step(make_observation(), 0)

        assert loss.requires_grad

    def test_predict_action_chunk_is_denormalized_to_dataset_width(self, policy: XR1) -> None:
        """Predictions come back at the dataset's action width, not padded."""
        policy.eval()

        chunk = policy.predict_action_chunk(make_observation())

        assert chunk.shape == (2, CHUNK_SIZE, ACTION_DIM)

    def test_select_action_drains_the_queue(self, policy: XR1) -> None:
        """The base-class queue serves one action per call from one chunk."""
        policy.eval()
        policy.reset()
        observation = make_observation()

        first = policy.select_action(observation)
        second = policy.select_action(observation)

        assert first.shape == (2, ACTION_DIM)
        assert second.shape == (2, ACTION_DIM)

    def test_compute_val_loss(self, policy: XR1) -> None:
        """Validation scores a full rollout against the ground truth."""
        policy.eval()

        loss, metrics = policy.compute_val_loss(make_observation())

        assert torch.isfinite(loss)
        assert "action_mse" in metrics


class TestExportSurface:
    """What the export mixin and the runtime manifest need."""

    def test_supported_backends(self) -> None:
        """Torch export is supported; graph backends are not claimed yet."""
        assert XR1.get_supported_export_backends() == [ExportBackend.TORCH]

    def test_schemas_absent_before_initialization(self) -> None:
        """Schemas depend on dataset statistics."""
        policy = XR1(**TINY_KWARGS)

        assert policy.inputs_schema is None
        assert policy.outputs_schema is None

    def test_inputs_schema_covers_state_images_and_language(self, policy: XR1) -> None:
        """Runtime preprocessing keys off these names."""
        names = [feature.name for feature in policy.inputs_schema or []]

        assert "state" in names
        assert "images" in names
        assert "task" in names

    def test_outputs_schema_shape(self, policy: XR1) -> None:
        """The action feature carries the full chunk shape."""
        schema = policy.outputs_schema or []

        assert len(schema) == 1
        assert schema[0].shape == (CHUNK_SIZE, ACTION_DIM)

    def test_extra_export_args_requires_stats(self) -> None:
        """Exporting without statistics cannot produce a valid manifest."""
        policy = XR1(**TINY_KWARGS)

        with pytest.raises(ValueError, match="Dataset stats are required"):
            _ = policy.extra_export_args

    def test_extra_export_args_has_torch_entry(self, policy: XR1) -> None:
        """The torch backend contributes its preprocessor spec."""
        assert "torch" in policy.extra_export_args

    def test_chunk_trimmer_added_when_horizons_differ(
        self,
        offline_backbone: XR1Qwen3VL,
        dataset_stats: dict[str, dict[str, Any]],
    ) -> None:
        """Executing fewer steps than predicted needs a trimming postprocessor."""
        del offline_backbone
        policy = XR1(dataset_stats=dataset_stats, **{**TINY_KWARGS, "chunk_size": 8, "n_action_steps": 2})

        specs = policy.extra_export_args["torch"].postprocessors_specs

        assert any(spec.type == "action_chunk_trimmer" for spec in specs)

class TestTorchExportRoundTrip:
    """Milestones 5 and 6: export through ``.to_torch()``, then load with Runtime.

    Everything here runs against the tiny backbone, so the round trip is exercised in
    CI rather than only in a manual script.
    """

    @pytest.fixture
    def exported(self, policy: XR1, tmp_path: Path, stub_processor: Any) -> Path:
        """Export the policy and make the reload path work offline.

        Args:
            policy: An initialized policy.
            tmp_path: Export directory.
            stub_processor: Processor stand-in, also used by the reloaded policy.

        Returns:
            The export directory.
        """
        export_dir = tmp_path / "export-torch"
        policy.export(export_dir, backend=ExportBackend.TORCH)
        # ``InferenceModel`` reconstructs the policy from the checkpoint, which would
        # otherwise fetch the real processor from the Hub.
        XR1Preprocessor.processor = property(lambda _self: stub_processor)  # type: ignore[method-assign, assignment]
        return export_dir

    @staticmethod
    def _sample(policy: XR1) -> dict[str, Any]:
        """Build one numpy observation matching the exported input schema.

        Args:
            policy: The policy, read for its input schema.

        Returns:
            An observation dict of numpy arrays, as Runtime receives it.
        """
        rng = np.random.RandomState(0)
        shapes = {feature.name: feature.shape for feature in policy.inputs_schema or []}
        return {
            "state": rng.randn(1, *shapes["state"]).astype(np.float32),
            "images": rng.rand(1, *shapes["images"]).astype(np.float32),
            "task": ["transfer the cube"],
        }

    def test_artifact_and_manifest(self, policy: XR1, exported: Path) -> None:
        """The export contract needs a backend-identifying file plus metadata."""
        manifest = json.loads((exported / "manifest.json").read_text())

        assert (exported / "xr1.pt").is_file()
        assert manifest["model"]["artifacts"] == {"torch": "xr1.pt"}
        assert manifest["policy"]["source"]["class_path"].endswith("xr1.policy.XR1")
        output_feature = manifest["model"]["output_features"][0]["init_args"]
        assert output_feature["name"] == "action"
        assert output_feature["shape"] == [policy.config.chunk_size, ACTION_DIM]

    def test_cli_matches_the_python_api(self, policy: XR1, exported: Path, tmp_path: Path) -> None:
        """Both routes must produce the same artifact, per the export contract."""
        cli_dir = tmp_path / "export-cli"

        cli_export.run(
            build_parser(),
            Namespace(
                policy="physicalai.policies.XR1",
                # ``to_torch`` writes a Lightning-shaped checkpoint, so the artifact
                # from the API route is itself a valid CLI input.
                ckpt_path=str(exported / "xr1.pt"),
                backend="torch",
                output_dir=str(cli_dir),
            ),
        )

        assert sorted(p.name for p in cli_dir.iterdir()) == sorted(p.name for p in exported.iterdir())
        assert json.loads((cli_dir / "manifest.json").read_text()) == json.loads(
            (exported / "manifest.json").read_text(),
        )

    def test_inference_model_loads_the_artifact(self, exported: Path) -> None:
        """Runtime must auto-detect the backend from the ``.pt`` extension."""
        model = InferenceModel(exported, device="cpu")

        assert model.backend == "torch"

    def test_matches_the_torch_policy_path(self, policy: XR1, exported: Path) -> None:
        """Parity is what proves the exported artifact is the same model.

        The sampler is stochastic, so both paths integrate from the same RNG state.
        """
        observation = self._sample(policy)
        model = InferenceModel(exported, device="cpu")
        policy.eval()

        state = torch.get_rng_state()
        runtime_chunk = np.asarray(model.predict_action_chunk(observation))
        torch.set_rng_state(state)
        with torch.no_grad():
            torch_chunk = policy.predict_action_chunk(Observation.from_dict(observation).to_torch("cpu"))

        expected = torch_chunk.squeeze(0).cpu().numpy()
        assert runtime_chunk.shape == expected.shape
        np.testing.assert_allclose(runtime_chunk, expected, rtol=0, atol=0)

    def test_select_action_drains_the_chunk(self, policy: XR1, exported: Path) -> None:
        """Runtime serves one action per step out of a single chunk."""
        observation = self._sample(policy)
        model = InferenceModel(exported, device="cpu")
        model.reset()

        actions = [np.asarray(model.select_action(observation)) for _ in range(3)]

        assert [a.shape for a in actions] == [(ACTION_DIM,)] * 3
