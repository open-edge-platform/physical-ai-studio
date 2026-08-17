# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for loading released XR-1 checkpoints."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from physicalai.policies.xr1.pretrained_utils import (
    EXPECTED_MISSING,
    LoadReport,
    infer_config_overrides,
    load_pretrained_weights,
    load_state_dict,
    remap_state_dict,
    resolve_checkpoint,
)
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

DIT_LAYERS = 3
DIT_HIDDEN = 256
HEAD_DIM = 64
ACTION_DIM = 60
STATE_DIM = 60


def released_key_shapes() -> dict[str, tuple[int, ...]]:
    """Return a miniature stand-in for a released checkpoint.

    Key names are taken verbatim from
    ``XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa``; only the sizes are shrunk.

    Returns:
        Parameter names mapped to shapes.
    """
    shapes: dict[str, tuple[int, ...]] = {
        "sink.weight": (1, DIT_HIDDEN),
        "action_projector.layers.0.weight": (DIT_HIDDEN, ACTION_DIM),
        "action_projector.layers.2.weight": (DIT_HIDDEN, DIT_HIDDEN),
        "state_projector.layers.0.weight": (DIT_HIDDEN, STATE_DIM),
        "state_projector.layers.2.weight": (DIT_HIDDEN, DIT_HIDDEN),
        "action_output_layer.layers.0.weight": (DIT_HIDDEN, DIT_HIDDEN),
        "action_output_layer.layers.2.weight": (ACTION_DIM, DIT_HIDDEN),
        "t_embedder.mlp.0.weight": (DIT_HIDDEN, 256),
        "t_embedder.mlp.2.weight": (DIT_HIDDEN, DIT_HIDDEN),
        "t_projector.layers.0.weight": (6 * DIT_HIDDEN, DIT_HIDDEN),
        "t_projector.layers.0.bias": (6 * DIT_HIDDEN,),
    }
    for layer in range(DIT_LAYERS):
        shapes[f"dit.layers.{layer}.attn.q_norm.weight"] = (HEAD_DIM,)
        shapes[f"dit.layers.{layer}.attn.k_norm.weight"] = (HEAD_DIM,)
        shapes[f"dit.layers.{layer}.adaln_table"] = (6, DIT_HIDDEN)
    return shapes


def released_state_dict() -> dict[str, torch.Tensor]:
    """Materialize the miniature checkpoint.

    Returns:
        Randomly filled tensors under released key names.
    """
    return {name: torch.zeros(shape) for name, shape in released_key_shapes().items()}


class TestRemapStateDict:
    """Wrapper prefixes must be stripped without touching real names."""

    def test_released_names_pass_through_unchanged(self) -> None:
        """The published names already match this implementation."""
        state_dict = released_state_dict()

        assert set(remap_state_dict(state_dict)) == set(state_dict)

    def test_strips_deepspeed_prefix(self) -> None:
        """The 5B base ships a DeepSpeed checkpoint with a module. prefix."""
        remapped = remap_state_dict({"module.dit.layers.0.adaln_table": torch.zeros(1)})

        assert set(remapped) == {"dit.layers.0.adaln_table"}

    def test_strips_policy_level_prefix(self) -> None:
        """A checkpoint saved at policy level nests everything under model."""
        remapped = remap_state_dict({"model.dit.layers.0.adaln_table": torch.zeros(1)})

        assert set(remapped) == {"dit.layers.0.adaln_table"}

    def test_preserves_backbone_inner_prefix(self) -> None:
        """The backbone's own vlm.model. prefix must survive."""
        remapped = remap_state_dict({"vlm.model.language_model.embed_tokens.weight": torch.zeros(1)})

        assert set(remapped) == {"vlm.model.language_model.embed_tokens.weight"}


class TestInferConfigOverrides:
    """Sizes are read from the checkpoint so they cannot be misconfigured."""

    def test_infers_every_derivable_field(self) -> None:
        """Depth, width, head dim and the action/state widths are all recoverable."""
        overrides = infer_config_overrides(released_state_dict())

        assert overrides == {
            "dit_num_layers": DIT_LAYERS,
            "dit_hidden_size": DIT_HIDDEN,
            "max_action_dim": ACTION_DIM,
            "max_state_dim": STATE_DIM,
            "dit_head_dim": HEAD_DIM,
        }

    def test_matches_the_released_checkpoint_values(self) -> None:
        """Guards the documented sizes of the published 5B checkpoints."""
        shapes = {
            "sink.weight": (1, 1024),
            "action_projector.layers.0.weight": (1024, 60),
            "dit.layers.35.attn.q_norm.weight": (128,),
        }
        overrides = infer_config_overrides({k: torch.zeros(v) for k, v in shapes.items()})

        assert overrides["dit_num_layers"] == 36
        assert overrides["dit_hidden_size"] == 1024
        assert overrides["max_action_dim"] == 60
        assert overrides["dit_head_dim"] == 128

    def test_ignores_absent_fields(self) -> None:
        """A partial checkpoint yields only what it determines."""
        assert infer_config_overrides({"sink.weight": torch.zeros(1, 8)}) == {"dit_hidden_size": 8}


class TinyModel(nn.Module):
    """Stand-in with one released-style parameter plus the known-omitted head."""

    def __init__(self) -> None:
        """Build the module."""
        super().__init__()
        self.sink = nn.Embedding(1, DIT_HIDDEN)
        self.vlm = nn.Module()
        self.vlm.lm_head = nn.Linear(4, 4, bias=False)  # type: ignore[assignment]


class TestLoadPretrainedWeights:
    """Loading reports what happened and fails loudly on a mismatch."""

    def test_known_omission_is_not_reported_missing(self) -> None:
        """Released checkpoints drop vlm.lm_head; that is expected, not an error."""
        model = TinyModel()

        report = load_pretrained_weights(model, {"sink.weight": torch.zeros(1, DIT_HIDDEN)}, strict=True)

        assert report.missing == []
        assert "vlm.lm_head.weight" in EXPECTED_MISSING

    def test_unexpected_key_raises_with_guidance(self) -> None:
        """A size mismatch usually means the config disagrees with the checkpoint."""
        model = TinyModel()

        with pytest.raises(RuntimeError, match="infer_config_overrides"):
            load_pretrained_weights(model, {"dit.layers.0.adaln_table": torch.zeros(6, 8)}, strict=True)

    def test_non_strict_returns_a_report(self) -> None:
        """Non-strict loading surfaces the same information without raising."""
        model = TinyModel()

        report = load_pretrained_weights(model, {"nonexistent": torch.zeros(1)}, strict=False)

        assert report.unexpected == ["nonexistent"]
        assert "unexpected" in report.summary()


class TestCheckpointLayouts:
    """The three layouts the released checkpoints ship in."""

    def test_reads_sharded_safetensors(self, tmp_path: Path) -> None:
        """Benchmark checkpoints ship as sharded safetensors with an index."""
        from safetensors.torch import save_file

        save_file({"sink.weight": torch.zeros(1, 4)}, str(tmp_path / "shard-1.safetensors"))
        save_file({"dit.layers.0.adaln_table": torch.zeros(6, 4)}, str(tmp_path / "shard-2.safetensors"))
        index = {
            "weight_map": {
                "sink.weight": "shard-1.safetensors",
                "dit.layers.0.adaln_table": "shard-2.safetensors",
            },
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

        state_dict = load_state_dict(tmp_path)

        assert set(state_dict) == {"sink.weight", "dit.layers.0.adaln_table"}

    def test_reads_deepspeed_checkpoint(self, tmp_path: Path) -> None:
        """The 5B base ships tensors under a module key."""
        path = tmp_path / "model_states.pt"
        torch.save({"module": {"module.sink.weight": torch.zeros(1, 4)}}, path)

        assert set(load_state_dict(path)) == {"sink.weight"}

    def test_missing_weights_raise(self, tmp_path: Path) -> None:
        """An empty directory is a clear error, not a silent empty load."""
        with pytest.raises(FileNotFoundError, match="No safetensors"):
            load_state_dict(tmp_path)

    def test_local_path_is_used_directly(self, tmp_path: Path) -> None:
        """An existing path is never treated as a Hub repo id."""
        assert resolve_checkpoint(tmp_path) == tmp_path


class TestLoadReport:
    """The report is what callers log."""

    def test_summary_counts(self) -> None:
        """Summary names all three counts."""
        summary = LoadReport(loaded=5, missing=["a"], unexpected=["b", "c"]).summary()

        assert "5 tensors" in summary
        assert "1 missing" in summary
        assert "2 unexpected" in summary
