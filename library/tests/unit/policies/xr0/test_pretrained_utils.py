# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for XR0 pretrained-weight loading (``pretrained_utils``).

Fast, self-contained tests with no external dependencies (no HuggingFace
downloads). A small ``XR0FlowModel`` provides realistic source-layout keys so
the remap is validated against the exact ``flow.*`` names the framework model
expects. File loading is exercised through ``.safetensors`` round-trips on
``tmp_path``.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from physicalai.policies.xr0.dit import XR0FlowModel
from physicalai.policies.xr0.preprocessor import make_xr0_preprocessors
from physicalai.policies.xr0.pretrained_utils import (
    extract_xr0_dataset_stats,
    load_xr0_pretrained_weights,
    remap_xr0_state_dict,
    resolve_pretrained_path,
)


def _source_layout_state_dict() -> dict[str, torch.Tensor]:
    """Build a source-``XR0``-layout state dict (flat, no ``flow.`` prefix)."""
    flow = XR0FlowModel(
        state_shape=(1, 8),
        action_shape=(4, 8),
        dit_num_layers=1,
        dit_hidden_size=128,
        dit_kv_heads=1,
        num_steps=2,
        dtype=torch.float32,
    )
    # Source keys are the flow submodules *without* the ``flow.`` prefix, plus
    # the VLM backbone under ``vlm.*`` and some recomputed buffers.
    source: dict[str, torch.Tensor] = dict(flow.state_dict())
    source["vlm.model.language_model.embed_tokens.weight"] = torch.randn(4, 8)
    source["vlm.lm_head.weight"] = torch.randn(4, 8)
    source["saved_causal_mask"] = torch.ones(1, 5, 5)
    source["rotary_emb.inv_freq"] = torch.randn(64)
    return source


def _expected_flow_keys() -> set[str]:
    """The ``flow.*`` keys the framework model expects for the flow submodules."""
    flow = XR0FlowModel(
        state_shape=(1, 8),
        action_shape=(4, 8),
        dit_num_layers=1,
        dit_hidden_size=128,
        dit_kv_heads=1,
        num_steps=2,
        dtype=torch.float32,
    )
    return {f"flow.{k}" for k in flow.state_dict()}


class TestRemap:
    """Key remapping from the source layout to the ``XR0Model`` namespace."""

    def test_flow_submodules_get_flow_prefix(self) -> None:
        """DiT / projector / embedder submodules are nested under ``flow.``."""
        remapped = remap_xr0_state_dict(_source_layout_state_dict())
        flow_keys = {k for k in remapped if k.startswith("flow.")}
        assert flow_keys == _expected_flow_keys()

    def test_vlm_keys_preserved(self) -> None:
        """VLM backbone keys are passed through unchanged (never ``flow.``)."""
        remapped = remap_xr0_state_dict(_source_layout_state_dict())
        assert "vlm.model.language_model.embed_tokens.weight" in remapped
        assert "vlm.lm_head.weight" in remapped
        assert not any(k.startswith("flow.vlm") for k in remapped)

    def test_recomputed_buffers_dropped(self) -> None:
        """Non-persistent / recomputed buffers are dropped."""
        remapped = remap_xr0_state_dict(_source_layout_state_dict())
        assert "saved_causal_mask" not in remapped
        assert not any(k.startswith("rotary_emb.") for k in remapped)

    def test_lm_head_recreated_when_tied(self) -> None:
        """A missing (weight-tied) ``vlm.lm_head.weight`` is recreated from embeddings."""
        source = _source_layout_state_dict()
        source.pop("vlm.lm_head.weight")
        remapped = remap_xr0_state_dict(source)
        assert "vlm.lm_head.weight" in remapped
        assert torch.equal(remapped["vlm.lm_head.weight"], remapped["vlm.model.language_model.embed_tokens.weight"])


    def test_strips_model_wrapper_prefix(self) -> None:
        """A leading ``model.`` runner prefix is stripped before remapping."""
        source = {f"model.{k}": v for k, v in _source_layout_state_dict().items()}
        remapped = remap_xr0_state_dict(source)
        assert remapped["flow.sink.weight"] is not None
        # ``vlm.model.*`` must not be corrupted by the ``model.`` strip.
        assert "vlm.model.language_model.embed_tokens.weight" in remapped

    def test_strips_module_wrapper_prefix(self) -> None:
        """A leading ``module.`` DeepSpeed prefix is stripped before remapping."""
        source = {f"module.{k}": v for k, v in _source_layout_state_dict().items()}
        remapped = remap_xr0_state_dict(source)
        assert {k for k in remapped if k.startswith("flow.")} == _expected_flow_keys()


class TestFileLoading:
    """Loading raw checkpoints from ``.safetensors`` files."""

    def test_load_safetensors(self, tmp_path) -> None:  # noqa: ANN001
        """A single ``model.safetensors`` directory checkpoint loads and remaps."""
        source = {k: v.contiguous() for k, v in _source_layout_state_dict().items()}
        save_file(source, str(tmp_path / "model.safetensors"))

        remapped = load_xr0_pretrained_weights(tmp_path)
        assert {k for k in remapped if k.startswith("flow.")} == _expected_flow_keys()
        assert "vlm.lm_head.weight" in remapped

    def test_resolve_local_path_is_identity(self, tmp_path) -> None:  # noqa: ANN001
        """An existing local path is returned as-is (no download)."""
        assert resolve_pretrained_path(tmp_path) == tmp_path


class TestLoadPretrainedWeights:
    """``load_xr0_pretrained_weights`` across every file / directory layout branch."""

    @staticmethod
    def _contiguous_source() -> dict[str, torch.Tensor]:
        return {k: v.contiguous() for k, v in _source_layout_state_dict().items()}

    def _assert_remapped(self, remapped: dict[str, torch.Tensor]) -> None:
        assert {k for k in remapped if k.startswith("flow.")} == _expected_flow_keys()
        assert "vlm.lm_head.weight" in remapped

    def test_single_safetensors_file(self, tmp_path) -> None:  # noqa: ANN001
        """A single ``*.safetensors`` *file* path (is_file + .safetensors branch)."""
        file = tmp_path / "xr0.safetensors"
        save_file(self._contiguous_source(), str(file))
        self._assert_remapped(load_xr0_pretrained_weights(file))

    def test_single_torch_file(self, tmp_path) -> None:  # noqa: ANN001
        """A single ``.pt`` file of raw tensors (is_file + torch-load branch)."""
        file = tmp_path / "xr0_pretrained.pt"
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(self._contiguous_source(), str(file))  # test fixture: tensor-only state dict
        self._assert_remapped(load_xr0_pretrained_weights(file))

    def test_deepspeed_container_file(self, tmp_path) -> None:  # noqa: ANN001
        """A ``{"module": ...}`` DeepSpeed-wrapped file is unwrapped."""
        file = tmp_path / "xr0.pt"
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save({"module": self._contiguous_source()}, str(file))  # test fixture: tensor-only state dict
        self._assert_remapped(load_xr0_pretrained_weights(file))

    def test_lightning_container_file(self, tmp_path) -> None:  # noqa: ANN001
        """A ``{"state_dict": ...}`` Lightning-wrapped file is unwrapped."""
        file = tmp_path / "xr0.ckpt"
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save({"state_dict": self._contiguous_source()}, str(file))  # test fixture: tensor-only state dict
        self._assert_remapped(load_xr0_pretrained_weights(file))

    def test_sharded_safetensors_dir(self, tmp_path) -> None:  # noqa: ANN001
        """A sharded ``model.safetensors.index.json`` directory checkpoint."""
        source = self._contiguous_source()
        keys = list(source)
        half = len(keys) // 2
        shards = {"a.safetensors": keys[:half], "b.safetensors": keys[half:]}
        for shard, shard_keys in shards.items():
            save_file({k: source[k] for k in shard_keys}, str(tmp_path / shard))
        weight_map = {k: shard for shard, shard_keys in shards.items() for k in shard_keys}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
        self._assert_remapped(load_xr0_pretrained_weights(tmp_path))

    def test_deepspeed_dir(self, tmp_path) -> None:  # noqa: ANN001
        """A DeepSpeed ``last.ckpt/checkpoint/mp_rank_00_model_states.pt`` directory."""
        ckpt = tmp_path / "last.ckpt" / "checkpoint"
        ckpt.mkdir(parents=True)
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save({"module": self._contiguous_source()}, str(ckpt / "mp_rank_00_model_states.pt"))  # test fixture
        self._assert_remapped(load_xr0_pretrained_weights(tmp_path))

    def test_sharded_bin_dir(self, tmp_path) -> None:  # noqa: ANN001
        """A sharded ``pytorch_model.bin.index.json`` directory checkpoint."""
        source = self._contiguous_source()
        keys = list(source)
        half = len(keys) // 2
        shards = {"pytorch_model-00001.bin": keys[:half], "pytorch_model-00002.bin": keys[half:]}
        for shard, shard_keys in shards.items():
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            torch.save({k: source[k] for k in shard_keys}, str(tmp_path / shard))  # test fixture: tensor-only shard
        weight_map = {k: shard for shard, shard_keys in shards.items() for k in shard_keys}
        (tmp_path / "pytorch_model.bin.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
        self._assert_remapped(load_xr0_pretrained_weights(tmp_path))

    def test_single_bin_candidate_dir(self, tmp_path) -> None:  # noqa: ANN001
        """A single ``pytorch_model.bin`` directory checkpoint."""
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(self._contiguous_source(), str(tmp_path / "pytorch_model.bin"))  # test fixture: tensor-only
        self._assert_remapped(load_xr0_pretrained_weights(tmp_path))

    def test_missing_weights_raises(self, tmp_path) -> None:  # noqa: ANN001
        """An empty directory (no recognized weights) raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError, match="No recognized XR0 weights"):
            load_xr0_pretrained_weights(tmp_path)



def _write_preprocessor_config(tmp_path, mean_row, std_row, *, time_steps=10) -> None:  # noqa: ANN001
    """Write a LIBERO-style ``preprocessor_config.json`` with time-invariant stats."""
    config = {
        "action_config": {
            "libero_all": {
                "mean": [list(mean_row) for _ in range(time_steps)],
                "std": [list(std_row) for _ in range(time_steps)],
            },
        },
    }
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(config), encoding="utf-8")


class TestExtractStats:
    """Extraction of action-normalization stats from the processor config."""

    def test_reduces_time_and_trims_padding(self, tmp_path) -> None:  # noqa: ANN001
        """Time-invariant (T, 32) stats collapse to the active leading dims."""
        mean = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -0.5] + [0.0] * 25
        std = [0.3, 0.4, 0.44, 0.04, 0.06, 0.08, 0.99] + [1e-6] * 25
        _write_preprocessor_config(tmp_path, mean, std)

        stats = extract_xr0_dataset_stats(tmp_path)
        assert stats is not None
        action = stats["action"]
        assert action["shape"] == (7,)  # padding dims (std <= 1e-5) trimmed
        assert action["mean"] == mean[:7]
        assert action["std"] == std[:7]

    def test_returns_none_without_action_config(self, tmp_path) -> None:  # noqa: ANN001
        """Missing ``action_config`` / config yields ``None`` (lazy fallback)."""
        (tmp_path / "preprocessor_config.json").write_text(json.dumps({"foo": 1}), encoding="utf-8")
        assert extract_xr0_dataset_stats(tmp_path) is None
        assert extract_xr0_dataset_stats(tmp_path / "missing") is None

    def test_flat_stats_pass_through(self, tmp_path) -> None:  # noqa: ANN001
        """Already time-reduced 1D stats are used as-is (no reshape)."""
        mean = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -0.5] + [0.0] * 25
        std = [0.3, 0.4, 0.44, 0.04, 0.06, 0.08, 0.99] + [1e-6] * 25
        config = {"action_config": {"libero_all": {"mean": mean, "std": std}}}
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(config), encoding="utf-8")

        stats = extract_xr0_dataset_stats(tmp_path)
        assert stats is not None
        action = stats["action"]
        assert action["shape"] == (7,)
        assert action["mean"] == mean[:7]
        assert action["std"] == std[:7]

    def test_stats_drive_postprocessor_action_dim(self, tmp_path) -> None:  # noqa: ANN001
        """Extracted stats make the postprocessor emit the true action size."""
        mean = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -0.5] + [0.0] * 25
        std = [0.3, 0.4, 0.44, 0.04, 0.06, 0.08, 0.99] + [1e-6] * 25
        _write_preprocessor_config(tmp_path, mean, std)

        stats = extract_xr0_dataset_stats(tmp_path)
        _, postprocessor = make_xr0_preprocessors(max_action_dim=32, stats=stats)
        out = postprocessor({"action": torch.zeros(1, 4, 32)})
        assert out["action"].shape == (1, 4, 7)
