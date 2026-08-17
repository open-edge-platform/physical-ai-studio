# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for exporting XR1's action expert to ONNX and OpenVINO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from physicalai.policies.xr1 import XR1Config, XR1Model
from physicalai.policies.xr1.graph_export import (
    BASE_INPUT_NAMES,
    ActionExpertInputs,
    XR1ActionExpert,
    build_action_expert_inputs,
    export_action_expert,
)

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.policies.xr1.vlm import XR1Qwen3VL

BATCH = 1
#: float32 round-off through two exporters; anything larger is a real divergence.
PARITY_TOLERANCE = 1e-5


@pytest.fixture
def model(tiny_config: XR1Config, tiny_vlm: XR1Qwen3VL) -> XR1Model:
    """Build the tiny model in eval mode.

    Args:
        tiny_config: Small configuration.
        tiny_vlm: Small randomly initialized backbone.

    Returns:
        The model, ready for inference.
    """
    return XR1Model(tiny_config, vlm=tiny_vlm).eval()


@pytest.fixture
def prompt_batch() -> dict[str, torch.Tensor]:
    """Return a text-and-state batch in the model's input format.

    Returns:
        Batch with token ids, attention mask and state.
    """
    return {
        "input_ids": torch.randint(0, 150, (BATCH, 9)),
        "attention_mask": torch.ones(BATCH, 9, dtype=torch.long),
        "state": torch.randn(BATCH, 1, 8),
    }


@pytest.fixture
def expert_inputs(model: XR1Model, prompt_batch: dict[str, torch.Tensor]) -> ActionExpertInputs:
    """Assemble the action expert's inputs from one backbone pass.

    Args:
        model: The model.
        prompt_batch: Preprocessed batch.

    Returns:
        Tracing inputs.
    """
    return build_action_expert_inputs(model, prompt_batch)


class TestActionExpertInputs:
    """The tensor contract between the backbone and the exported graph."""

    def test_names_match_the_arguments(self, expert_inputs: ActionExpertInputs) -> None:
        """Input names are what the exported graph is keyed by, so they must line up."""
        assert len(expert_inputs.names) == len(expert_inputs.as_args())
        assert expert_inputs.names[: len(BASE_INPUT_NAMES)] == list(BASE_INPUT_NAMES)

    def test_carries_one_key_and_value_per_layer(
        self,
        model: XR1Model,
        expert_inputs: ActionExpertInputs,
    ) -> None:
        """Each DiT layer attends over one cached backbone layer."""
        assert len(expert_inputs.cache) == 2 * model.config.dit_num_layers
        assert expert_inputs.names[-1] == f"value_{model.config.dit_num_layers - 1}"

    def test_shapes_follow_the_config(self, model: XR1Model, expert_inputs: ActionExpertInputs) -> None:
        """The graph is static, so every shape is a configuration consequence."""
        assert expert_inputs.noise.shape == (BATCH, model.config.chunk_size, model.config.max_action_dim)
        assert expert_inputs.state_embed.shape == (BATCH, model.config.state_len, model.config.dit_hidden_size)

    def test_as_dict_round_trips(self, expert_inputs: ActionExpertInputs) -> None:
        """The dict form is what an inference session is fed."""
        as_dict = expert_inputs.as_dict()

        assert set(as_dict) == set(expert_inputs.names)
        assert torch.equal(as_dict["noise"], expert_inputs.noise)


class TestActionExpertModule:
    """The traceable wrapper must be the same computation as the policy path."""

    def test_matches_predict_action_chunk(
        self,
        model: XR1Model,
        prompt_batch: dict[str, torch.Tensor],
    ) -> None:
        """The wrapper must be the policy's own sampler, not a re-derivation.

        Both paths draw their starting noise with one ``randn_like`` after the same
        preceding operations, so pinning the RNG state before each gives them the same
        noise and the outputs must then be identical.
        """
        expert = XR1ActionExpert(model).eval()
        rng_state = torch.get_rng_state()

        inputs = build_action_expert_inputs(model, prompt_batch)
        with torch.no_grad():
            wrapped = expert(*inputs.as_args())

        torch.set_rng_state(rng_state)
        with torch.no_grad():
            direct = model.predict_action_chunk(prompt_batch)

        assert torch.equal(wrapped, direct)

    def test_is_deterministic_for_fixed_noise(
        self,
        model: XR1Model,
        prompt_batch: dict[str, torch.Tensor],
    ) -> None:
        """Everything after the noise draw is deterministic."""
        noise = torch.zeros(BATCH, model.config.chunk_size, model.config.max_action_dim)
        expert = XR1ActionExpert(model).eval()
        inputs = build_action_expert_inputs(model, prompt_batch, noise=noise)

        with torch.no_grad():
            first = expert(*inputs.as_args())
            second = expert(*inputs.as_args())

        assert torch.equal(first, second)


@pytest.mark.slow
class TestGraphExport:
    """Export and numerical parity, the two checks the export skill requires."""

    @staticmethod
    def _reference(model: XR1Model, inputs: ActionExpertInputs) -> torch.Tensor:
        """Run the Torch path on the same inputs the graph will receive.

        Args:
            model: The model.
            inputs: Graph inputs.

        Returns:
            The Torch action chunk.
        """
        with torch.no_grad():
            return XR1ActionExpert(model).eval()(*inputs.as_args())

    def test_onnx_export_and_parity(
        self,
        model: XR1Model,
        expert_inputs: ActionExpertInputs,
        tmp_path: Path,
    ) -> None:
        """The ONNX graph must reproduce the Torch path within float32 round-off."""
        onnxruntime = pytest.importorskip("onnxruntime")

        path = export_action_expert(model, tmp_path, expert_inputs, backend="onnx")
        session = onnxruntime.InferenceSession(str(path))
        outputs = session.run(None, {name: tensor.numpy() for name, tensor in expert_inputs.as_dict().items()})

        expected = self._reference(model, expert_inputs).numpy()
        assert outputs[0].shape == expected.shape
        np.testing.assert_allclose(outputs[0], expected, rtol=PARITY_TOLERANCE, atol=PARITY_TOLERANCE)

    def test_openvino_export_and_parity(
        self,
        model: XR1Model,
        expert_inputs: ActionExpertInputs,
        tmp_path: Path,
    ) -> None:
        """The OpenVINO graph must reproduce the Torch path within float32 round-off."""
        openvino = pytest.importorskip("openvino")

        path = export_action_expert(model, tmp_path, expert_inputs, backend="openvino")
        assert path.suffix == ".xml"
        assert path.with_suffix(".bin").is_file()

        compiled = openvino.Core().compile_model(openvino.Core().read_model(str(path)), "CPU")
        result = compiled({name: tensor.numpy() for name, tensor in expert_inputs.as_dict().items()})

        expected = self._reference(model, expert_inputs).numpy()
        actual = next(iter(result.values()))
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=PARITY_TOLERANCE, atol=PARITY_TOLERANCE)

    def test_rejects_an_unsupported_backend(
        self,
        model: XR1Model,
        expert_inputs: ActionExpertInputs,
        tmp_path: Path,
    ) -> None:
        """Only the two graph backends make sense for a component export."""
        with pytest.raises(ValueError, match="'onnx' and 'openvino'"):
            export_action_expert(model, tmp_path, expert_inputs, backend="executorch")


@pytest.mark.slow
class TestBackboneExportBlockers:
    """Why the whole policy is not offered as a graph.

    These are upstream ``transformers`` limitations, not porting mistakes. Pinning
    them down means a future ``transformers`` bump can be checked by running these:
    an xfail that starts passing is the signal to widen the export surface.
    """

    @staticmethod
    def _image_batch(model: XR1Model) -> dict[str, Any]:
        """Build a batch that exercises the vision tower.

        Args:
            model: The model, read for its vision geometry.

        Returns:
            A batch with one image.
        """
        vision = model.vlm.config.vision_config
        grid = (1, vision.spatial_merge_size, vision.spatial_merge_size)
        merged = grid[0] * grid[1] * grid[2] // vision.spatial_merge_size**2
        image_token_id = model.vlm.config.image_token_id

        input_ids = torch.randint(0, image_token_id, (BATCH, 9))
        mm_token_type_ids = torch.zeros(BATCH, 9, dtype=torch.int32)
        input_ids[:, 1 : 1 + merged] = image_token_id
        mm_token_type_ids[:, 1 : 1 + merged] = 1
        patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(BATCH, 9, dtype=torch.long),
            "pixel_values": torch.randn(grid[1] * grid[2], patch_dim),
            "image_grid_thw": torch.tensor([grid], dtype=torch.long),
            "mm_token_type_ids": mm_token_type_ids,
            "state": torch.randn(BATCH, 1, 8),
        }

    def test_text_only_graph_does_export(self, model: XR1Model, prompt_batch: dict[str, Any], tmp_path: Path) -> None:
        """Without images the whole policy traces, which isolates the blocker."""
        torch.onnx.export(
            model,
            args=(),
            kwargs={"batch": prompt_batch},
            f=str(tmp_path / "text_only.onnx"),
            input_names=list(prompt_batch),
            dynamo=True,
        )

        assert (tmp_path / "text_only.onnx").is_file()

    def test_vision_tower_breaks_torch_export(self, model: XR1Model, tmp_path: Path) -> None:
        """``torch.export`` cannot resolve the vision tower's data-dependent split."""
        batch = self._image_batch(model)

        with pytest.raises(Exception, match="data-dependent|GuardOnDataDependent"):
            torch.onnx.export(
                model,
                args=(),
                kwargs={"batch": batch},
                f=str(tmp_path / "with_images.onnx"),
                input_names=list(batch),
                dynamo=True,
            )

    def test_masking_utils_break_torchscript(self, model: XR1Model) -> None:
        """``torch.jit.trace`` - OpenVINO's PyTorch frontend - breaks earlier still.

        ``transformers/masking_utils.py`` reads ``q_length.shape[0]`` on a value that
        tracing has turned into a 0-d tensor.
        """

        class Wrapper(torch.nn.Module):
            def __init__(self, inner: XR1Model, batch: dict[str, Any]) -> None:
                super().__init__()
                self.inner = inner
                self.batch = batch

            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return self.inner({**self.batch, "state": state})

        batch = self._image_batch(model)
        wrapper = Wrapper(model, batch).eval()

        with pytest.raises(IndexError, match="tuple index out of range"):
            torch.jit.trace(wrapper, (batch["state"],), strict=False, check_trace=False)
