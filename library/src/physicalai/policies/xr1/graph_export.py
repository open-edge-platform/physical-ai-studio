# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Graph export for XR1's action expert.

Why this is a component and not the whole policy
------------------------------------------------

XR1 is a Qwen3-VL backbone plus a DiT action expert. The action expert is our own
code and traces cleanly; the backbone is ``transformers``' Qwen3-VL, and with images
present it cannot be captured by either exporter in the supported ``transformers``
line (>=5.5,<5.6):

* ``torch.export`` (what ``torch.onnx.export(dynamo=True)`` uses) raises
  ``GuardOnDataDependentSymNode`` inside ``Qwen3VLVisionModel``: the vision tower
  splits its output using values read out of ``image_grid_thw``
  (``grid_thw_list = image_grid_thw.tolist()``), so the split sizes are unbacked
  symbols. Freezing the grid as a constant does not help, because the read happens
  on a tensor.
* ``torch.jit.trace`` (what OpenVINO's PyTorch frontend uses) raises ``IndexError:
  tuple index out of range`` at ``transformers/masking_utils.py:492``, where
  ``sdpa_mask`` does ``q_length.shape[0]`` on what tracing has turned into a 0-d
  tensor.

Both are upstream limitations rather than porting mistakes, and both are reproduced
in ``tests/unit/policies/xr1/test_graph_export.py``. A text-and-state-only graph does
export, because the vision tower is then never traced - but a vision-language-action
policy that cannot see is not the model, so this module does not offer it.

What *is* exportable is the part that dominates inference cost: the action expert is
evaluated ``num_inference_steps`` times per chunk, while the backbone runs once.
:class:`XR1ActionExpert` wraps those iterations - the whole Euler loop, not a single
step - into one graph whose inputs are plain tensors, including the backbone's
key/value cache. The backbone stays on the Torch path and feeds it.

Example:
    >>> from physicalai.policies.xr1.graph_export import build_action_expert_inputs, export_action_expert
    >>> inputs = build_action_expert_inputs(policy.model, processed_batch)
    >>> export_action_expert(policy.model, "./export-expert", inputs=inputs)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from physicalai.inference.postprocessors.base import Postprocessor
from torch import nn

from physicalai.data.observation import ACTION

if TYPE_CHECKING:
    from physicalai.policies.xr1.vla import XR1Model

#: Basename of the exported action-expert artifact.
ARTIFACT_STEM = "xr1_action_expert"

#: Inputs that are always present, in graph order, before the cache tensors.
BASE_INPUT_NAMES = ("noise", "state_embed", "cos", "sin", "attn_mask")
OUTPUT_NAME = "action"


@dataclass
class ActionExpertInputs:
    """Tensors the exported action-expert graph consumes.

    Attributes:
        noise: Starting sample of shape ``(batch, chunk_size, max_action_dim)``.
        state_embed: Projected state of shape ``(batch, state_len, dit_hidden_size)``.
        cos: Rotary cosines for the query sequence.
        sin: Rotary sines for the query sequence.
        attn_mask: Attention mask over the cache and the query, as float so it
            survives ONNX and OpenVINO conversion unchanged.
        cache: Flattened backbone cache, ``key`` and ``value`` per layer in order.
    """

    noise: torch.Tensor
    state_embed: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    attn_mask: torch.Tensor
    cache: list[torch.Tensor] = field(default_factory=list)

    def as_args(self) -> tuple[torch.Tensor, ...]:
        """Return the tensors in graph-input order.

        Returns:
            Positional arguments for :meth:`XR1ActionExpert.forward`.
        """
        return (self.noise, self.state_embed, self.cos, self.sin, self.attn_mask, *self.cache)

    @property
    def names(self) -> list[str]:
        """Graph input names, in the same order as :meth:`as_args`.

        Returns:
            One name per input tensor.
        """
        layers = len(self.cache) // 2
        cache_names = [f"{kind}_{index}" for index in range(layers) for kind in ("key", "value")]
        return [*BASE_INPUT_NAMES, *cache_names]

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the tensors keyed by graph input name.

        Returns:
            Mapping from input name to tensor.
        """
        return dict(zip(self.names, self.as_args(), strict=True))


class XR1ActionExpert(nn.Module):
    """The DiT action expert and its Euler sampler, as one traceable module.

    The loop is unrolled at trace time, which is what makes the graph static: the
    step count is a configuration value, not data.
    """

    def __init__(self, model: XR1Model) -> None:
        """Wrap a model's action expert.

        Args:
            model: The XR1 model whose DiT, projectors and flow settings are used.
        """
        super().__init__()
        self.model = model
        self.num_layers = model.config.dit_num_layers
        self.num_inference_steps = model.config.num_inference_steps

    def forward(
        self,
        noise: torch.Tensor,
        state_embed: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        *cache: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate the velocity field from ``noise`` to an action chunk.

        Args:
            noise: Starting sample of shape ``(batch, chunk_size, max_action_dim)``.
            state_embed: Projected state.
            cos: Rotary cosines for the query sequence.
            sin: Rotary sines for the query sequence.
            attn_mask: Attention mask over cache and query, as float.
            *cache: Backbone cache, ``key`` then ``value`` for each layer.

        Returns:
            The integrated action chunk, same shape as ``noise``.
        """
        pairs = [(cache[2 * index], cache[2 * index + 1]) for index in range(self.num_layers)]
        mask = attn_mask.to(torch.bool)
        # Inference never masks action dimensions; the padded slots are trimmed by the
        # postprocessor instead, so this stays a constant rather than an input.
        action_mask = torch.ones_like(noise)

        sample = noise
        step_size = 1.0 / self.num_inference_steps
        for index in range(self.num_inference_steps):
            timestep = torch.full((sample.shape[0], 1, 1), index * step_size, dtype=sample.dtype)
            velocity = self.model.dit_forward(
                sample,
                timestep,
                action_mask,
                state_embed,
                (cos, sin),
                pairs,
                mask,
            )
            sample = sample + velocity * step_size  # noqa: PLR6104 - in-place would alias the graph input
        return sample


@torch.no_grad()
def build_action_expert_inputs(
    model: XR1Model,
    batch: dict[str, Any],
    noise: torch.Tensor | None = None,
) -> ActionExpertInputs:
    """Run the backbone once and assemble the action expert's tensor inputs.

    This reproduces exactly what :meth:`XR1Model.predict_action_chunk` assembles, so a
    parity check against the Torch path compares the same computation rather than an
    approximation of it.

    Args:
        model: The XR1 model.
        batch: Preprocessed batch, as :class:`~physicalai.policies.xr1.preprocessor.XR1Preprocessor`
            produces it.
        noise: Optional starting sample. Random when omitted; pass it explicitly to
            make a comparison against the Torch path deterministic.

    Returns:
        The assembled inputs.
    """
    state = batch["state"].to(model.dtype)
    batch_size = state.shape[0]
    placeholder = torch.zeros(
        (batch_size, model.config.chunk_size, model.config.max_action_dim),
        device=state.device,
        dtype=model.dtype,
    )

    vlm_outputs = model.encode_prompt(batch)
    dit_kwargs, placeholder, _ = model._prepare_dit_inputs(  # noqa: SLF001 - the graph must match this assembly exactly
        batch,
        vlm_outputs,
        placeholder,
        torch.ones_like(placeholder),
        0,
    )

    cos, sin = dit_kwargs["position_embeds"]
    cache: list[torch.Tensor] = []
    for key, value in dit_kwargs["past_key_values"]:
        cache.extend((key.detach(), value.detach()))

    return ActionExpertInputs(
        noise=torch.randn_like(placeholder) if noise is None else noise.to(placeholder.dtype),
        state_embed=dit_kwargs["state_embed"].detach(),
        cos=cos.detach(),
        sin=sin.detach(),
        attn_mask=dit_kwargs["attn_mask"].to(model.dtype),
        cache=cache,
    )


@torch.no_grad()
def export_action_expert(
    model: XR1Model,
    output_dir: Path | str,
    inputs: ActionExpertInputs,
    *,
    backend: str = "openvino",
    compress_to_fp16: bool = False,
) -> Path:
    """Export the action expert to ONNX, and optionally on to OpenVINO.

    Args:
        model: The XR1 model.
        output_dir: Directory for the artifacts.
        inputs: Tracing inputs from :func:`build_action_expert_inputs`.
        backend: ``"onnx"`` or ``"openvino"``. OpenVINO is produced by converting the
            ONNX graph, because OpenVINO's PyTorch frontend traces with
            ``torch.jit.trace``, which the backbone's masking utilities break.
        compress_to_fp16: Store OpenVINO weights as fp16.

    Returns:
        Path to the written model file.

    Raises:
        ValueError: If ``backend`` is not one this component supports.
    """
    if backend not in {"onnx", "openvino"}:
        msg = f"The action expert supports 'onnx' and 'openvino', got {backend!r}"
        raise ValueError(msg)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    expert = XR1ActionExpert(model).eval()

    def write_onnx(path: Path) -> None:
        torch.onnx.export(
            expert,
            inputs.as_args(),
            f=str(path),
            input_names=inputs.names,
            output_names=[OUTPUT_NAME],
            dynamo=True,
        )

    if backend == "onnx":
        onnx_path = directory / f"{ARTIFACT_STEM}.onnx"
        write_onnx(onnx_path)
        return onnx_path

    import openvino  # noqa: PLC0415  # optional at import time

    # The ONNX graph is scaffolding, and large models come with an external-data
    # sidecar next to it. Building it in a temporary directory keeps both out of the
    # export, which otherwise ships tens of megabytes nothing reads.
    with tempfile.TemporaryDirectory() as staging:
        onnx_path = Path(staging) / f"{ARTIFACT_STEM}.onnx"
        write_onnx(onnx_path)
        ov_model = openvino.convert_model(str(onnx_path))

    xml_path = directory / f"{ARTIFACT_STEM}.xml"
    openvino.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)
    return xml_path


class XR1BackboneStage(nn.Module):
    """First stage of the hybrid export: an observation in, expert tensors out.

    Runtime's Torch adapter reconstructs whatever ``manifest.policy.source.class_path``
    names, calls it with an :class:`~physicalai.data.observation.Observation` and
    converts the result to numpy. Naming this class there gives the two-stage runner a
    first stage that stops after the backbone, without changing what :class:`XR1`
    itself does.

    Preprocessing stays here rather than moving into Runtime: the Qwen3-VL chat
    template, image letterboxing and tokenizer all live on the Studio side already,
    and Runtime ships no Qwen3-VL preprocessing component.
    """

    def __init__(self, policy: Any) -> None:  # noqa: ANN401 - an XR1, typed loosely to avoid a circular import
        """Wrap a policy as the hybrid export's first stage.

        Args:
            policy: The :class:`~physicalai.policies.xr1.policy.XR1` policy.
        """
        super().__init__()
        self.policy = policy

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        map_location: str = "cpu",
        **kwargs: Any,  # noqa: ANN401 - forwarded to the policy loader
    ) -> XR1BackboneStage:
        """Load the policy from a checkpoint and wrap it.

        Args:
            checkpoint_path: Path to the ``.pt`` written by ``to_torch``.
            map_location: Device to map storages onto.
            **kwargs: Forwarded to :meth:`XR1.load_from_checkpoint`.

        Returns:
            The wrapped first stage, in eval mode.
        """
        from physicalai.policies.xr1.policy import XR1  # noqa: PLC0415  # circular at module import time

        kwargs.setdefault("weights_only", False)
        policy = XR1.load_from_checkpoint(checkpoint_path, map_location=map_location, **kwargs)
        return cls(policy.eval())

    @property
    def extra_export_args(self) -> dict[str, Any]:
        """Export parameters, read by Runtime's Torch adapter for output names.

        Returns:
            The policy's own export parameters.
        """
        return dict(self.policy.extra_export_args)

    @torch.no_grad()
    def forward(self, observation: Any) -> dict[str, torch.Tensor]:  # noqa: ANN401 - an Observation
        """Run preprocessing and the backbone, and emit the expert's inputs.

        Args:
            observation: Observation batch.

        Returns:
            The action expert's graph inputs, keyed by graph input name.
        """
        batch = self.policy._preprocessor(observation.to(self.policy.device).to_dict())  # noqa: SLF001 - the stage is the policy's own preprocessing
        return build_action_expert_inputs(self.policy.model, batch).as_dict()


class XR1ActionTrimmer(Postprocessor):
    """Trim the padded action slots off the exported graph's output.

    The action expert predicts ``max_action_dim`` slots because the architecture is
    sized by config, not by dataset. Everything past the dataset's own width is
    padding and must be dropped before denormalization, which carries per-dimension
    statistics.
    """

    def __init__(self, action_dim: int, key: str = "action") -> None:
        """Configure the trimmer.

        Args:
            action_dim: The dataset's action width.
            key: Output key to trim.
        """
        self.action_dim = action_dim
        self.key = key

    def __call__(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Trim the action tensor.

        Args:
            outputs: Runner outputs.

        Returns:
            The same dict with the action trimmed to the dataset width.
        """
        action = outputs.get(self.key)
        if action is None:
            return outputs
        return {**outputs, self.key: action[..., : self.action_dim]}

    def __repr__(self) -> str:
        """Return string representation of the postprocessor.

        Returns:
            A representation naming the width kept.
        """
        return f"{self.__class__.__name__}(action_dim={self.action_dim})"


def export_hybrid(policy: Any, output_dir: Path | str, *, compress_to_fp16: bool = False) -> Path:  # noqa: ANN401 - an XR1
    """Write a hybrid export: Torch backbone plus an OpenVINO action expert.

    The result is a normal export directory that Runtime's ``InferenceModel`` loads
    unchanged. Its manifest declares two artifacts and a ``two_stage`` runner, so the
    backbone runs through the Torch adapter and the action expert through the
    OpenVINO one.

    Args:
        policy: An initialized :class:`~physicalai.policies.xr1.policy.XR1`.
        output_dir: Directory for the artifacts and the manifest.
        compress_to_fp16: Store the OpenVINO weights as fp16.

    Returns:
        The export directory.

    Raises:
        ValueError: If the policy has not been initialized with dataset statistics,
            which the manifest needs for its feature schema and denormalization.
    """
    from physicalai.inference.manifest import (  # noqa: PLC0415  # heavy import, deferred
        ComponentSpec,
        Manifest,
        ModelSpec,
        PolicySource,
        PolicySpec,
    )

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    sample = policy._get_default_export_input_sample()  # noqa: SLF001 - the mixin's own tracing sample
    if sample is None:
        msg = "Dataset stats are required for a hybrid export. Initialize the policy with dataset_stats."
        raise ValueError(msg)

    inputs = build_action_expert_inputs(policy.model, sample)
    expert_path = export_action_expert(
        policy.model,
        directory,
        inputs,
        backend="openvino",
        compress_to_fp16=compress_to_fp16,
    )

    # Reuse the tested checkpoint writer, then replace the manifest it leaves behind:
    # the artifact is the same, only the way it is driven differs.
    policy.to_torch(directory)

    action_stats = policy._dataset_stats[ACTION]  # noqa: SLF001 - the mixin exposes no public accessor
    action_dim = int(cast("tuple", action_stats["shape"])[-1])
    stage = XR1BackboneStage
    manifest = Manifest(
        policy=PolicySpec(
            name=policy.__class__.__name__.lower(),
            source=PolicySource(class_path=f"{stage.__module__}.{stage.__qualname__}"),
        ),
        model=ModelSpec(
            # class_path mode on purpose: in type mode every extra field is passed to
            # the constructor, and chunk_size is manifest metadata that Runtime reads
            # off the spec rather than a runner argument.
            runner=ComponentSpec(
                class_path="physicalai.inference.runners.TwoStage",
                init_args={"backend": "openvino", "artifact": expert_path.name},
                chunk_size=policy.config.chunk_size,
            ),
            artifacts={"torch": f"{policy.__class__.__name__.lower()}.pt", "openvino": expert_path.name},
            preprocessors=[ComponentSpec(type="to_float_tensor")],
            postprocessors=[
                ComponentSpec(
                    class_path=f"{XR1ActionTrimmer.__module__}.{XR1ActionTrimmer.__qualname__}",
                    init_args={"action_dim": action_dim},
                ),
                ComponentSpec(
                    type="denormalize",
                    stats={ACTION: action_stats},
                    mode=policy.config.normalization_mode.lower(),
                ),
            ],
            input_features=policy._to_component_specs(policy.inputs_schema or []),  # noqa: SLF001 - mixin helper
            output_features=policy._to_component_specs(policy.outputs_schema or []),  # noqa: SLF001 - mixin helper
        ),
    )
    manifest.save(directory / "manifest.json")
    return directory
