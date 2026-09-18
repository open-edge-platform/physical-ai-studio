# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runnable smoke test for the native policy design.

Run with: python -m docs.explanation.policy.template.demo (from the library root)
"""

from __future__ import annotations

from dataclasses import replace
from tempfile import TemporaryDirectory
from typing import cast

import torch
from torch import Tensor

from physicalai.data import DataModule, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK
from physicalai.export.backends import (
    ExecuTorchExportParameters,
    ONNXExportParameters,
    OpenVINOExportParameters,
    TorchExportParameters,
)
from physicalai.inference.data import InferenceFeatureDtype, InferenceFeatureType
from physicalai.policies.mixins.peft import is_lora_injected
from physicalai.train import Trainer

from .config import NewPolicyModelConfig
from .model import NewPolicyModel
from .policy import NewPolicy


class _FakePolicyDataset(Dataset):
    def __init__(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
        *,
        size: int,
        sequence_length: int,
        chunk_size: int,
        action_dim: int,
        vocab_size: int,
        state_dim: int,
        image_size: tuple[int, int],
    ) -> None:
        self._input_features = {cast("str", feature.name): feature for feature in input_features}
        self._output_features = {cast("str", feature.name): feature for feature in output_features}
        self._size = size
        self._sequence_length = sequence_length
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self._vocab_size = vocab_size
        self._state_dim = state_dim
        self._image_size = image_size
        self._delta_indices: dict[str, list[int]] = {}

    def __getitem__(self, index: int) -> Observation:
        generator = torch.Generator().manual_seed(index)
        return Observation(
            task=torch.randint(self._vocab_size, (self._sequence_length,), generator=generator),
            state=torch.randn(self._state_dim, generator=generator),
            images={
                "front": torch.randn(3, *self._image_size, generator=generator),
            },
            action=torch.randn(self._chunk_size, self._action_dim, generator=generator),
        )

    def __len__(self) -> int:
        return self._size

    @property
    def raw_features(self) -> dict[str, Feature]:
        return {**self._input_features, **self._output_features}

    @property
    def observation_features(self) -> dict[str, Feature]:
        return self._input_features

    @property
    def action_features(self) -> dict[str, Feature]:
        return self._output_features

    @property
    def fps(self) -> int:
        return 30

    @property
    def tolerance_s(self) -> float:
        return 1e-4

    @property
    def delta_indices(self) -> dict[str, list[int]]:
        return self._delta_indices

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        self._delta_indices = indices


if __name__ == "__main__":
    torch.manual_seed(7)
    sequence_length = 6
    chunk_size = 3
    action_dim = 2
    vocab_size = 32
    state_dim = 4
    image_size = (16, 16)

    input_features = [
        Feature(name=STATE, shape=(state_dim,), ftype=FeatureType.STATE),
        Feature(name="front", shape=(3, *image_size), ftype=FeatureType.VISUAL),
    ]
    output_features = [
        Feature(
            name="action",
            shape=(action_dim,),
            ftype=FeatureType.ACTION,
            normalization_data=NormalizationParameters(
                mean=[0.25, -0.25],
                std=[0.5, 2.0],
            ),
        ),
    ]

    tiny_config = NewPolicyModelConfig(
        input_features=input_features,
        output_features=output_features,
        action_dim=action_dim,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        attention_dropout=0.2,
        layer_norm_eps=1e-4,
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        n_action_steps=2,
        image_size=image_size,
        tokenizer_max_length=sequence_length,
    )

    dataset = _FakePolicyDataset(
        input_features,
        output_features,
        size=8,
        sequence_length=sequence_length,
        chunk_size=chunk_size,
        action_dim=action_dim,
        vocab_size=vocab_size,
        state_dim=state_dim,
        image_size=image_size,
    )
    datamodule = DataModule(
        train_dataset=dataset,
        train_batch_size=2,
        num_workers=0,
    )
    policy = NewPolicy.from_config(tiny_config, optimizer_lr=1e-3)

    batch = next(iter(datamodule.train_dataloader()))
    policy.eval()
    prediction = cast("Tensor", policy(batch))
    assert isinstance(policy.model, NewPolicyModel)
    model_prediction = policy.model.predict_action_chunk(policy._prepare_batch(batch, require_actions=False))
    assert model_prediction.shape == (2, chunk_size, action_dim)
    assert prediction.shape == (2, tiny_config.n_action_steps, action_dim)
    print(f"Forward pass: shape={tuple(prediction.shape)}")

    inputs_schema = policy.inputs_schema
    outputs_schema = policy.outputs_schema
    assert inputs_schema is not None and [feature.name for feature in inputs_schema] == [STATE, IMAGES, TASK]
    assert inputs_schema[-1].ftype == InferenceFeatureType.LANGUAGE
    assert inputs_schema[-1].dtype == InferenceFeatureDtype.STRING
    assert outputs_schema is not None and outputs_schema[0].shape == (chunk_size, action_dim)
    assert outputs_schema[0].name == ACTION
    onnx_export_args = cast("ONNXExportParameters", policy.extra_export_args["onnx"])
    torch_export_args = cast("TorchExportParameters", policy.extra_export_args["torch"])
    openvino_export_args = cast("OpenVINOExportParameters", policy.extra_export_args["openvino"])
    executorch_export_args = cast("ExecuTorchExportParameters", policy.extra_export_args["executorch"])
    assert onnx_export_args.exporter_kwargs["output_names"] == [ACTION]
    assert openvino_export_args.outputs == [ACTION]
    assert executorch_export_args.preprocessors_specs[0].type == "resize"
    assert torch_export_args.postprocessors_specs[0].type == "action_chunk_trimmer"
    print("Export contract: torch, openvino, onnx, executorch")

    model = policy.model
    renamed_image = replace(input_features[1], name="wrist")
    policy.set_features([input_features[0], renamed_image], output_features)
    assert policy.model is model
    assert policy._config is not None and policy._config.input_features[1].name == "wrist"
    policy.rename_features({"wrist": "front"})
    assert policy._config.input_features == input_features
    incompatible_output = [replace(output_features[0], shape=(action_dim + 1,))]
    try:
        policy.set_features(input_features, incompatible_output)
    except ValueError:
        pass
    else:
        raise AssertionError("set_features must reject an incompatible action width")
    print("Feature contract updated without rebuilding the model")

    lora_policy = NewPolicy.from_config(replace(tiny_config, lora_enabled=True, lora_rank=2))
    assert lora_policy.model is not None and is_lora_injected(lora_policy.model)
    lora_checkpoint: dict[str, object] = {}
    lora_policy.on_save_checkpoint(lora_checkpoint)
    restored_lora_policy = NewPolicy(n_action_steps=2)
    restored_lora_policy.on_load_checkpoint(lora_checkpoint)
    assert restored_lora_policy.model is not None
    assert is_lora_injected(restored_lora_policy.model)

    rtc_policy = NewPolicy.from_config(replace(tiny_config, n_action_steps=chunk_size))
    assert isinstance(rtc_policy.model, NewPolicyModel)
    rtc_policy.rtc_enabled = True
    assert rtc_policy.model.enable_rtc
    rtc_prediction = rtc_policy.model.predict_action_chunk(
        rtc_policy._prepare_batch(batch, require_actions=False)
    )
    assert rtc_prediction.shape == (2, chunk_size, action_dim)
    rtc_checkpoint: dict[str, object] = {}
    rtc_policy.on_save_checkpoint(rtc_checkpoint)
    restored_rtc_policy = NewPolicy(n_action_steps=chunk_size)
    restored_rtc_policy.on_load_checkpoint(rtc_checkpoint)
    assert restored_rtc_policy.rtc_enabled
    assert isinstance(restored_rtc_policy.model, NewPolicyModel)
    assert restored_rtc_policy.model.enable_rtc
    print("PEFT and RTC mixins applied through their real policy/model contracts")

    cli_policy = NewPolicy(pretrained_name_or_path="fake/repository", n_action_steps=2)
    assert cli_policy.model is not None
    cli_model = cli_policy.model
    cli_policy.configure_model()
    assert cli_policy.model is cli_model
    assert cli_policy.model.weights_load_count == 1
    assert "pretrained_name_or_path" not in cli_policy.hparams

    checkpoint: dict[str, object] = {}
    policy.on_save_checkpoint(checkpoint)
    restored_policy = NewPolicy(n_action_steps=2)
    restored_policy.on_load_checkpoint(checkpoint)
    assert restored_policy._config == policy._config
    assert restored_policy.model is not None
    assert restored_policy.model.weights_load_count == 0
    print("Checkpoint restored from resolved config without pretrained resolution")

    trainer = Trainer(
        accelerator="cpu",
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    policy.train()
    trainer.fit(policy, datamodule=datamodule)
    assert trainer.global_step == 1
    print(f"Training complete: global_step={trainer.global_step}")
