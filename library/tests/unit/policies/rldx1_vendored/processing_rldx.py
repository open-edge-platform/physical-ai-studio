# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Literal
import warnings

import numpy as np
from PIL import Image
from physicalai.policies.rldx1.augmentations import AspectAreaResizeAndCrop
from physicalai.policies.rldx1.components.embodiments import GENERAL_EMBODIMENT_ID
from tests.unit.policies.rldx1_vendored.data_utils import (
    parse_modality_configs,
    to_json_serializable,
)
from .data_types import ModalityConfig
from physicalai.policies.rldx1.components.processing.qwen_vision_process import (
    process_vision_info as qwen_process_vision_info,
)
from tests.unit.policies.rldx1_vendored.state_action_processor import StateActionProcessor
import torch
from transformers import AutoProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import cached_file

warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")


def build_processor(model_name: str, transformers_loading_kwargs: dict) -> ProcessorMixin:
    """Load a processor from a local path or HF Hub id.

    `AutoProcessor.from_pretrained` already accepts both local paths (absolute
    or relative) and HF repo ids, so no private-filesystem fallback is needed
    here. If you need to redirect a repo id to a local cache, pass the local
    path directly as `model_name`.
    """
    return AutoProcessor.from_pretrained(model_name, **transformers_loading_kwargs)


class RLDXDataCollator:
    def __init__(
        self,
        model_name: str,
        model_type: Literal["qwen3_vl", "vtc_qwen3_vl"] = "qwen3_vl",
        transformers_loading_kwargs: dict = {},
    ):
        ### We need to use the same  processor for padding input ids and concat
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name

    def _collate_vlm_content(self, values: list[dict]) -> dict:
        """Collate vlm_content from B samples into batched VLM inputs."""
        text_list = []
        image_inputs = []
        video_inputs: list[Any] = []
        for v in values:
            text_list.append(v["text"])
            image_inputs += v["images"]

        if "qwen" in self.model_type:
            image_inputs = qwen_process_vision_info(
                [v["conversation"] for v in values], image_patch_size=16
            )

        processor_kwargs = {
            "text": text_list,
            "return_tensors": "pt",
            "padding": True,
            "do_resize": False,
        }
        if len(image_inputs) > 0:
            processor_kwargs["images"] = image_inputs
        if len(video_inputs) > 0:
            processor_kwargs["videos"] = video_inputs
            processor_kwargs["do_sample_frames"] = False

        return self.processor(**processor_kwargs)

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                vlm_inputs = self._collate_vlm_content(values)
                for k, v in vlm_inputs.items():
                    batch[k] = v

                if "vtc" in self.model_type:
                    batch["image_wise_encoding"] = torch.tensor(
                        [v["image_wise_encoding"] for v in values]
                    )
                    batch["num_views"] = torch.tensor([v["num_views"] for v in values])
                    if "num_frames" in values[0]:
                        batch["num_frames"] = torch.tensor([v["num_frames"] for v in values])

            elif key == "vlm_content_list":
                # Memory mode: each sample has K vlm_contents
                # Flatten B samples × K timesteps into B*K backbone inputs
                # Order: [sample0_t0, sample0_t1, ..., sample0_tK-1, sample1_t0, ...]
                all_vlm_contents = []
                for sample_contents in values:
                    all_vlm_contents.extend(sample_contents)

                vlm_inputs = self._collate_vlm_content(all_vlm_contents)
                for k, v in vlm_inputs.items():
                    batch[k] = v

                if "vtc" in self.model_type:
                    batch["image_wise_encoding"] = torch.tensor([1] * len(all_vlm_contents))
                    batch["num_views"] = torch.tensor(
                        [
                            content["num_views"]
                            for sample_contents in values
                            for content in sample_contents
                        ]
                    )
                    if "num_frames" in all_vlm_contents[0]:
                        batch["num_frames"] = torch.tensor(
                            [
                                content["num_frames"]
                                for sample_contents in values
                                for content in sample_contents
                            ]
                        )

            elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
                raise Exception("Not implemented")

            else:
                # state, state_mask, action and action_mask - stack to form batch dimension
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        return f"RLDXDataCollator(model_name={self.model_name}, model_type={self.model_type})"


class RLDXProcessor(ProcessorMixin):
    data_collator_class = RLDXDataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        # Image pipeline (see rldx/data/augmentations.py)
        image_max_area: int = 65536,
        image_resize_m: int = 32,
        random_crop_fraction: float | None = None,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict[str, float] | None = None,
        formalize_language: bool = True,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        model_type: Literal["qwen3_vl", "vtc_qwen3_vl"] = "qwen3_vl",
        max_state_dim: int = 32,
        max_action_dim: int = 64,
        apply_sincos_state_encoding: bool = False,
        max_action_horizon: int = 16,
        transformers_loading_kwargs: dict = {"trust_remote_code": True, "use_fast": True},
        memory_length: int = 1,
        general_embodiment_train_ratio: float = 0,
        embodiment_id: int = GENERAL_EMBODIMENT_ID,
        conversation_image_first: bool = False,
        physics_keys: list[str] | None = None,
        physics_dims: list[int] | None = None,
        allow_missing_physics: bool = False,
    ):
        self.physics_keys = physics_keys or []
        self.physics_dims = physics_dims or []
        self.allow_missing_physics = allow_missing_physics

        # Pre-compute physics temporal length for zero-filling when data is missing
        self._physics_t_len = 0
        if self.allow_missing_physics and self.physics_keys:
            for emb_cfg in modality_configs.values():
                if isinstance(emb_cfg, dict):
                    for pk in self.physics_keys:
                        if pk in emb_cfg:
                            mc = emb_cfg[pk]
                            di = (
                                mc.delta_indices
                                if hasattr(mc, "delta_indices")
                                else getattr(mc, "delta_indices", [])
                            )
                            if di:
                                self._physics_t_len = len(di)
                                break
                if self._physics_t_len > 0:
                    break

        self.modality_configs = parse_modality_configs(modality_configs)

        # Initialize StateActionProcessor for state/action normalization
        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
        )

        # Save state action processor settings
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding

        # Save VLM settings
        self.formalize_language = formalize_language
        self.model_name = model_name
        self.model_type = model_type

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon

        # Save image pipeline settings. Some legacy checkpoints persisted
        # ``image_max_area`` / ``image_resize_m`` as ``null`` because the
        # training recipe that produced them never tripped the
        # ``math.sqrt(max_area / (h * w))`` path (256x256 inputs are a no-op
        # at the default 65536 budget). At inference we still need a concrete
        # int, so coerce here and warn loudly instead of crashing later.
        if image_max_area is None:
            print(
                f"[!] processor: image_max_area=None in checkpoint; "
                f"falling back to default {65536}. Set it explicitly to silence."
            )
            image_max_area = 65536
        if image_resize_m is None:
            print(
                f"[!] processor: image_resize_m=None in checkpoint; "
                f"falling back to default {32}. Set it explicitly to silence."
            )
            image_resize_m = 32
        self.image_max_area = image_max_area
        self.image_resize_m = image_resize_m
        self.random_crop_fraction = random_crop_fraction
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params

        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"

        self.memory_length = memory_length
        self.general_embodiment_train_ratio = general_embodiment_train_ratio
        self.embodiment_id = embodiment_id
        self.conversation_image_first = conversation_image_first

        # Deterministic torchvision geometry (no train-time stochastic aug), shared
        # by train/eval -- see physicalai.policies.rldx1.augmentations docstring.
        image_transform = AspectAreaResizeAndCrop(target_area=image_max_area, m_alignment=image_resize_m)
        self.train_image_transform = image_transform
        self.eval_image_transform = image_transform
        self._collator = self.data_collator_class(
            model_name=model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        self.train()

    @property
    def collator(self):
        return self._collator

    def train(self):
        self.training = True
        self.state_action_processor.train()

    def eval(self):
        self.training = False
        self.state_action_processor.eval()

    def get_modality_configs(self) -> dict[str, ModalityConfig]:
        """Return the parsed modality configs (``[embodiment_tag][modality]``)."""
        return self.modality_configs

    def set_statistics(
        self,
        statistics: dict[str, Any],
        override: bool = False,
    ) -> None:
        """Set dataset statistics for normalization."""
        self.state_action_processor.set_statistics(statistics, override=override)

        # Compute action dimensions for convenience
        self.action_dim = self.state_action_processor.get_action_dim()

    def decode_action(
        self,
        action: np.ndarray,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Undo action normalization and convert relative actions to absolute."""
        # Split concatenated action into joint groups
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs["action"].modality_keys
        action_horizon = len(self.modality_configs["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params["action"][key]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Use StateActionProcessor to unnormalize and convert to absolute
        return self.state_action_processor.unapply_action(
            out_dict, state=state
        )

    def _apply_vlm_processing(
        self, images: np.ndarray | list[np.ndarray], language: str
    ) -> BatchFeature:
        """
        Args:
            images: [T*V, C, H, W] or list of [C, H, W] arrays
        Returns: vlm_content format for collation
        """
        # Convert images to PIL format
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]

        if not self.conversation_image_first:
            # Create conversation with images and text
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": language},
                        *[{"type": "image", "image": img} for img in pil_images],
                    ],
                }
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in pil_images],
                        {"type": "text", "text": language},
                    ],
                }
            ]

        # Apply chat template but don't process yet - let collator handle it
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Return vlm_content format for collation
        return {
            "vlm_content": {
                "text": text,
                "images": pil_images,
                "conversation": conversation,
            }
        }

    def __call__(
        self,
        messages: list[dict[str, Any]],
    ):
        assert len(messages) == 1
        content = messages[0]["content"]
        action_data = content["actions"]
        state_data = content["states"]

        # Use StateActionProcessor to handle relative conversion and normalization
        normalized_states, normalized_actions = self.state_action_processor.apply(
            state=state_data,
            action=action_data,
        )

        if normalized_actions:
            # Concatenate actions
            action_keys = self.modality_configs["action"].modality_keys
            normalized_actions = torch.cat(
                [torch.from_numpy(normalized_actions[key]) for key in action_keys], dim=-1
            )  # (t, d)
            action_dim = normalized_actions.shape[1]
            # Pad action to max_action_dim
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        normalized_actions.shape[0],
                        self.max_action_dim - normalized_actions.shape[1],
                    ),
                ],
                dim=-1,
            )  # (t, max_action_dim)
            # Pad action to max_action_horizon
            action_horizon = normalized_actions.shape[0]
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        self.max_action_horizon - normalized_actions.shape[0],
                        self.max_action_dim,
                    ),
                ],
                dim=0,
            )  # (max_action_horizon, max_action_dim)
            # Create action mask
            action_mask = torch.ones_like(normalized_actions)
            action_mask[action_horizon:] = 0
            action_mask[:, action_dim:] = 0
        else:
            assert not self.training, "Action is required in training mode"
            normalized_actions = None  # type: ignore[assignment]
            action_mask = None

        # Concatenate states
        state_keys = self.modality_configs["state"].modality_keys
        normalized_states = torch.cat(
            [torch.from_numpy(normalized_states[key]) for key in state_keys], dim=-1
        )
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(
                    normalized_states.shape[0], self.max_state_dim - normalized_states.shape[1]
                ),
            ],
            dim=-1,
        )

        # Crop and resize images.
        if self.training:
            image_transform = self.train_image_transform
        else:
            image_transform = self.eval_image_transform
        image_keys = self.modality_configs["video"].modality_keys

        if self.formalize_language:
            language = content["text"].lower()
            language = re.sub(r"[^\w\s]", "", language)
        else:
            language = content["text"]

        vlm_inputs = self._get_vlm_inputs(
            image_keys=image_keys,
            images=content["images"],
            image_transform=image_transform,
            language=language,
            memory_length=self.memory_length,
        )

        transformed_inputs = {}
        transformed_inputs.update(vlm_inputs)
        if normalized_states is not None:
            transformed_inputs["state"] = normalized_states.to(torch.get_default_dtype())
        if normalized_actions is not None:
            transformed_inputs["action"] = normalized_actions.to(torch.get_default_dtype())  # type: ignore[attr-defined]
        if action_mask is not None:
            transformed_inputs["action_mask"] = action_mask

        # Process physics signals (tactile, torque, etc.) if present
        physics_available = False
        content_physics = content.get("physics")
        if content_physics and self.physics_keys:
            filtered_physics = {
                k: v
                for k, v in content_physics.items()
                if any(k == pk or k.startswith(pk + ".") for pk in self.physics_keys)
            }
            if filtered_physics:
                normalized_physics = self.state_action_processor.apply_physics(
                    filtered_physics
                )
                # apply_physics always returns a dict (possibly empty when every
                # requested modality was dropped as all-zero). The previous
                # `None` sentinel was a symptom of the mid-loop early-return
                # that silently discarded sibling valid modalities.
                assert isinstance(normalized_physics, dict), (
                    f"apply_physics must return a dict; got {type(normalized_physics)!r}"
                )
                # Validate per-key dims and concat in physics_keys order. Any
                # missing modality (apply_physics dropped it as all-zero, or the
                # dataset never shipped it) invalidates the whole physics tensor:
                # concatenating a subset silently produces a tensor smaller than
                # sum(physics_dims), which then mismatches the model's
                # physics_cond_encoder input dim downstream. Require ALL keys.
                per_key_tensors = []
                partial_physics = False
                for pk, expected_dim in zip(self.physics_keys, self.physics_dims):
                    # Sub-keys for this physics key (e.g. "tactile" → "tactile.left", "tactile.right")
                    sub_keys = sorted(
                        k for k in normalized_physics if k == pk or k.startswith(pk + ".")
                    )
                    if not sub_keys:
                        partial_physics = True
                        break
                    key_tensor = torch.cat(
                        [torch.from_numpy(normalized_physics[k]) for k in sub_keys], dim=-1
                    )
                    actual_dim = key_tensor.shape[-1]
                    assert actual_dim == expected_dim, (
                        f"Physics dim mismatch for '{pk}': data has {actual_dim} "
                        f"but --physics-dims specifies {expected_dim}. "
                        f"Sub-keys: {sub_keys}"
                    )
                    per_key_tensors.append(key_tensor)

                if per_key_tensors and not partial_physics:
                    physics_tensor = torch.cat(per_key_tensors, dim=-1)  # (T, sum(physics_dims))
                    transformed_inputs["physics"] = physics_tensor.to(torch.get_default_dtype())
                    physics_available = True

        # When allow_missing_physics=True, always produce physics + physics_mask
        if self.allow_missing_physics and self.physics_keys:
            if physics_available:
                transformed_inputs["physics_mask"] = np.array([1.0], dtype=np.float32)
            else:
                total_dim = sum(self.physics_dims)
                transformed_inputs["physics"] = torch.zeros(
                    self._physics_t_len, total_dim, dtype=torch.get_default_dtype()
                )
                transformed_inputs["physics_mask"] = np.array([0.0], dtype=np.float32)

        # Route every sample through the configured embodiment projector slot.
        # Default 0 (general_embodiment) for a fresh new-robot FT; set the slot a
        # released checkpoint was trained on to reuse it (see
        # EMBODIMENT_TAG_TO_PROJECTOR_INDEX).
        transformed_inputs["embodiment_id"] = self.embodiment_id

        return transformed_inputs

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: dict[str, Any],
        image_transform: AspectAreaResizeAndCrop,
        language: str,
        memory_length: int = 1,
    ):
        temporal_stacked_images = {}

        # Deterministic per-frame resize+crop; no shared-randomness replay needed.
        for view in image_keys:
            assert view in images, f"{view} not in {images}"
            transformed_images = [
                image_transform(torch.from_numpy(np.array(img)).permute(2, 0, 1))
                for img in images[view]
            ]
            temporal_stacked_images[view] = torch.stack(transformed_images)  # (T, C, H, W)

        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"
            assert v.shape[1] == 3, f"{v} is not a 3 channel tensor"

        T = temporal_stacked_images[image_keys[0]].shape[0]

        if memory_length > 1 and self.training:
            # Memory mode: process each timestep separately
            # Produce K separate vlm_content items for per-timestep backbone processing
            # For combined mode (video+memory), T will be memory_length * video_length
            video_length = T // memory_length
            vlm_content_list = []
            for t in range(memory_length):
                stacked_images = []
                start_frame = t * video_length
                end_frame = (t + 1) * video_length
                for frame_idx in range(start_frame, end_frame):
                    for view in image_keys:
                        stacked_images.append(
                            temporal_stacked_images[view][frame_idx].numpy()
                        )  # (C,H,W)
                vlm_inputs = self._apply_vlm_processing(stacked_images, language)

                if "vtc" in self.model_type:
                    vlm_inputs["vlm_content"]["image_wise_encoding"] = True
                    vlm_inputs["vlm_content"]["num_views"] = len(image_keys)
                    vlm_inputs["vlm_content"]["num_frames"] = video_length
                vlm_content_list.append(vlm_inputs["vlm_content"])

            return {"vlm_content_list": vlm_content_list}
        else:
            # Standard mode: all images in a single VLM message
            stacked_images = []
            for t in range(T):
                for view in image_keys:
                    stacked_images.append(temporal_stacked_images[view][t].numpy())
            vlm_inputs = self._apply_vlm_processing(stacked_images, language)

            if "vtc" in self.model_type:
                vlm_inputs["vlm_content"]["image_wise_encoding"] = True
                vlm_inputs["vlm_content"]["num_views"] = len(image_keys)
                vlm_inputs["vlm_content"]["num_frames"] = T

            return vlm_inputs

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        # dump modality configs to dict using the recursive function
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        main_config_file = Path(save_directory) / "processor_config.json"
        statistics_file = Path(save_directory) / "statistics.json"

        config = {
            "processor_class": self.__class__.__name__,
            "processor_kwargs": {
                "modality_configs": to_json_serializable(self.modality_configs),
                # Image pipeline settings
                "image_max_area": self.image_max_area,
                "image_resize_m": self.image_resize_m,
                "random_crop_fraction": self.random_crop_fraction,
                "random_rotation_angle": self.random_rotation_angle,
                "color_jitter_params": self.color_jitter_params,
                # VLM settings
                "model_name": self.model_name,
                "model_type": self.model_type,
                "formalize_language": self.formalize_language,
                # State action dimensions
                "max_state_dim": self.max_state_dim,
                "max_action_dim": self.max_action_dim,
                "max_action_horizon": self.max_action_horizon,
                # StateActionProcessor settings
                "use_percentiles": self.use_percentiles,
                "clip_outliers": self.clip_outliers,
                "apply_sincos_state_encoding": self.apply_sincos_state_encoding,
                "use_relative_action": self.use_relative_action,
                "memory_length": self.memory_length,
                "general_embodiment_train_ratio": self.general_embodiment_train_ratio,
                # Persisted so a reloaded processor routes through the same
                # projector slot the checkpoint was trained on. Reverting to the
                # ctor default (35) would silently break released benchmark FTs.
                "embodiment_id": self.embodiment_id,
                "allow_missing_physics": self.allow_missing_physics,
                "physics_keys": self.physics_keys if self.physics_keys else None,
                "physics_dims": self.physics_dims if self.physics_dims else None,
                # Persisted so AutoProcessor.from_pretrained(ckpt) doesn't
                # silently revert to the ctor default (False). The attribute
                # is used by the transform path, so a wrong value degrades
                # inference without any error surface.
                "conversation_image_first": self.conversation_image_first,
            },
        }
        with open(main_config_file, "w") as f:
            json.dump(config, f, indent=2)
        # Save statistics
        with open(statistics_file, "w") as f:
            json.dump(to_json_serializable(self.state_action_processor.statistics), f, indent=2)
        return [main_config_file, statistics_file]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        transformers_loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config_file = pretrained_model_name_or_path / "processor_config.json"
        statistics_file = pretrained_model_name_or_path / "statistics.json"
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if not is_local:
            config_file = Path(cached_file(pretrained_model_name_or_path, "processor_config.json"))
            statistics_file = Path(cached_file(pretrained_model_name_or_path, "statistics.json"))

        with open(config_file, "r") as f:
            config = json.load(f)
        with open(statistics_file, "r") as f:
            statistics = json.load(f)
        processor_kwargs = config["processor_kwargs"]
        processor_kwargs["statistics"] = statistics

        memory_length = processor_kwargs.get("memory_length", 1)
        print(f"[i] Loaded memory_length={memory_length} from checkpoint\n")

        # Directly override other processor kwargs
        if kwargs:
            # Override modality configs with flat modality keys.
            modality_configs = kwargs.pop("modality_configs", {})
            for modality, modality_config in modality_configs.items():
                processor_kwargs["modality_configs"][modality] = modality_config
            override_keys = [
                "random_rotation_angle",
                "color_jitter_params",
                "use_relative_action",
                "general_embodiment_train_ratio",
                "memory_length",
                "physics_keys",
                "physics_dims",
                "allow_missing_physics",
                "max_action_horizon",
                # ``setup.py`` passes these as kwargs to
                # AutoProcessor.from_pretrained when the CLI specifies a
                # non-default value; without them on this whitelist the
                # loaded processor would silently fall back to the
                # checkpoint-saved value, producing a drift between
                # model.config and the processor.
                "image_max_area",
                "image_resize_m",
                "random_crop_fraction",
                "formalize_language",
                "apply_sincos_state_encoding",
                "conversation_image_first",
            ]
            for key in override_keys:
                if key in kwargs:
                    override = kwargs.pop(key)
                    if override is not None:
                        processor_kwargs[key] = override
        return cls(**processor_kwargs, transformers_loading_kwargs=transformers_loading_kwargs)
