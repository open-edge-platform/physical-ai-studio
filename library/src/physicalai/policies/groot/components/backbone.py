# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Eagle VLM components for Groot policy.

This module provides the Eagle2 Vision-Language Model components:

- `EagleBackbone`: nn.Module wrapper for the Eagle2 model backbone
- `EagleProcessor`: HuggingFace processor wrapper for image/text encoding

The Eagle2 model is used as the vision-language encoder in NVIDIA's GR00T-N1.5
foundation model, converting images and task descriptions into embeddings
that are then processed by the action head.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torch import nn

if TYPE_CHECKING:
    import types
    from collections.abc import Mapping

    from transformers import ProcessorMixin

logger = logging.getLogger(__name__)

# Default Eagle model/processor locations
DEFAULT_EAGLE_REPO = "nvidia/Eagle2-2B"
DEFAULT_TOKENIZER_ASSETS_REPO = "lerobot/eagle2hg-processor-groot-n1p5"

# Number of channels for RGB images
NUM_RGB_CHANNELS = 3

# Cache directory for Eagle model assets
HF_HOME = Path.home() / ".cache" / "huggingface" / "hub"
HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


def _ensure_eagle_cache_ready(
    cache_dir: Path,
    assets_repo: str,
    *,
    revision: str | None = None,
) -> None:
    """Populate the Eagle processor directory in cache.

    First tries to copy vendored files from lerobot package, then downloads
    missing assets from HuggingFace. This ensures we use compatible versions.

    Args:
        cache_dir: Local cache directory for Eagle assets.
        assets_repo: HuggingFace repo ID for assets.
        revision: Git revision (branch, tag, or commit hash) to download from.
    """
    from shutil import copytree  # noqa: PLC0415

    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Try to copy from lerobot's vendored Eagle files first
    # This ensures we get compatible versions of the model files
    try:
        import lerobot  # noqa: PLC0415

        lerobot_path = Path(lerobot.__file__).parent
        vendor_dir = lerobot_path / "policies" / "groot" / "eagle2_hg_model"
        if vendor_dir.exists():
            logger.info("[Eagle] Copying vendored files from lerobot: %s -> %s", vendor_dir, cache_dir)
            copytree(vendor_dir, cache_dir, dirs_exist_ok=True)
    except ImportError:
        logger.warning("[Eagle] lerobot not installed, will download from HuggingFace")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Eagle] Failed to copy vendored files: %s", exc)

    # Download required tokenizer assets from HuggingFace
    required_assets = [
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.json",
        "special_tokens_map.json",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ]

    logger.info("[Eagle] Preparing cache at %s from %s", cache_dir, assets_repo)

    for fname in required_assets:
        dst = cache_dir / fname
        if not dst.exists():
            try:
                logger.info("[Eagle] Fetching %s", fname)
                hf_hub_download(  # nosec B615 - revision param available for pinning
                    repo_id=assets_repo,
                    filename=fname,
                    repo_type="model",
                    local_dir=str(cache_dir),
                    revision=revision,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Eagle] Failed to download %s: %s", fname, exc)

    # Fix inconsistent class names in config.json
    # NVIDIA's Eagle2-2B has a mismatch: config.json uses "Eagle2_5_VLConfig" but
    # the Python files define "Eagle25VLConfig" (without underscore)
    config_json = cache_dir / "config.json"
    if config_json.exists():
        import json  # noqa: PLC0415

        try:
            with config_json.open(encoding="utf-8") as f:
                config_data = json.load(f)
            if "auto_map" in config_data:
                auto_map = config_data["auto_map"]
                patched = False
                for key, value in auto_map.items():
                    if "Eagle2_5_VL" in value:
                        auto_map[key] = value.replace("Eagle2_5_VL", "Eagle25VL")
                        patched = True
                if patched:
                    with config_json.open("w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=4)
                    logger.info("[Eagle] Patched config.json class names")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Eagle] Failed to patch config.json: %s", exc)


def _find_eagle_module(config: Any) -> types.ModuleType | None:  # noqa: ANN401
    """Locate the dynamically-loaded Eagle2 modelling module in ``sys.modules``.

    Falls back to forcing an import via ``get_class_from_dynamic_module`` when the
    module has not been loaded yet (e.g. only ``AutoConfig`` was called so far).

    Returns:
        The Python module containing ``Eagle25VLForConditionalGeneration``, or *None*.
    """
    import sys  # noqa: PLC0415

    def _scan() -> types.ModuleType | None:
        for _mod_name, mod in sys.modules.items():
            if "modeling_eagle2_5_vl" in _mod_name and hasattr(mod, "Eagle25VLForConditionalGeneration"):
                return mod
        return None

    found = _scan()
    if found is not None:
        return found

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: PLC0415

        auto_map = getattr(config, "auto_map", {})
        model_ref = auto_map.get("AutoModel", "")
        if model_ref:
            repo_id = getattr(config, "_name_or_path", "") or str(HF_LEROBOT_HOME / DEFAULT_TOKENIZER_ASSETS_REPO)
            get_class_from_dynamic_module(model_ref, repo_id)
            return _scan()
    except (ImportError, OSError, ValueError, KeyError):
        logger.debug("[Eagle] Failed to force-load Eagle2 module", exc_info=True)

    return None


def _patch_eagle_model_class(config: Any, attn_implementation: str) -> None:  # noqa: ANN401
    """Patch Eagle2 to replace hardcoded ``flash_attention_2`` with *attn_implementation*.

    NVIDIA's upstream ``Eagle25VLForConditionalGeneration.__init__`` hardcodes
    ``flash_attention_2`` for its sub-models. With ``transformers >= 4.54`` the
    implementation is validated early, so we temporarily guard ``__setattr__`` on
    the vision / text sub-configs to silently redirect those writes.

    The patch is idempotent.
    """
    import functools  # noqa: PLC0415

    eagle_module = _find_eagle_module(config)
    if eagle_module is None:
        logger.warning("[Eagle] Could not find Eagle2 module in sys.modules; skipping attention patch")
        return

    eagle_cls = eagle_module.Eagle25VLForConditionalGeneration
    if getattr(eagle_cls, "_attn_patched", False):
        return

    original_init = eagle_cls.__init__

    @functools.wraps(original_init)
    def _patched_init(
        self_inner: nn.Module,
        cfg: object,
        vision_model: nn.Module | None = None,
        language_model: nn.Module | None = None,
    ) -> None:
        guards: list[tuple[type, object]] = []
        for attr in ("vision_config", "text_config"):
            sub_cfg = getattr(cfg, attr, None)
            if sub_cfg is None:
                continue
            sub_cfg._attn_implementation = attn_implementation  # noqa: SLF001
            cfg_cls = type(sub_cfg)
            orig_setattr = cfg_cls.__setattr__

            def _guarded_setattr(
                self: object,
                name: str,
                value: object,
                *,
                _orig: object = orig_setattr,
                _impl: str = attn_implementation,
            ) -> None:
                if name == "_attn_implementation":
                    value = _impl
                _orig(self, name, value)  # type: ignore[operator]

            cfg_cls.__setattr__ = _guarded_setattr  # type: ignore[assignment]
            guards.append((cfg_cls, orig_setattr))

        try:
            original_init(self_inner, cfg, vision_model=vision_model, language_model=language_model)
        finally:
            for cfg_cls, orig in guards:
                cfg_cls.__setattr__ = orig  # type: ignore[assignment]

    eagle_cls.__init__ = _patched_init
    eagle_cls._attn_patched = True  # noqa: SLF001
    logger.info("[Eagle] Patched Eagle25VLForConditionalGeneration to use %s", attn_implementation)


def _import_huggingface_components() -> tuple[Any, ...]:
    """Import HuggingFace components for Eagle backbone.

    The Eagle VLM is loaded from HuggingFace transformers with trust_remote_code=True.
    Model weights are downloaded via huggingface_hub.

    Returns:
        Tuple of imported components in order:
        (snapshot_download, HfHubHTTPError, RepositoryNotFoundError,
         AutoConfig, AutoModel, BatchFeature)

    Raises:
        ImportError: If huggingface_hub or transformers is not installed.
    """
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError  # noqa: PLC0415
        from transformers import AutoConfig, AutoModel  # noqa: PLC0415
        from transformers.feature_extraction_utils import BatchFeature  # noqa: PLC0415

    except ImportError as e:
        msg = (
            "Eagle components require huggingface_hub and transformers.\n\n"
            "Install with:\n"
            "    pip install huggingface_hub transformers"
        )
        raise ImportError(msg) from e

    return (
        snapshot_download,
        HfHubHTTPError,
        RepositoryNotFoundError,
        AutoConfig,
        AutoModel,
        BatchFeature,
    )


class EagleBackbone(nn.Module):
    """Eagle2 VLM backbone for vision-language encoding.

    This is a wrapper around the Eagle2 model that:
    - Uses SDPA attention by default (no Flash Attention dependency)
    - Supports selective fine-tuning of LLM and vision components
    - Loads pretrained weights from HuggingFace transformers

    Args:
        tokenizer_assets_repo: HF repo ID for Eagle model/tokenizer assets.
        attn_implementation: Attention implementation ('sdpa', 'flash_attention_2', 'eager').
        tune_llm: Whether to fine-tune the LLM backbone.
        tune_visual: Whether to fine-tune the vision tower.

    Examples:
        >>> backbone = EagleBackbone(attn_implementation="sdpa")
        >>> # Input batch with eagle_* tensors from EagleProcessor
        >>> outputs = backbone(batch)
        >>> features = outputs["backbone_features"]  # (B, seq_len, 1536)
    """

    def __init__(
        self,
        tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
        attn_implementation: str = "sdpa",
        *,
        tune_llm: bool = False,
        tune_visual: bool = False,
        project_to_dim: int | None = None,
    ) -> None:
        """Initialize Eagle backbone.

        Args:
            tokenizer_assets_repo: HF repo ID for Eagle tokenizer assets.
            attn_implementation: Attention implementation ('sdpa', 'flash_attention_2', 'eager').
            tune_llm: Whether to fine-tune the LLM backbone.
            tune_visual: Whether to fine-tune the vision tower.
            project_to_dim: Optional output dimension for linear projection.
                If None, outputs raw Eagle hidden size (2048).
        """
        super().__init__()

        # Import HuggingFace components
        (
            _snapshot_download,
            _hf_hub_http_error,
            _repo_not_found_error,
            auto_config_cls,
            auto_model_cls,
            _batch_feature,
        ) = _import_huggingface_components()

        self.tune_llm = tune_llm
        self.tune_visual = tune_visual

        # Prepare cache directory with all required Eagle assets
        cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
        _ensure_eagle_cache_ready(cache_dir, tokenizer_assets_repo)

        # Load config and override attention implementation
        eagle_config = auto_config_cls.from_pretrained(str(cache_dir), trust_remote_code=True)

        # Override attention implementation to avoid Flash Attention dependency
        # Set on all sub-configs so SDPA is used everywhere
        eagle_config._attn_implementation = attn_implementation  # noqa: SLF001
        eagle_config._attn_implementation_autoset = False  # noqa: SLF001
        if hasattr(eagle_config, "text_config"):
            eagle_config.text_config._attn_implementation = attn_implementation  # noqa: SLF001
            eagle_config.text_config._attn_implementation_autoset = False  # noqa: SLF001
        if hasattr(eagle_config, "vision_config"):
            eagle_config.vision_config._attn_implementation = attn_implementation  # noqa: SLF001
            eagle_config.vision_config._attn_implementation_autoset = False  # noqa: SLF001

        # Patch NVIDIA's Eagle2 model class to remove hardcoded flash_attention_2 requirements.
        # The upstream code (modeling_eagle2_5_vl.py) hardcodes flash_attention_2 in two places:
        #   1. config.vision_config._attn_implementation = "flash_attention_2" (for SiglipVisionModel)
        #   2. assert config.text_config._attn_implementation == "flash_attention_2" (for Qwen2)
        # With transformers >= 4.54, flash_attention_2 is validated during PreTrainedModel.__init__,
        # so we must patch the class to use our desired implementation instead.
        _patch_eagle_model_class(eagle_config, attn_implementation)

        self.eagle_model = auto_model_cls.from_config(
            eagle_config,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        # Force patch attention implementation on all submodules AFTER model creation
        # HuggingFace creates separate config instances during model init, so we need
        # to patch them recursively after the model is constructed
        self._patch_attention_implementation(attn_implementation)

        # Optional linear projection from Eagle hidden size (2048) to output dim
        # If project_to_dim is None, output raw 2048 hidden size (used by pretrained GR00T)
        self.project_to_dim = project_to_dim
        if project_to_dim is not None:
            self.eagle_linear: nn.Module = nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = nn.Identity()

        # Which hidden state layer to use (-1 = last)
        self.select_layer = -1

        # Remove unused layers to save compute (matches Isaac-GR00T behavior)
        # When select_layer=-1, we use the last layer so no pruning needed
        # For other values, we'd prune: while len(layers) > select_layer: layers.pop(-1)
        # Currently keeping all layers since select_layer=-1 uses last hidden state

        self._set_trainable_parameters()

    def _set_trainable_parameters(self) -> None:
        """Configure which parameters are trainable."""
        for p in self.parameters():
            p.requires_grad = True
        if not self.tune_llm:
            self.eagle_model.language_model.requires_grad_(requires_grad=False)
        if not self.tune_visual:
            self.eagle_model.vision_model.requires_grad_(requires_grad=False)
            self.eagle_model.mlp1.requires_grad_(requires_grad=False)

    def _patch_attention_implementation(self, impl: str) -> None:
        """Recursively patch _attn_implementation on all submodule configs.

        HuggingFace models create separate config instances during initialization,
        so setting the config beforehand doesn't propagate to attention layers.
        This method patches all configs after model construction.

        Args:
            impl: Attention implementation ('sdpa', 'flash_attention_2', 'eager').
        """
        patched_count = 0
        for _name, module in self.eagle_model.named_modules():
            if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
                module.config._attn_implementation = impl  # noqa: SLF001
                patched_count += 1
        logger.debug("[Eagle] Patched %d modules with attention implementation: %s", patched_count, impl)

    def _set_frozen_modules_to_eval_mode(self) -> None:
        """Set frozen modules to eval mode during training."""
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass through Eagle backbone.

        Args:
            batch: Input batch with eagle_* prefixed tensors from EagleProcessor.

        Returns:
            Dict with backbone_features and backbone_attention_mask.
        """
        self._set_frozen_modules_to_eval_mode()

        # Extract eagle inputs and move to model device
        device = next(self.parameters()).device
        eagle_prefix = "eagle_"
        eagle_input = {}
        for k, v in batch.items():
            if k.startswith(eagle_prefix):
                key = k.removeprefix(eagle_prefix)
                if key != "image_sizes":  # Skip image_sizes
                    eagle_input[key] = v.to(device) if isinstance(v, torch.Tensor) else v

        # Forward through Eagle
        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        # Apply projection (either Linear or Identity)
        eagle_features = self.eagle_linear(eagle_features)

        # DDP compatibility hack: ensure all trainable vision parameters are used in forward pass
        # This prevents DDP from complaining about unused parameters when tune_visual=True
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0,
                device=eagle_features.device,
                dtype=eagle_features.dtype,
                requires_grad=True,
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term += 0.0 * param.sum()
            eagle_features += dummy_term

        return {
            "backbone_features": eagle_features,
            "backbone_attention_mask": eagle_input["attention_mask"],
        }


@dataclass
class EagleProcessor:
    """Eagle VLM processor for image and text encoding.

    Wraps HuggingFace's AutoProcessor for Eagle2, handling:
    - Image preprocessing (resize, tile, normalize with ImageNet stats)
    - Text tokenization with chat template
    - Batching and padding

    Args:
        processor_repo: HuggingFace repo ID or local path for Eagle processor.
        min_dynamic_tiles: Minimum number of image tiles.
        max_dynamic_tiles: Maximum number of image tiles.
        use_thumbnail: Whether to include thumbnail in tiling.

    Examples:
        Basic usage:

        >>> processor = EagleProcessor()
        >>> images = [Image.open("frame.jpg")]
        >>> text = "Pick up the cube"
        >>> eagle_inputs = processor(images, text)
        >>> # eagle_inputs contains: eagle_input_ids, eagle_attention_mask,
        >>> # eagle_pixel_values, eagle_image_grid_thw

        Batch processing:

        >>> batch_images = [[img1, img2], [img3, img4]]  # 2 samples, 2 views each
        >>> batch_text = ["task 1", "task 2"]
        >>> eagle_inputs = processor.batch_encode(batch_images, batch_text)
    """

    processor_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO
    processor_revision: str | None = None
    min_dynamic_tiles: int = 1
    max_dynamic_tiles: int = 1
    use_thumbnail: bool = False

    _processor: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def processor(self) -> ProcessorMixin:
        """Lazy-load the HuggingFace processor."""
        if self._processor is None:
            self._processor = self._load_processor()
        return self._processor

    def _load_processor(self) -> ProcessorMixin:
        """Load Eagle processor from HuggingFace.

        Returns:
            Loaded HuggingFace processor with left padding configured.

        Raises:
            ImportError: If transformers is not installed.
        """
        try:
            from transformers import AutoProcessor  # noqa: PLC0415
            from transformers.image_processing_utils_fast import BaseImageProcessorFast  # noqa: PLC0415
        except ImportError as e:
            msg = "EagleProcessor requires transformers. Install with: pip install transformers"
            raise ImportError(msg) from e

        # Monkey-patch missing method if needed (transformers version compatibility)
        # The vendored Eagle processor expects _prepare_image_like_inputs but
        # transformers 4.53.x uses _prepare_input_images
        if not hasattr(BaseImageProcessorFast, "_prepare_image_like_inputs") and hasattr(
            BaseImageProcessorFast,
            "_prepare_input_images",
        ):
            BaseImageProcessorFast._prepare_image_like_inputs = (  # noqa: SLF001
                BaseImageProcessorFast._prepare_input_images  # noqa: SLF001
            )
            logger.info("[EagleProcessor] Patched _prepare_image_like_inputs method")

        # Prepare cache directory with all required Eagle assets
        cache_dir = HF_LEROBOT_HOME / self.processor_repo
        _ensure_eagle_cache_ready(cache_dir, self.processor_repo, revision=self.processor_revision)

        logger.info("[EagleProcessor] Loading processor from %s", cache_dir)

        # Load from local cache (already downloaded with revision pinning above)
        processor = AutoProcessor.from_pretrained(  # nosec B615
            str(cache_dir),
            trust_remote_code=True,
        )
        # Set left padding for decoder-style models
        processor.tokenizer.padding_side = "left"
        return processor

    def __call__(
        self,
        images: list[Image.Image],
        text: str,
        *,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Encode images and text for a single sample.

        Args:
            images: List of PIL Images (e.g., from multiple camera views).
            text: Task description text.
            return_tensors: Output format ("pt" for PyTorch tensors).

        Returns:
            Dict with eagle_* tensors:
                - eagle_input_ids: Tokenized text (1, seq_len)
                - eagle_attention_mask: Attention mask (1, seq_len)
                - eagle_pixel_values: Processed images (1, num_patches, channels)
                - eagle_image_grid_thw: Image grid dimensions
        """
        # Format as conversation for chat template
        image_content = [{"type": "image", "image": img} for img in images]
        # Format language as string list representation to match GR00T format
        text_formatted = str([text])
        text_content = [{"type": "text", "text": text_formatted}]
        conversation = [{"role": "user", "content": image_content + text_content}]

        # Apply chat template
        text_list = [self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)]

        # Process vision info
        image_inputs, _video_inputs = self.processor.process_vision_info(conversation)

        # Encode with processor
        eagle_inputs = self.processor(
            text=text_list,
            images=image_inputs,
            images_kwargs={
                "min_dynamic_tiles": self.min_dynamic_tiles,
                "max_dynamic_tiles": self.max_dynamic_tiles,
                "use_thumbnail": self.use_thumbnail,
            },
            return_tensors=return_tensors,
            padding=True,
        )

        # Prefix keys with eagle_
        return {f"eagle_{k}": v for k, v in eagle_inputs.items()}

    def batch_encode(
        self,
        batch_images: list[list[Image.Image]],
        batch_text: list[str],
        *,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of images and text.

        Args:
            batch_images: List of image lists, one per sample.
            batch_text: List of task descriptions, one per sample.
            return_tensors: Output format ("pt" for PyTorch tensors).

        Returns:
            Dict with batched eagle_* tensors.
        """
        all_text: list[str] = []
        all_images: list[Any] = []

        for images, text in zip(batch_images, batch_text, strict=True):
            # Format as conversation
            image_content = [{"type": "image", "image": img} for img in images]
            text_formatted = str([text])
            text_content = [{"type": "text", "text": text_formatted}]
            conversation = [{"role": "user", "content": image_content + text_content}]

            # Apply chat template
            text_str = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            all_text.append(text_str)

            # Process vision info
            image_inputs, _video_inputs = self.processor.process_vision_info(conversation)
            all_images.extend(image_inputs)

        # Batch encode
        eagle_inputs = self.processor(
            text=all_text,
            images=all_images,
            images_kwargs={
                "min_dynamic_tiles": self.min_dynamic_tiles,
                "max_dynamic_tiles": self.max_dynamic_tiles,
                "use_thumbnail": self.use_thumbnail,
            },
            return_tensors=return_tensors,
            padding=True,
        )

        return {f"eagle_{k}": v for k, v in eagle_inputs.items()}

    def encode_video(
        self,
        video: np.ndarray | torch.Tensor,
        text: str,
        *,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Encode video frames and text.

        Convenience method for encoding video data in the format used by
        robotics datasets: (B, T, V, C, H, W) or (B, T, V, H, W, C).

        Args:
            video: Video tensor of shape (B, T, V, C, H, W) or (B, T, V, H, W, C).
                   Can be uint8 [0, 255] or float [0, 1].
            text: Task description text (same for all samples in batch).
            return_tensors: Output format ("pt" for PyTorch tensors).

        Returns:
            Dict with batched eagle_* tensors.
        """
        # Convert to numpy if tensor
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()

        # Normalize to uint8 if float
        if video.dtype in {np.float32, np.float64}:
            video = (np.clip(video, 0, 1) * 255).astype(np.uint8)

        # Determine layout: (B, T, V, C, H, W) vs (B, T, V, H, W, C)
        # If shape[-1] == 3, assume BHWC; otherwise BCHW
        if video.shape[-1] == NUM_RGB_CHANNELS:
            # (B, T, V, H, W, C) -> (B, T, V, C, H, W)
            video = np.transpose(video, (0, 1, 2, 5, 3, 4))

        batch_size = video.shape[0]
        num_timesteps = video.shape[1]
        num_views = video.shape[2]

        batch_images: list[list[Image.Image]] = []
        batch_text: list[str] = []

        for b in range(batch_size):
            images: list[Image.Image] = []
            for t in range(num_timesteps):
                for v in range(num_views):
                    # (C, H, W) -> (H, W, C)
                    frame = np.transpose(video[b, t, v], (1, 2, 0))
                    images.append(Image.fromarray(frame))
            batch_images.append(images)
            batch_text.append(text)

        return self.batch_encode(batch_images, batch_text, return_tensors=return_tensors)
