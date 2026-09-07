# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: noqa: T201

"""VTC Qwen3-VL model with motion module support."""

import json
import pathlib
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open
from transformers import AutoConfig

from .layer_wrapper import LayerWrapper
from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration

# Pinned commit for the upstream Qwen3-VL reference architecture config (lib.security
# rule 9). This only supplies the base config schema for the VTC variant, never weights,
# but is still pinned to avoid an unpinned Hub read at import/build time.
_QWEN3_VL_REFERENCE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def _checkpoint_has_motion_weights(  # noqa: PLR0911
    path_or_name: str,
    revision: str | None = None,
) -> bool | None:
    """Detect whether a checkpoint carries `motion_block.*` parameter tensors.

    Returns:
        True:  definitively contains motion module weights (skip re-init)
        False: definitively does not contain motion module weights (safe to re-init)
        None:  could not determine (conservative: preserve loaded weights)

    Works for both sharded (model.safetensors.index.json + model-*-of-*.safetensors)
    and non-sharded (single model.safetensors) layouts. Handles local directories
    and HF Hub identifiers. Any probe failure returns None rather than a
    false-negative, because a false-negative here silently overwrites trained
    motion module weights via `motion_block.initialize_weights()`.
    """
    if pathlib.Path(path_or_name).is_dir():
        shards = sorted(pathlib.Path(path_or_name).glob("*.safetensors"))
        if not shards:
            print(f"[w] motion module probe: no *.safetensors under {path_or_name}")
            return None
        try:
            for shard in shards:
                with safe_open(shard, framework="pt") as f:
                    if any("motion_block" in k for k in f):
                        return True
        except OSError as exc:
            print(f"[w] motion module probe: scan failed on {path_or_name}: {exc!r}")
            return None
        else:
            return False

    try:
        index_path = hf_hub_download(
            path_or_name,
            "model.safetensors.index.json",
            revision=revision,
        )
    except EntryNotFoundError:
        index_path = None
    except OSError as exc:
        print(f"[w] motion module probe: index.json download failed for {path_or_name}: {exc!r}")
        return None

    if index_path is not None:
        try:
            with pathlib.Path(index_path).open(encoding="utf-8") as f:
                index = json.load(f)
            return any("motion_block" in k for k in index.get("weight_map", {}))
        except (OSError, ValueError) as exc:
            print(f"[w] motion module probe: index.json parse failed: {exc!r}")
            return None

    # Non-sharded HF Hub checkpoint: don't download the full model.safetensors
    # (potentially 10s of GB) just to read key names. Return indeterminate and
    # let the caller preserve whatever weights from_pretrained already loaded.
    print(
        f"[w] motion module probe: {path_or_name} has no sharded index.json. Not "
        f"downloading full model.safetensors for key scan; preserving "
        f"loaded motion module weights (skipping re-init).",
    )
    return None


class VTCQwen3Model(Qwen3VLForConditionalGeneration):
    """VTC Qwen3-VL model with motion module support."""

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        pretrained_model_name_or_path: str,
        motion_config: dict | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Qwen3VLForConditionalGeneration:
        """Load a VTC_Qwen3VL model from a pretrained checkpoint.

        Args:
            pretrained_model_name_or_path: HuggingFace Hub repo ID or local
                path to the pretrained model.
            motion_config: Optional motion-module configuration dict injected
                into the VTC architecture.
            **kwargs: Additional keyword arguments forwarded to
                ``Qwen3VLForConditionalGeneration.from_pretrained`` (e.g.
                ``revision``, ``cache_dir``, ``token``).

        Returns:
            Loaded ``Qwen3VLForConditionalGeneration`` instance with VTC
            weights applied.
        """
        # Pop HF download kwargs out so they reach every snapshot_download /
        # *.from_pretrained call below that actually hits the Hub. Leaving
        # them in ``kwargs`` would route them only into
        # ``Qwen3VLForConditionalGeneration.from_pretrained`` and break
        # ``_from_config`` (which takes model-init kwargs, not download
        # kwargs) on the VTC branch. Pinning ``revision`` here keeps the
        # weight blobs aligned with ``--model-revision``.
        download_kwargs = {k: kwargs.pop(k) for k in ("revision", "cache_dir", "token") if k in kwargs}
        revision = download_kwargs.pop("revision", None)

        if "vtc" in pretrained_model_name_or_path.lower():
            print(
                f"[i] VTC loading pretrained VTC + Qwen3VL weights from {pretrained_model_name_or_path}",
            )
            # Reference architecture config — always the upstream Qwen3-VL.
            # revision pins the RLDX repo, not this reference, so don't thread
            # download_kwargs here; pin to a fixed commit instead (lib.security rule 9).
            base_config = AutoConfig.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
                revision=_QWEN3_VL_REFERENCE_REVISION,
            )
        else:
            print(f"[i] VTC loading Qwen3-VL weights from {pretrained_model_name_or_path}")
            base_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                revision=revision,
                **download_kwargs,
            )

        # Inject motion module config into vision_config before model construction
        if motion_config is not None:
            for k, v in motion_config.items():
                setattr(base_config.vision_config, k, v)
            print(f"[i] motion module config injected into vision_config: {motion_config}")

        if "vtc" in pretrained_model_name_or_path.lower():
            model = Qwen3VLForConditionalGeneration._from_config(base_config, **kwargs)  # noqa: SLF001
        else:
            # Only pass explicit config when motion module modifies it; otherwise use default loading
            extra: dict[str, Any] = {"config": base_config} if motion_config is not None else {}
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path,
                **extra,
                revision=revision,
                **download_kwargs,
                **kwargs,
            )  # type: ignore[arg-type, misc]

        # Re-apply motion module init only when motion module is newly added (not in checkpoint).
        # from_pretrained's _init_weights overwrites kaiming Conv3d init and any
        # custom init applied in MotionModule.initialize_weights().
        if motion_config is not None and hasattr(model.model.visual, "motion_block"):
            probe = _checkpoint_has_motion_weights(pretrained_model_name_or_path, revision=revision)
            if probe is True:
                print(
                    "[i] motion module weights loaded from checkpoint, skipping re-initialization",
                )
            elif probe is False:
                model.model.visual.motion_block.initialize_weights()
                print("[i] motion module weights re-initialized (not found in checkpoint)")
            else:
                # Indeterminate: preserve whatever from_pretrained already
                # loaded. Re-initializing here risked overwriting trained
                # motion module weights when the probe couldn't reach the checkpoint
                # (missing index.json for non-sharded ckpts, HF download
                # failure, malformed json, etc.).
                print(
                    "[w] motion module ckpt probe indeterminate — skipping re-init to "
                    "avoid overwriting loaded weights. If fresh motion module init is "
                    "intended, verify the checkpoint layout.",
                )

        for layer_idx in range(len(model.model.language_model.layers)):
            model.model.language_model.layers[layer_idx] = LayerWrapper(
                model.model.language_model.layers[layer_idx],
                layer_idx=layer_idx,
                internal_projection=4,
                img_pattern=[151652],
                motion_token=1,
            )
        return model
