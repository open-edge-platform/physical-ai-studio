# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer helpers for MolmoAct2 preprocessing."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from transformers import Qwen2Tokenizer

from physicalai.policies.molmoact2.constants import MOLMOACT2_TOKENIZER_REPO_ID, MOLMOACT2_TOKENIZER_REVISION

_TOKENIZER_JSON_FILENAME = "tokenizer.json"
_MIN_TOKEN_LEN = 2
_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")

logger = logging.getLogger(__name__)


class MolmoAct2Tokenizers:
    """Lazy tokenizer utilities used by the MolmoAct2 preprocessor.

    Steps:
        1. Validate and retain the local tokenizer assets.
        2. Lazily load the Qwen tokenizer when first needed.
        3. Tokenize, truncate, and pad prompt text.
        4. Insert BOS while preserving valid-token attention masks.
        5. Expose the tokenizer for OpenVINO conversion.
    """

    def __init__(
        self,
        *,
        tokenizer_name_or_path: str,
        tokenizer_revision: str | None = None,
        max_token_len: int = 256,
        padding: Literal["max_length", "longest"] = "max_length",
        tokenizer_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize tokenizer helpers from a local tokenizer directory.

        Raises:
            ValueError: If max token length is less than two.
        """
        if max_token_len < _MIN_TOKEN_LEN:
            msg = "max_token_len must be at least 2 to reserve space for BOS."
            raise ValueError(msg)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer_revision = tokenizer_revision
        self.max_token_len = max_token_len
        self.padding = padding
        self.tokenizer_config = tokenizer_config or {}
        self._tokenizer: Qwen2Tokenizer | None = None
        self._tokenizer_file: str | None = None
        self._tokenizer_dir = self._resolve_tokenizer_dir()

    def _resolve_tokenizer_dir(self) -> str:
        """Resolve local tokenizer assets or download the pinned default tokenizer.

        Returns:
            The validated tokenizer directory.

        Raises:
            FileNotFoundError: If tokenizer.json is unavailable locally and cannot be downloaded.
            ValueError: If the tokenizer revision is not a full commit SHA.
        """
        local_path = Path(self.tokenizer_name_or_path)
        if local_path.is_file() and local_path.suffix.lower() == ".json":
            self._tokenizer_file = str(local_path)
            return str(local_path.parent)
        if local_path.is_dir() and (local_path / _TOKENIZER_JSON_FILENAME).is_file():
            return str(local_path)

        if self.tokenizer_name_or_path != MOLMOACT2_TOKENIZER_REPO_ID:
            logger.warning(
                "MolmoAct2 tokenizer.json was not found at %s; downloading the pinned default tokenizer.",
                local_path,
            )
        revision = self.tokenizer_revision or MOLMOACT2_TOKENIZER_REVISION
        if _COMMIT_SHA_PATTERN.fullmatch(revision) is None:
            msg = f"MolmoAct2 tokenizer revision must be a full 40-character commit SHA, got {revision!r}."
            raise ValueError(msg)
        try:
            tokenizer_path = Path(
                hf_hub_download(
                    repo_id=MOLMOACT2_TOKENIZER_REPO_ID,
                    filename=_TOKENIZER_JSON_FILENAME,
                    revision=revision,
                ),
            )
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as error:
            msg = (
                "MolmoAct2 tokenizer.json is unavailable locally and could not be downloaded from "
                f"{MOLMOACT2_TOKENIZER_REPO_ID}@{revision}. Supply a valid tokenizer_json_path."
            )
            logger.warning(msg)
            raise FileNotFoundError(msg) from error
        return str(tokenizer_path.parent)

    def _qwen_tokenizer(self) -> Qwen2Tokenizer:
        if self._tokenizer is None:
            tokenizer_config = dict(self.tokenizer_config)
            if self._tokenizer_file is not None:
                tokenizer_config["tokenizer_file"] = self._tokenizer_file
            self._tokenizer = Qwen2Tokenizer.from_pretrained(  # nosec: B615
                self._tokenizer_dir,
                local_files_only=True,
                **tokenizer_config,
            )
        if self._tokenizer is None:
            msg = "Tokenizer initialization failed"
            raise RuntimeError(msg)
        return self._tokenizer

    @property
    def tokenizer(self) -> Qwen2Tokenizer:
        """The tokenizer used for OpenVINO conversion."""
        return self._qwen_tokenizer()

    @property
    def pad_token_id(self) -> int:
        """The configured tokenizer padding token ID.

        Raises:
            TypeError: If the tokenizer does not define a pad token ID.
        """
        pad_token_id = self._qwen_tokenizer().pad_token_id
        if not isinstance(pad_token_id, int):
            msg = "Tokenizer must define a pad token ID."
            raise TypeError(msg)
        return pad_token_id

    @staticmethod
    def _insert_bos(
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        bos_token_id: int,
        pad_token_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]
            squeeze = True
        else:
            squeeze = False
        batch_size, seq_len = input_ids.shape
        valid_rows = [input_ids[index][attention_mask[index].astype(bool)] for index in range(batch_size)]
        if all(row.size > 0 and int(row[0]) == bos_token_id for row in valid_rows):
            return (input_ids[0], attention_mask[0]) if squeeze else (input_ids, attention_mask)
        out_ids = np.full((batch_size, seq_len + 1), pad_token_id, dtype=input_ids.dtype)
        out_mask = np.zeros((batch_size, seq_len + 1), dtype=attention_mask.dtype)
        for batch_idx, row_ids in enumerate(valid_rows):
            row_tokens = row_ids
            if row_tokens.size == 0 or int(row_tokens[0]) != bos_token_id:
                row_tokens = np.concatenate((np.asarray([bos_token_id], dtype=input_ids.dtype), row_tokens))
            out_ids[batch_idx, : row_tokens.size] = row_tokens
            out_mask[batch_idx, : row_tokens.size] = 1
        return (out_ids[0], out_mask[0]) if squeeze else (out_ids, out_mask)

    def tokenize_prompts(
        self,
        prompt_texts: list[str],
        *,
        padding: Literal["max_length", "longest"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompt text and ensure BOS insertion.

        Returns:
            Input IDs and attention-mask tensors.

        Raises:
            TypeError: If tokenizer special token IDs are missing.
        """
        resolved_padding = self.padding if padding is None else padding
        tokenizer = self._qwen_tokenizer()
        text_inputs = tokenizer(
            prompt_texts,
            max_length=self.max_token_len - 1,
            truncation=True,
            padding=resolved_padding,
        )
        input_ids = np.asarray(text_inputs["input_ids"])
        attention_mask = np.asarray(text_inputs["attention_mask"])
        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        if not isinstance(bos_token_id, int) or not isinstance(pad_token_id, int):
            msg = "Tokenizer must define BOS/EOS and pad token IDs."
            raise TypeError(msg)
        input_ids, attention_mask = self._insert_bos(input_ids, attention_mask, bos_token_id, pad_token_id)
        pad_width = self.max_token_len - input_ids.shape[-1]
        if resolved_padding == "max_length" and pad_width > 0:
            input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)), constant_values=pad_token_id)
            attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)), constant_values=0)
        return torch.as_tensor(input_ids), torch.as_tensor(attention_mask)
