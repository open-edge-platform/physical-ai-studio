# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MolmoAct2 tokenizer utilities."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from physicalai.policies.molmoact2.processors.tokenizers import MolmoAct2Tokenizers


class StubTokenizer:
    bos_token_id = 9
    eos_token_id = 8
    pad_token_id = 0

    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
        width = int(kwargs["max_length"]) if kwargs["padding"] == "max_length" else 2  # type: ignore[call-overload]
        return {
            "input_ids": [[5, 6, *([0] * (width - 2))] for _ in prompts],
            "attention_mask": [[1, 1, *([0] * (width - 2))] for _ in prompts],
        }


def test_loads_local_tokenizer_once(tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loader = Mock(return_value=StubTokenizer())
    monkeypatch.setattr(
        "physicalai.policies.molmoact2.processors.tokenizers.Qwen2Tokenizer.from_pretrained",
        loader,
    )
    tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(tokenizer_dir))

    assert tokenizers._qwen_tokenizer() is tokenizers._qwen_tokenizer()
    loader.assert_called_once_with(str(tokenizer_dir), local_files_only=True)


@pytest.mark.parametrize(("padding", "width"), [("max_length", 6), ("longest", 3)])
def test_tokenization_inserts_bos(
    tokenizer_dir: Path,
    padding: str,
    width: int,
) -> None:
    tokenizers = MolmoAct2Tokenizers(
        tokenizer_name_or_path=str(tokenizer_dir),
        max_token_len=6,
        padding=padding,  # type: ignore[arg-type]
    )
    tokenizers._tokenizer = StubTokenizer()  # type: ignore[assignment]

    input_ids, attention_mask = tokenizers.tokenize_prompts(["task"])

    assert input_ids.shape == attention_mask.shape == (1, width)
    assert input_ids[0, 0].item() == 9


def test_requires_local_tokenizer_assets(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        MolmoAct2Tokenizers(tokenizer_name_or_path=str(tmp_path))
