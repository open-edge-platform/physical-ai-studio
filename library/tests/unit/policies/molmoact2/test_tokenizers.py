# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MolmoAct2 tokenizer utilities."""

from pathlib import Path
from shutil import copyfile
from unittest.mock import Mock

import numpy as np
import openvino as ov
import pytest
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from openvino_tokenizers import convert_tokenizer

from physicalai.policies.molmoact2.processors.tokenizers import MolmoAct2Tokenizers

_MOLMOACT2_REPOSITORY = "allenai/MolmoAct2"
_MOLMOACT2_REVISION = "e432d85f6e039edca44afb93c262f3084ab72a9c"


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


def test_accepts_explicit_tokenizer_json_path(tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer_path = tokenizer_dir / "custom-tokenizer.json"
    (tokenizer_dir / "tokenizer.json").rename(tokenizer_path)
    loader = Mock(return_value=StubTokenizer())
    download = Mock()
    monkeypatch.setattr(
        "physicalai.policies.molmoact2.processors.tokenizers.Qwen2Tokenizer.from_pretrained",
        loader,
    )
    monkeypatch.setattr("physicalai.policies.molmoact2.processors.tokenizers.hf_hub_download", download)

    tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(tokenizer_path))
    tokenizers._qwen_tokenizer()

    assert tokenizers._tokenizer_dir == str(tokenizer_dir)
    loader.assert_called_once_with(
        str(tokenizer_dir),
        local_files_only=True,
        tokenizer_file=str(tokenizer_path),
    )
    download.assert_not_called()


def test_downloads_pinned_default_when_local_tokenizer_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    downloaded_path = tmp_path / "download" / "tokenizer.json"
    downloaded_path.parent.mkdir()
    downloaded_path.write_text("{}", encoding="utf-8")
    download = Mock(return_value=str(downloaded_path))
    monkeypatch.setattr("physicalai.policies.molmoact2.processors.tokenizers.hf_hub_download", download)

    tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(tmp_path / "missing"))

    assert tokenizers._tokenizer_dir == str(downloaded_path.parent)
    download.assert_called_once_with(
        repo_id=_MOLMOACT2_REPOSITORY,
        filename="tokenizer.json",
        revision=_MOLMOACT2_REVISION,
    )
    assert "downloading the pinned default tokenizer" in caplog.text


def test_missing_tokenizer_reports_local_override_when_download_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "physicalai.policies.molmoact2.processors.tokenizers.hf_hub_download",
        Mock(side_effect=LocalEntryNotFoundError("offline")),
    )

    with pytest.raises(FileNotFoundError, match="Supply a valid tokenizer_json_path"):
        MolmoAct2Tokenizers(tokenizer_name_or_path=str(tmp_path / "missing"))


@pytest.mark.parametrize("revision", ["main", "e432d85"])
def test_rejects_unpinned_revision_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revision: str,
) -> None:
    download = Mock()
    monkeypatch.setattr("physicalai.policies.molmoact2.processors.tokenizers.hf_hub_download", download)

    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        MolmoAct2Tokenizers(
            tokenizer_name_or_path=str(tmp_path / "missing"),
            tokenizer_revision=revision,
        )

    download.assert_not_called()


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


@pytest.fixture(scope="module")
def real_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    download_dir = tmp_path_factory.mktemp("molmoact2-download")
    tokenizer_path = hf_hub_download(
        repo_id=_MOLMOACT2_REPOSITORY,
        filename="tokenizer.json",
        revision=_MOLMOACT2_REVISION,
        local_dir=download_dir,
    )
    tokenizer_dir = tmp_path_factory.mktemp("molmoact2-tokenizer")
    copyfile(tokenizer_path, tokenizer_dir / "tokenizer.json")
    return tokenizer_dir


@pytest.mark.integration
@pytest.mark.requires_download
def test_real_molmoact2_tokenizer_tokenizes_prompts(real_tokenizer_dir: Path) -> None:
    tokenizers = MolmoAct2Tokenizers(
        tokenizer_name_or_path=str(real_tokenizer_dir),
        max_token_len=32,
    )

    input_ids, attention_mask = tokenizers.tokenize_prompts(["pick up the cube", "move left"])
    tokenizer = tokenizers._qwen_tokenizer()

    assert input_ids.shape == attention_mask.shape == (2, 32)
    assert input_ids.dtype == torch.int64
    assert attention_mask.dtype == torch.int64
    assert tokenizer.bos_token_id is None
    assert isinstance(tokenizer.eos_token_id, int)
    assert input_ids[:, 0].tolist() == [tokenizer.eos_token_id] * 2
    assert attention_mask[:, 0].tolist() == [1, 1]
    assert tokenizers.tokenizer is tokenizers.tokenizer


@pytest.mark.integration
@pytest.mark.requires_download
def test_real_molmoact2_openvino_tokenizer_matches_huggingface(real_tokenizer_dir: Path) -> None:
    tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(real_tokenizer_dir))
    tokenizer = tokenizers._qwen_tokenizer()
    source_tokens = tokenizer.added_tokens_decoder
    export_tokens = tokenizers.tokenizer.added_tokens_decoder
    prompts = [
        "<|image|><state_start><state_0><state_255><state_end><action_output>",
        "<setup_start>tabletop<setup_end> <control_start>joint<control_end> <action_0> <extra_0>",
        " ".join(["token"] * 100),
    ]
    expected = tokenizer(
        prompts,
        max_length=16,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    model = convert_tokenizer(
        tokenizers.tokenizer,
        with_detokenizer=False,
        max_length=16,
        use_max_padding=True,
        truncation=True,
    )
    compiled = ov.Core().compile_model(model, "CPU")
    actual = compiled({compiled.inputs[0]: prompts})
    outputs = {port.get_any_name(): value for port, value in actual.items()}

    assert any(token.content.startswith("<action_") for token in source_tokens.values())
    assert any(token.content.startswith("<extra_") for token in source_tokens.values())
    assert export_tokens == source_tokens
    np.testing.assert_array_equal(outputs["input_ids"], expected["input_ids"])
    np.testing.assert_array_equal(outputs["attention_mask"], expected["attention_mask"])
