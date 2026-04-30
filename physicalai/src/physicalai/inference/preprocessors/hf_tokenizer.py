# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lambda preprocessor that wraps an arbitrary callable."""

from __future__ import annotations

import numpy as np

from physicalai.inference.constants import TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.preprocessors.base import Preprocessor


class HFTokenizer(Preprocessor):
    """Preprocessor that wraps a Hugging Face tokenizer.

    Args:
        tokenizer_name: Name of the Hugging Face tokenizer to use.
        revision: Revision of the tokenizer to use.
        max_token_len: Maximum token length for the tokenizer.

    Examples:
        >>> prep = HFTokenizer("bert-base-uncased", "main", 512)
        >>> outputs = prep({"task": ["Here is a sample text."]})
    """

    def __init__(self, tokenizer_name: str, revision: str, max_token_len: int = 512) -> None:
        """Initialize the HFTokenizer.

        Args:
            tokenizer_name: Name of the Hugging Face tokenizer to use.
            revision: Revision of the tokenizer to use.
            max_token_len: Maximum token length for the tokenizer.

        Raises:
            ImportError: If transformers library is not installed.
        """
        super().__init__()
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                revision=revision,
                use_fast=True,
                padding_side="right",
            )
        except ImportError as e:
            msg = "Tokenizer requires transformers. Install with: pip install transformers"
            raise ImportError(msg) from e

        self._max_token_len = max_token_len

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Tokenize input tasks and add tokenized prompts to the inputs dictionary.

        This method extracts task descriptions from the inputs, tokenizes them using
        the HuggingFace tokenizer, and adds the resulting token IDs and attention masks
        back to the inputs dictionary.

        Args:
            inputs: A dictionary containing input arrays, including a TASK key with
                    a list of task descriptions to tokenize.

        Returns:
            The modified inputs dictionary with two new keys:
                - TOKENIZED_PROMPT: numpy array of tokenized input IDs
                - TOKENIZED_PROMPT_MASK: numpy array of attention masks (boolean)
            The original TASK key is removed from the returned dictionary.

        Raises:
            TypeError: If TASK is not a list of strings.

        Note:
            - Tokens are padded/truncated to max_length
            - Padding is applied on the right side
            - The original TASK key is removed from the returned dictionary
        """
        batch_tasks = inputs[TASK]
        outputs = dict(inputs)
        outputs.pop(TASK)

        if not isinstance(batch_tasks, list):
            msg = f"Expected TASK to be a list of strings, got {type(batch_tasks)}"
            raise TypeError(msg)

        encoded_tokens = self._tokenizer(
            batch_tasks,
            max_length=self._max_token_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        outputs[TOKENIZED_PROMPT] = encoded_tokens["input_ids"]
        outputs[TOKENIZED_PROMPT_MASK] = encoded_tokens["attention_mask"].astype(np.bool)

        return outputs

    def __repr__(self) -> str:
        """Return string representation of the preprocessor."""
        return (
            f"{self.__class__.__name__}(tokenizer_name={self._tokenizer.name_or_path!r}, "
            f"revision={self._tokenizer.config.revision!r})"
        )
