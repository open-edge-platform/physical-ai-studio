# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO tokenizer inference preprocessor."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from physicalai.inference.preprocessors import Preprocessor
from physicalai.inference.adapters import OpenVINOAdapter
from physicalai.inference.constants import TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK


class OVTokenizer(Preprocessor):
    """Preprocessor that wraps an OpenVINO tokenizer.

    Args:
        artifact: Path to the OpenVINO tokenizer artifact.
    """

    def __init__(self, artifact: str | Path) -> None:
        """Initialize the OVTokenizer.

        Args:
            artifact: Path to the OpenVINO tokenizer artifact.
        """
        super().__init__()
        self._artifact = Path(artifact)
        # OV Tokenizers support CPU only:
        # https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#usage
        self._adapter = OpenVINOAdapter(device="CPU")
        self._adapter.load(self._artifact)

        if len(self._adapter.input_names) != 1:
            raise ValueError("Expected exactly one input for the OV tokenizer, "
                             f"but got {len(self._adapter.input_names)}")
        self._input_name = self._adapter.input_names[0]

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Tokenize inputs using the OpenVINO tokenizer.

        Args:
            inputs: Dictionary of input arrays containing the key `TASK`.

        Returns:
            Dictionary with tokenized outputs.
        """
        batch_tasks = inputs[TASK]
        outputs = dict(inputs)
        outputs.pop(TASK)

        if not isinstance(batch_tasks, list):
            msg = f"Expected TASK to be a list of strings, got {type(batch_tasks)}"
            raise TypeError(msg)

        adapter_output = self._adapter.predict({self._input_name: batch_tasks})
        outputs[TOKENIZED_PROMPT] = adapter_output["input_ids"]
        outputs[TOKENIZED_PROMPT_MASK] = adapter_output["attention_mask"].astype(np.bool)

        return outputs

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"{self.__class__.__name__}(artifact={self._artifact!r})"
