# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from physicalai.inference.constants import TASK
from physicalai.inference.preprocessors import NewLinePreprocessor, Preprocessor


class TestNewLinePreprocessorInit:
    def test_is_preprocessor(self) -> None:
        prep = NewLinePreprocessor()
        assert isinstance(prep, Preprocessor)


class TestNewLinePreprocessorCall:
    def test_appends_newline(self) -> None:
        prep = NewLinePreprocessor()
        inputs = {TASK: ["pick up the cup"]}
        result = prep(inputs)
        assert result[TASK] == ["pick up the cup\n"]

    def test_does_not_double_newline(self) -> None:
        prep = NewLinePreprocessor()
        inputs = {TASK: ["pick up the cup\n"]}
        result = prep(inputs)
        assert result[TASK] == ["pick up the cup\n"]

    def test_multiple_tasks(self) -> None:
        prep = NewLinePreprocessor()
        inputs = {TASK: ["hello", "world\n", "test"]}
        result = prep(inputs)
        assert result[TASK] == ["hello\n", "world\n", "test\n"]

    def test_empty_string(self) -> None:
        prep = NewLinePreprocessor()
        inputs = {TASK: [""]}
        result = prep(inputs)
        assert result[TASK] == ["\n"]

    def test_preserves_other_keys(self) -> None:
        prep = NewLinePreprocessor()
        inputs = {TASK: ["do something"], "other": 42}
        result = prep(inputs)
        assert result["other"] == 42

    def test_non_list_task_raises(self) -> None:
        prep = NewLinePreprocessor()
        inputs = {TASK: "not a list"}
        with pytest.raises(TypeError, match="Expected TASK to be a list"):
            prep(inputs)

    def test_non_string_element_raises(self) -> None:
        prep = NewLinePreprocessor()
        inputs = {TASK: [123]}
        with pytest.raises(TypeError, match="Expected TASK to be a string"):
            prep(inputs)
