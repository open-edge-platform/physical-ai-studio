# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.utils.action_cursor import ActionCursor


class TestActionCursor:
    def test_init_starts_empty_and_repr(self) -> None:
        cursor = ActionCursor()
        assert cursor.empty is True
        assert repr(cursor) == "ActionCursor(buffered=0)"

    def test_push_chunk_2d_pops_in_row_order(self) -> None:
        cursor = ActionCursor()
        chunk = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        )

        cursor.push_chunk(chunk)

        np.testing.assert_array_equal(cursor.pop(), np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(cursor.pop(), np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(cursor.pop(), np.array([5.0, 6.0], dtype=np.float32))
        assert cursor.empty is True

    def test_push_chunk_3d_pops_time_slices(self) -> None:
        cursor = ActionCursor()
        chunk = np.array(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            ],
            dtype=np.float32,
        )

        cursor.push_chunk(chunk)

        np.testing.assert_array_equal(
            cursor.pop(),
            np.array([[1.0, 10.0], [4.0, 40.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            cursor.pop(),
            np.array([[2.0, 20.0], [5.0, 50.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            cursor.pop(),
            np.array([[3.0, 30.0], [6.0, 60.0]], dtype=np.float32),
        )
        assert cursor.empty is True

    def test_pop_empty_raises_index_error(self) -> None:
        cursor = ActionCursor()
        with pytest.raises(IndexError, match="ActionCursor is empty; call push_chunk before pop"):
            cursor.pop()

    def test_reset_clears_buffered_actions(self) -> None:
        cursor = ActionCursor()
        cursor.push_chunk(np.array([[1.0], [2.0]], dtype=np.float32))

        cursor.reset()

        assert cursor.empty is True
        with pytest.raises(IndexError):
            cursor.pop()

    def test_multiple_pushes_append_in_fifo_order(self) -> None:
        cursor = ActionCursor()
        first = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        second = np.array(
            [
                [[3.0, 3.0], [4.0, 4.0]],
            ],
            dtype=np.float32,
        )

        cursor.push_chunk(first)
        cursor.push_chunk(second)

        np.testing.assert_array_equal(cursor.pop(), np.array([1.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(cursor.pop(), np.array([2.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(cursor.pop(), np.array([[3.0, 3.0]], dtype=np.float32))
        np.testing.assert_array_equal(cursor.pop(), np.array([[4.0, 4.0]], dtype=np.float32))
        assert cursor.empty is True
