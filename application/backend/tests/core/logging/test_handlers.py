# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator

import pytest
from loguru import logger

from core.logging.handlers import LoggerStdoutWriter


@pytest.fixture
def captured_logs() -> Generator[list[str], None, None]:
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    yield messages
    logger.remove(sink_id)


def test_write_strips_ansi_cursor_codes(captured_logs: list[str]) -> None:
    # tqdm emits "\x1b[A" cursor-up codes when redrawing multi-line progress bars.
    LoggerStdoutWriter(level="WARNING").write("\x1b[A\x1b[A")

    assert captured_logs == []


def test_write_keeps_final_carriage_return_segment(captured_logs: list[str]) -> None:
    # Progress bars redraw in place with carriage returns; keep only the last frame.
    LoggerStdoutWriter(level="WARNING").write(
        "\rfile.parquet:   0%| | 0.00B / 46.6kB\rfile.parquet: 100%|#####| 46.6kB / 46.6kB",
    )

    assert captured_logs == ["file.parquet: 100%|#####| 46.6kB / 46.6kB"]


def test_write_skips_empty_messages(captured_logs: list[str]) -> None:
    LoggerStdoutWriter(level="WARNING").write("\n")
    LoggerStdoutWriter(level="WARNING").write("   ")

    assert captured_logs == []


def test_write_logs_plain_message(captured_logs: list[str]) -> None:
    LoggerStdoutWriter(level="INFO").write("hello world\n")

    assert captured_logs == ["hello world"]
