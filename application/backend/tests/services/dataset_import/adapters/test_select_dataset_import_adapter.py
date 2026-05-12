"""Tests for select_dataset_import_adapter covering auto and explicit-hint modes."""

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING

from services.archive_safety import SafeZipArchive
from services.dataset_import.adapters import (
    DatasetAdapterSelectionResult,
    get_registered_dataset_import_adapters,
    select_dataset_import_adapter,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _open(path: Path) -> SafeZipArchive:
    return SafeZipArchive(path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Fixtures: valid v2 and v3 archives
# ---------------------------------------------------------------------------


def _v2_archive(tmp_path: Path) -> Path:
    path = tmp_path / "v2.zip"
    _write_zip(
        path,
        {
            "meta/info.json": json.dumps({"codebase_version": "v2.1"}).encode(),
            "meta/tasks.jsonl": b'{"task_index":0,"task":"pick"}\n',
            "meta/episodes.jsonl": b'{"episode_index":0,"tasks":["pick"],"length":1}\n',
            "data/chunk-000/episode_000000.parquet": b"PAR1",
        },
    )
    return path


def _v3_archive(tmp_path: Path) -> Path:
    path = tmp_path / "v3.zip"
    _write_zip(
        path,
        {
            "meta/info.json": json.dumps({"codebase_version": "v3.0"}).encode(),
            "meta/tasks.parquet": b"PAR1",
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )
    return path


def _empty_archive(tmp_path: Path) -> Path:
    path = tmp_path / "empty.zip"
    _write_zip(path, {"readme.txt": b"nothing useful"})
    return path


# ---------------------------------------------------------------------------
# Auto mode — successful matches
# ---------------------------------------------------------------------------


def test_auto_selects_v3_adapter_for_v3_archive(tmp_path: Path) -> None:
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_v3_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "auto", archive)
    assert isinstance(result, DatasetAdapterSelectionResult)
    assert result.adapter is not None
    assert result.adapter.source.value == "lerobot_v3"
    assert result.report is None


# ---------------------------------------------------------------------------
# Auto mode — no match
# ---------------------------------------------------------------------------


def test_auto_returns_none_adapter_when_no_adapter_matches(tmp_path: Path) -> None:
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_empty_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "auto", archive)
    assert result.adapter is None


def test_auto_no_match_report_has_generic_error(tmp_path: Path) -> None:
    """Auto-mode failure must surface a single generic error; no adapter-specific leakage."""
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_empty_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "auto", archive)
    assert result.report is not None
    assert len(result.report.messages) >= 1
    # Generic message should mention "format" or "adapter" — not internal adapter names
    combined = " ".join(m.message for m in result.report.messages).lower()
    assert "format" in combined or "adapter" in combined or "unsupported" in combined


def test_auto_no_match_does_not_raise(tmp_path: Path) -> None:
    """Auto mode must not raise even when no adapter matches."""
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_empty_archive(tmp_path))
    # Should not raise
    select_dataset_import_adapter(adapters, "auto", archive)


def test_auto_no_match_report_does_not_leak_per_adapter_details(tmp_path: Path) -> None:
    """Auto-mode failure message must not contain per-adapter mismatch details."""
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_empty_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "auto", archive)
    assert result.report is not None
    combined = " ".join(m.message for m in result.report.messages)
    # Adapter-specific markers like "episodes.jsonl" or "tasks.jsonl" must NOT appear
    assert "episodes.jsonl" not in combined
    assert "tasks.jsonl" not in combined


def test_auto_no_match_report_mentions_lerobot_formats(tmp_path: Path) -> None:
    """Generic no-match message should mention supported LeRobot versions."""
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_empty_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "auto", archive)
    assert result.report is not None
    combined = " ".join(m.message for m in result.report.messages).lower()
    assert "lerobot" in combined


def test_auto_no_match_report_mentions_metadata_or_data_layout(tmp_path: Path) -> None:
    """Generic no-match message should hint at required archive structure."""
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_empty_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "auto", archive)
    assert result.report is not None
    combined = " ".join(m.message for m in result.report.messages).lower()
    assert "meta" in combined or "data" in combined or "parquet" in combined


# ---------------------------------------------------------------------------
# Explicit hint mode — matching
# ---------------------------------------------------------------------------


def test_explicit_hint_returns_v3_adapter_for_v3_archive(tmp_path: Path) -> None:
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_v3_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "lerobot_v3", archive)
    assert result.adapter is not None
    assert result.adapter.source.value == "lerobot_v3"
    assert result.report is None


# ---------------------------------------------------------------------------
# Explicit hint mode — mismatch: returns report, does NOT raise
# ---------------------------------------------------------------------------


def test_explicit_hint_v3_mismatch_for_v2_archive_returns_no_adapter(tmp_path: Path) -> None:
    """When v3 is hinted but archive is v2, adapter is None (no raise)."""
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_v2_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "lerobot_v3", archive)
    assert result.adapter is None


def test_explicit_hint_v3_mismatch_report_includes_detect_details(tmp_path: Path) -> None:
    """Mismatch report should contain detect-report messages from the adapter."""
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_v2_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "lerobot_v3", archive)
    assert result.report is not None
    assert len(result.report.messages) >= 1
    combined = " ".join(m.message for m in result.report.messages).lower()
    # Adapter detect report should mention something v2-specific or lerobot-related
    assert "v2" in combined or "lerobot" in combined or "episodes.jsonl" in combined


def test_explicit_hint_v3_mismatch_does_not_raise(tmp_path: Path) -> None:
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_v2_archive(tmp_path))
    # Must not raise
    select_dataset_import_adapter(adapters, "lerobot_v3", archive)


def test_explicit_hint_v2_mismatch_for_v3_archive_returns_no_adapter(tmp_path: Path) -> None:
    """When v2 is hinted but archive is v3, adapter is None (no raise)."""
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_v3_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "lerobot_v2", archive)
    assert result.adapter is None


def test_explicit_hint_v2_unknown_hint_report_has_generic_message(tmp_path: Path) -> None:
    """'lerobot_v2' is not a registered adapter; the generic fallback message is returned."""
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_v3_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "lerobot_v2", archive)
    assert result.report is not None
    combined = " ".join(m.message for m in result.report.messages).lower()
    assert "supported" in combined or "format" in combined or "detected" in combined


# ---------------------------------------------------------------------------
# Explicit hint mode — unknown hint
# ---------------------------------------------------------------------------


def test_explicit_unknown_hint_returns_no_adapter(tmp_path: Path) -> None:
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_v3_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "unknown_format", archive)
    assert result.adapter is None


def test_explicit_unknown_hint_report_contains_not_recognized_message(tmp_path: Path) -> None:
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_v3_archive(tmp_path))
    result = select_dataset_import_adapter(adapters, "unknown_format", archive)
    assert result.report is not None
    combined = " ".join(m.message for m in result.report.messages).lower()
    # Generic fallback message must mention supported formats or detection failure
    assert "supported" in combined or "format" in combined or "detected" in combined


def test_explicit_unknown_hint_does_not_raise(tmp_path: Path) -> None:
    adapters = get_registered_dataset_import_adapters()
    archive = _open(_v3_archive(tmp_path))
    # Must not raise
    select_dataset_import_adapter(adapters, "unknown_format", archive)
