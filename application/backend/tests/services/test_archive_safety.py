from pathlib import Path

from services.archive_safety import flatten_single_root_directory


def test_flatten_single_root_directory_moves_nested_contents(tmp_path: Path) -> None:
    (tmp_path / "dataset").mkdir(parents=True)
    (tmp_path / "dataset" / "meta").mkdir(parents=True)
    (tmp_path / "dataset" / "data").mkdir(parents=True)
    (tmp_path / "dataset" / "meta" / "info.json").write_text("{}")

    flatten_single_root_directory(tmp_path)

    assert (tmp_path / "meta" / "info.json").exists()
    assert (tmp_path / "data").exists()
    assert not (tmp_path / "dataset").exists()


def test_flatten_single_root_directory_noop_when_multiple_roots(tmp_path: Path) -> None:
    (tmp_path / "dataset-a").mkdir(parents=True)
    (tmp_path / "dataset-b").mkdir(parents=True)

    flatten_single_root_directory(tmp_path)

    assert (tmp_path / "dataset-a").exists()
    assert (tmp_path / "dataset-b").exists()


def test_flatten_single_root_directory_ignores_macos_junk(tmp_path: Path) -> None:
    # macOS Finder "Compress" adds a __MACOSX sidecar dir and .DS_Store/._* files
    # alongside the real single-root dataset directory.
    (tmp_path / "dataset" / "meta").mkdir(parents=True)
    (tmp_path / "dataset" / "meta" / "info.json").write_text("{}")
    (tmp_path / "__MACOSX" / "dataset").mkdir(parents=True)
    (tmp_path / "__MACOSX" / "dataset" / "._info.json").write_text("junk")
    (tmp_path / ".DS_Store").write_text("junk")
    (tmp_path / "._dataset").write_text("junk")

    flatten_single_root_directory(tmp_path)

    assert (tmp_path / "meta" / "info.json").exists()
    assert not (tmp_path / "dataset").exists()
    assert not (tmp_path / "__MACOSX").exists()
    assert not (tmp_path / ".DS_Store").exists()
    assert not (tmp_path / "._dataset").exists()
