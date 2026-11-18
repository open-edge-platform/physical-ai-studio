# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from utils.filesystem import FileTree


def test_walk_only_excludes_explicit_top_level_entries(tmp_path: Path) -> None:
    for relative in ("model.bin", ".tokenizer", "variants/int8/model.bin", "assets/variants/config.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data")
    assert {path.relative_to(tmp_path).as_posix() for path in FileTree(tmp_path).files(exclude=("variants",))} == {
        "model.bin",
        ".tokenizer",
        "assets/variants/config.json",
    }


@pytest.mark.parametrize("kind", ["file", "directory", "broken", "internal"])
def test_walk_rejects_all_symlink_targets(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "model"
    root.mkdir()
    target = tmp_path / "target"
    if kind == "file":
        target.write_text("outside")
    elif kind == "directory":
        target.mkdir()
        (target / "weights").write_text("outside")
    elif kind == "internal":
        target = root / "weights"
        target.write_text("inside")
    (root / "link").symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises(ValueError, match="symbolic"):
        list(FileTree(root).files())


def test_resolving_a_future_path_does_not_create_it(tmp_path: Path) -> None:
    path = FileTree(tmp_path).resolve(Path("exports/openvino/variants/int8"))
    assert path == tmp_path / "exports/openvino/variants/int8"
    assert not path.exists()


def test_resolve_rejects_paths_outside_the_root(tmp_path: Path) -> None:
    for relative in (Path("../outside"), tmp_path.parent / "outside"):
        with pytest.raises(ValueError, match="inside"):
            FileTree(tmp_path).resolve(relative)


def test_walk_rejects_a_symlinked_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "model"
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        list(FileTree(root).files())


def test_pruning_removes_only_empty_directories(tmp_path: Path) -> None:
    root = tmp_path / "backend"
    (root / "variants/empty/nested").mkdir(parents=True)
    (root / "assets").mkdir()
    weights = root / "assets/weights.bin"
    weights.write_bytes(b"keep")
    FileTree(root).prune_empty_directories()
    assert not (root / "variants").exists()
    assert weights.read_bytes() == b"keep"
    weights.unlink()
    FileTree(root).prune_empty_directories()
    assert not root.exists()


def test_pruning_never_follows_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "outside"
    (target / "empty").mkdir(parents=True)
    root = tmp_path / "backend"
    root.mkdir()
    (root / "linked").symlink_to(target, target_is_directory=True)
    FileTree(root).prune_empty_directories()
    assert (target / "empty").is_dir()
    assert (root / "linked").is_symlink()
    with pytest.raises(ValueError, match="symbolic"):
        FileTree(root / "linked").prune_empty_directories()
