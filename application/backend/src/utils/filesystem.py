# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileTree:
    """Safe traversal of a model or export directory."""

    root: Path

    def prune_empty_directories(self) -> None:
        """Remove empty directories bottom-up, including the root, without following symlinks."""
        self.resolve(Path())
        for directory, _, _ in self.root.walk(top_down=False, follow_symlinks=False):
            with suppress(OSError):
                directory.rmdir()

    def resolve(self, relative: Path) -> Path:
        """Resolve a child without following symbolic links or leaving the tree."""
        path = self.root / relative
        root = self.root.absolute()
        if not path.resolve().is_relative_to(root.resolve()) or any(
            parent.is_symlink() for parent in (path, *path.parents) if parent.absolute().is_relative_to(root)
        ):
            raise ValueError("Paths must stay inside their directory and must not contain symbolic links")
        return path

    def files(self, *, exclude: tuple[str, ...] = ()) -> Iterator[Path]:
        """Walk files, pruning excluded top-level entries and rejecting symlinks."""
        self.resolve(Path())
        for directory, children, names in self.root.walk(follow_symlinks=False):
            if directory == self.root:
                children[:] = [name for name in children if name not in exclude]
                names[:] = [name for name in names if name not in exclude]
            for name in (*children, *names):
                path = self.resolve((directory / name).relative_to(self.root))
                if path.is_file():
                    yield path
