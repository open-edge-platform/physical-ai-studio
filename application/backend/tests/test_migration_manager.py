from types import SimpleNamespace
from typing import Literal

import pytest

import db.migration as migration_module
from alembic.script.revision import ResolutionError
from db.migration import MigrationManager, RevisionNotFoundError
from settings import Settings


class _DummyConnectionContext:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        return False


class _DummyEngine:
    def connect(self) -> _DummyConnectionContext:
        return _DummyConnectionContext()


def test_check_migration_status_accepts_intermediate_revision(monkeypatch, tmp_path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path)
    manager = MigrationManager(settings)

    current_rev = "d4e5f6a7b8c9"
    head_rev = "e4b2f1c8a907"

    script = SimpleNamespace(
        get_current_head=lambda: head_rev,
        get_revision=lambda revision: object() if revision == current_rev else None,
    )

    context = SimpleNamespace(get_current_revision=lambda: current_rev)

    monkeypatch.setattr(migration_module, "sync_engine", _DummyEngine())
    monkeypatch.setattr(migration_module.ScriptDirectory, "from_config", lambda _: script)
    monkeypatch.setattr(migration_module.migration.MigrationContext, "configure", lambda _: context)

    needs_migration, status = manager.check_migration_status()

    assert needs_migration is True
    assert status == f"Current: {current_rev}, Head: {head_rev}"


def test_check_migration_status_raises_for_unknown_revision(monkeypatch, tmp_path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path)
    manager = MigrationManager(settings)

    missing_rev = "deadbeefdead"

    def _raise_missing_revision(_: str) -> None:
        raise ResolutionError("missing", missing_rev)

    script = SimpleNamespace(
        get_current_head=lambda: "e4b2f1c8a907",
        get_revision=_raise_missing_revision,
    )

    context = SimpleNamespace(get_current_revision=lambda: missing_rev)

    monkeypatch.setattr(migration_module, "sync_engine", _DummyEngine())
    monkeypatch.setattr(migration_module.ScriptDirectory, "from_config", lambda _: script)
    monkeypatch.setattr(migration_module.migration.MigrationContext, "configure", lambda _: context)

    with pytest.raises(RevisionNotFoundError, match=missing_rev):
        manager.check_migration_status()
