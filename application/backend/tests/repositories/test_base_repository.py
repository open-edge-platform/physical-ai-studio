"""Unit tests for the update-refresh guard in repositories/base.py.

Both ``BaseRepository.update`` and ``ProjectBaseRepository.update`` must
raise ``TypeError`` when the item has no usable ``id`` attribute, so the
caller gets a clear error instead of an ``AttributeError`` deep in
SQLAlchemy refresh logic.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from repositories.base import BaseRepository, ProjectBaseRepository

# ---------------------------------------------------------------------------
# Shared fake model — has id=None so the guard fires
# ---------------------------------------------------------------------------


class _NoIdModel:
    """Pydantic-like model without a usable id field."""

    id = None

    def model_copy(self, **_: object) -> "_NoIdModel":
        return self

    def model_dump(self) -> dict:
        return {}

    @classmethod
    def model_validate(cls, data: object) -> "_NoIdModel":
        return cls()


# ---------------------------------------------------------------------------
# Concrete repository subclasses (needed because the base classes are abstract)
# ---------------------------------------------------------------------------


class _StubBaseRepository(BaseRepository):
    @property
    def to_schema(self) -> object:
        return lambda x: MagicMock()

    @property
    def from_schema(self) -> object:
        return lambda x: x


class _StubProjectBaseRepository(ProjectBaseRepository):
    @property
    def to_schema(self) -> object:
        return lambda x: MagicMock()

    @property
    def from_schema(self) -> object:
        return lambda x: x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _mock_session() -> MagicMock:
    session = MagicMock()
    session.merge = AsyncMock()
    session.commit = AsyncMock()
    return session


def test_base_repository_update_raises_type_error_when_id_is_none() -> None:
    """BaseRepository.update must raise TypeError if item has no usable id."""
    repo = _StubBaseRepository(_mock_session(), MagicMock())

    with pytest.raises(TypeError, match="does not provide a usable `id`"):
        asyncio.run(repo.update(_NoIdModel(), {}))


def test_project_base_repository_update_raises_type_error_when_id_is_none() -> None:
    """ProjectBaseRepository.update must raise TypeError if item has no usable id."""
    repo = _StubProjectBaseRepository(_mock_session(), uuid4(), MagicMock())

    with pytest.raises(TypeError, match="does not provide a usable `id`"):
        asyncio.run(repo.update(_NoIdModel(), {}))
