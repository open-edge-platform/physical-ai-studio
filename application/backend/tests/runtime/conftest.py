from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_runtime_locks(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep session lock files out of the developer's XDG runtime directory."""
    xdg = tmp_path / "xdg-runtime"
    xdg.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
