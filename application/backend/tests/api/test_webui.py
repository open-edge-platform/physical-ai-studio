from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.webui import SPAStaticFiles

INDEX_HTML = b'<!DOCTYPE html><html><head><script defer src="/static/js/index.41ae3e4b67.js"></script></head></html>'


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Serve a UI build laid out the way RSBuild writes it."""
    (tmp_path / "index.html").write_bytes(INDEX_HTML)
    (tmp_path / "physicalai-studio-logo.svg").write_text("<svg/>")
    (tmp_path / "static" / "js").mkdir(parents=True)
    (tmp_path / "static" / "js" / "index.41ae3e4b67.js").write_text("console.log('ui');")
    (tmp_path / "static" / "js" / "index.41ae3e4b67.js.LICENSE.txt").write_text("MIT")
    (tmp_path / "static" / "font").mkdir()
    (tmp_path / "static" / "font" / "inter-v20-latin-variable.6c596dfcdd.woff2").write_bytes(b"font")

    app = FastAPI()
    app.mount("/", SPAStaticFiles(directory=tmp_path, html=True), name="webui")
    return TestClient(app)


@pytest.mark.parametrize("path", ["/", "/index.html"])
def test_index_html_must_be_revalidated(client: TestClient, path: str) -> None:
    """index.html names the bundle, so the browser must ask for it again or it keeps running an old UI."""
    response = client.get(path)

    assert response.status_code == 200
    assert response.content == INDEX_HTML
    assert response.headers["cache-control"] == "no-cache"


def test_spa_fallback_must_be_revalidated(client: TestClient) -> None:
    """A client-side route is answered with index.html, so it needs the same header."""
    response = client.get("/projects/123/robots")

    assert response.status_code == 200
    assert response.content == INDEX_HTML
    assert response.headers["cache-control"] == "no-cache"


def test_unchanged_index_html_answers_304_and_keeps_no_cache(client: TestClient) -> None:
    """The check costs a round trip, and the 304 still tells the browser to check next time."""
    etag = client.get("/").headers["etag"]

    response = client.get("/", headers={"If-None-Match": etag})

    assert response.status_code == 304
    assert response.content == b""
    assert response.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize(
    "path",
    [
        "/static/js/index.41ae3e4b67.js",
        "/static/js/index.41ae3e4b67.js.LICENSE.txt",
        "/static/font/inter-v20-latin-variable.6c596dfcdd.woff2",
    ],
)
def test_hashed_build_assets_are_immutable(client: TestClient, path: str) -> None:
    """A content-hashed asset never changes under its name, so the browser can keep it without asking."""
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_unhashed_files_must_be_revalidated(client: TestClient) -> None:
    """Files copied as they are (the favicon) keep their name across builds."""
    response = client.get("/physicalai-studio-logo.svg")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"


def test_missing_hashed_asset_falls_back_without_immutable(client: TestClient) -> None:
    """An old chunk name answered with index.html must never be pinned in the cache for a year."""
    response = client.get("/static/js/index.0000000000.js")

    assert response.status_code == 200
    assert response.content == INDEX_HTML
    assert response.headers["cache-control"] == "no-cache"


def test_api_paths_do_not_fall_back(client: TestClient) -> None:
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
