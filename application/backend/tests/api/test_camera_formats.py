from unittest.mock import patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from api.camera import router


@pytest.mark.parametrize("fingerprint", ["", "not JSON", "null", "[]", "{}"])
def test_supported_formats_rejects_invalid_fingerprints(fingerprint: str) -> None:
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client, patch("api.camera._query_formats") as query_formats:
        response = client.get("/api/cameras/supported_formats/usb_camera", params={"fingerprint": fingerprint})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "fingerprint" in response.json()["detail"]
    query_formats.assert_not_called()
