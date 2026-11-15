from starlette.exceptions import HTTPException
from starlette.responses import HTMLResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

_SO101_SCRIPT_TAG = '<script defer src="/api/so101/ui-extension.js"></script></head>'


class SPAStaticFiles(StaticFiles):
    """StaticFiles subclass that serves index.html for unknown paths and injects the SO-101 Architecture Workbench."""

    def _inject_so101_extension(self, response: Response) -> Response:
        content_type = response.headers.get("content-type", "")
        body = getattr(response, "body", None)
        if "text/html" in content_type and isinstance(body, (bytes, bytearray)):
            html = body.decode("utf-8", errors="ignore")
            if "/api/so101/ui-extension.js" not in html and "</head>" in html:
                html = html.replace("</head>", _SO101_SCRIPT_TAG, 1)
                headers = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower() not in ("content-length", "etag", "last-modified")
                }
                headers["Cache-Control"] = "no-store, max-age=0"
                return HTMLResponse(content=html, status_code=response.status_code, headers=headers)
        return response

    async def get_response(self, path: str, scope: Scope) -> Response:
        is_api_path = path == "api" or path.startswith("api/")
        try:
            response = await super().get_response(path, scope)

            if response.status_code == 404 and not is_api_path:
                response = await super().get_response("index.html", scope)

            if path in ("", "index.html") or (
                not is_api_path and "text/html" in response.headers.get("content-type", "")
            ):
                index_file = self.lookup_path("index.html")[0]
                from pathlib import Path

                p = Path(index_file)
                if p.is_file():
                    html = p.read_text(encoding="utf-8")
                    if "/api/so101/ui-extension.js" not in html and "</head>" in html:
                        html = html.replace("</head>", _SO101_SCRIPT_TAG, 1)
                    return HTMLResponse(
                        content=html, headers={"Cache-Control": "no-store, max-age=0"}
                    )

            return self._inject_so101_extension(response)
        except HTTPException as e:
            if e.status_code == 404 and not is_api_path:
                index_file = self.lookup_path("index.html")[0]
                from pathlib import Path

                p = Path(index_file)
                if p.is_file():
                    html = p.read_text(encoding="utf-8")
                    if "/api/so101/ui-extension.js" not in html and "</head>" in html:
                        html = html.replace("</head>", _SO101_SCRIPT_TAG, 1)
                    return HTMLResponse(
                        content=html, headers={"Cache-Control": "no-store, max-age=0"}
                    )
                return await super().get_response("index.html", scope)
            raise e
