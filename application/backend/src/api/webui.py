import os
import re

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import PathLike, StaticFiles
from starlette.types import Scope

# RSBuild writes every bundled asset under ``static/`` with a content hash in its name
# (``static/js/index.41ae3e4b67.js``). A new build gets new names, so these never change in place.
_HASHED_ASSET_NAME = re.compile(r"\.[0-9a-f]{8,}\.")

# The browser may reuse a hashed asset without asking again: its name fixes its content.
IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"
# The browser must ask before reusing anything else (index.html, the favicon). While the file is
# unchanged the answer is a 304, so the check is cheap and an update is seen on the next load.
REVALIDATE_CACHE_CONTROL = "no-cache"


class SPAStaticFiles(StaticFiles):
    """StaticFiles subclass that serves index.html for unknown paths.

    Serves actual files when they exist (JS, CSS, URDF models, etc.)
    and falls back to index.html for everything else, allowing the
    frontend router to handle client-side routes.

    The `/api` namespace is excluded from the fallback: a missing API route or
    a defined route hit with an unsupported method should surface its own 404/405,
    not the SPA shell with a misleading 200.

    Every file is sent with a Cache-Control header. Without one, browsers guess a
    freshness time from Last-Modified and can keep an old index.html, and with it
    the old UI bundle, for hours after Studio is updated.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        is_api_path = path == "api" or path.startswith("api/")
        try:
            response = await super().get_response(path, scope)

            if response.status_code == 404 and not is_api_path:
                response = await super().get_response("index.html", scope)

            return response
        except HTTPException as e:
            if e.status_code == 404 and not is_api_path:
                return await super().get_response("index.html", scope)
            raise e

    def file_response(
        self,
        full_path: PathLike,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        # Every file goes through here: 200 and 304 answers, and index.html served as the SPA fallback.
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = self._cache_control(full_path)
        return response

    def _cache_control(self, full_path: PathLike) -> str:
        if self.directory is None:
            return REVALIDATE_CACHE_CONTROL
        relative = os.path.relpath(os.fspath(full_path), os.path.realpath(self.directory))
        parts = relative.split(os.sep)
        if parts[0] == "static" and _HASHED_ASSET_NAME.search(parts[-1]):
            return IMMUTABLE_CACHE_CONTROL
        return REVALIDATE_CACHE_CONTROL
