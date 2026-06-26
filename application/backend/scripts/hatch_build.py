import re
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

GITHUB_REPO = "https://github.com/open-edge-platform/physical-ai-studio"
RAW_REPO = "https://raw.githubusercontent.com/open-edge-platform/physical-ai-studio"
BRANCH = "main"

_RELATIVE_URL_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")
_IMG_SRC_RE = re.compile(r'(<img\s[^>]*src=")([^"]+)(")')


def _resolve_relative(url: str) -> str:
    if url.startswith("./"):
        return "application/" + url[2:]
    if url.startswith("../"):
        return url[3:]
    return url


def _make_link_absolute(match: re.Match) -> str:
    prefix = match.group(1)
    text = match.group(2)
    url = match.group(3)
    if url.startswith(("http://", "https://", "#", "mailto:")):
        return match.group(0)
    resolved = _resolve_relative(url)
    base = RAW_REPO if prefix == "!" else f"{GITHUB_REPO}/blob"
    return f"{prefix}[{text}]({base}/{BRANCH}/{resolved})"


def _make_img_src_absolute(match: re.Match) -> str:
    url = match.group(2)
    if url.startswith(("http://", "https://", "#")):
        return match.group(0)
    resolved = _resolve_relative(url)
    return f"{match.group(1)}{RAW_REPO}/{BRANCH}/{resolved}{match.group(3)}"


def _transform_readme(content: str) -> str:
    content = _RELATIVE_URL_RE.sub(_make_link_absolute, content)
    return _IMG_SRC_RE.sub(_make_img_src_absolute, content)


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if version == "editable":
            return

        ui_dist = Path(self.root).parent / "ui" / "dist"
        index_html = ui_dist / "index.html"

        if not index_html.exists():
            msg = (
                "Missing application/ui/dist/index.html. "
                "Build the UI before building the package, or run "
                "application/backend/scripts/build_package.sh."
            )
            raise FileNotFoundError(msg)

        build_data["force_include"] = {"../ui/dist": "webui"}

        app_readme = Path(self.root).parent / "README.md"
        target = Path(self.root) / "README.md"
        content = app_readme.read_text(encoding="utf-8")
        target.write_text(_transform_readme(content), encoding="utf-8")
