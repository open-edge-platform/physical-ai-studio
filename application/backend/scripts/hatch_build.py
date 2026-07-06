import re
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

GITHUB_REPO = "https://github.com/open-edge-platform/physical-ai-studio"
RAW_REPO = "https://raw.githubusercontent.com/open-edge-platform/physical-ai-studio"
BRANCH = "main"

_RELATIVE_URL_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")
_IMG_SRC_RE = re.compile(r'(<img\s[^>]*src=")([^"]+)(")')
_REQUIRED_ROBOT_ASSET_PATHS = (
    Path("SO101/so101_new_calib.urdf"),
    Path("widowx/urdf/generated/wxai/wxai_follower.urdf"),
    Path("widowx/urdf/generated/stationary_ai.urdf"),
)


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

        robot_assets_root = Path(self.root) / "src" / "static" / "robot-assets"
        missing_assets = [
            str(relative_path)
            for relative_path in _REQUIRED_ROBOT_ASSET_PATHS
            if not (robot_assets_root / relative_path).exists()
        ]
        if missing_assets:
            msg = (
                "Missing required robot assets in application/backend/src/static/robot-assets: "
                + ", ".join(missing_assets)
                + ". Run 'physicalai-studio sync-robot-assets' before building the package, "
                "or run application/backend/scripts/build_package.sh."
            )
            raise FileNotFoundError(msg)

        force_include = {
            "../ui/dist": "webui",
        }

        # When building from a git checkout, Hatch respects .gitignore and would
        # skip src/static/robot-assets by default. In Docker/CI contexts without
        # .git metadata, these files are already discovered under src and forcing
        # them again would create duplicate archive entries.
        if (Path(self.root).parent.parent / ".git").exists():
            force_include["src/static/robot-assets"] = "static/robot-assets"

        build_data["force_include"] = force_include

        app_readme = Path(self.root).parent / "README.md"
        target = Path(self.root) / "README.md"
        content = app_readme.read_text(encoding="utf-8")
        target.write_text(_transform_readme(content), encoding="utf-8")
