# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Install a per-user physicalai-streamdeck://start handler; no administrator access needed."""

# Commands use fixed OS utilities and saved installation paths, never commands from browser links.
# ruff: noqa: S603, S607

from __future__ import annotations

import argparse
import contextlib
import json
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

SCHEME = "physicalai-streamdeck"
LINK = f"{SCHEME}://start"
APP_NAME = "Physical AI Stream Deck"
DESKTOP_NAME = "physicalai-streamdeck.desktop"


def install_dir() -> Path:
    """Use persistent per-user application data, never the temporary uv script cache."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "physicalai" / "streamdeck"


def validate_backend(url: str) -> str:
    """Only installation chooses the backend; protocol links cannot redirect it."""
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or parts.path not in {"", "/"}
        or any(character.isspace() for character in url)
    ):
        raise ValueError("Backend URL must be an http(s) origin without credentials, path, query or fragment.")
    _ = parts.port
    return url.rstrip("/")


def desktop_argument(value: str) -> str:
    """Quote an argument for Desktop Entry Exec syntax, which is not POSIX shell syntax."""
    for character in ("\\", '"', "`", "$"):
        value = value.replace(character, "\\" + character)
    return '"' + value.replace("\\", "\\\\").replace("%", "%%") + '"'


def register(root: Path, uv: str) -> None:
    """Register a fixed launcher command with this user's OS URL handler registry."""
    command = [uv, "run", "--no-project", str(root / "host_launcher.py"), "--open"]
    if sys.platform == "darwin":
        app = Path.home() / "Applications" / f"{APP_NAME}.app"
        app.parent.mkdir(parents=True, exist_ok=True)
        prefix = shlex.join(command) + " "
        suffix = " >> " + shlex.quote(str(root / "launcher.log")) + " 2>&1 &"

        def apple_string(value: str) -> str:
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

        source = (
            "on open location launchURL\n"
            f"do shell script {apple_string(prefix)} & quoted form of launchURL & {apple_string(suffix)}\n"
            "end open location\n"
        )
        subprocess.run(["/usr/bin/osacompile", "-o", str(app), "-e", source], check=True)
        info = app / "Contents" / "Info.plist"
        with info.open("rb") as file:
            metadata = plistlib.load(file)
        metadata.update(
            {
                "CFBundleIdentifier": "org.physicalai.streamdeck-launcher",
                "LSUIElement": True,
                "NSMicrophoneUsageDescription": "Dictate task prompts locally for Physical AI Studio inference.",
                "CFBundleURLTypes": [{"CFBundleURLName": APP_NAME, "CFBundleURLSchemes": [SCHEME]}],
            }
        )
        with info.open("wb") as file:
            plistlib.dump(metadata, file)
        # osacompile signs the app; changing URL metadata invalidates that signature.
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(app)], check=True)
        subprocess.run(
            [
                "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister",
                "-f",
                str(app),
            ],
            check=True,
        )
    elif sys.platform == "win32":
        import winreg

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{SCHEME}") as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, f"URL:{APP_NAME}")
            winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{SCHEME}\shell\open\command") as key:
            # Always quote %1, even though list2cmdline would leave it unquoted without spaces.
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, subprocess.list2cmdline(command) + ' "%1"')
    elif sys.platform == "linux":
        applications = root.parents[1] / "applications"
        applications.mkdir(parents=True, exist_ok=True)
        (applications / DESKTOP_NAME).write_text(
            "[Desktop Entry]\nType=Application\n"
            f"Name={APP_NAME}\nNoDisplay=true\nTerminal=false\n"
            f"MimeType=x-scheme-handler/{SCHEME};\n"
            f"Exec={' '.join(desktop_argument(arg) for arg in command)} %u\n",
            encoding="utf-8",
        )
        subprocess.run(["xdg-mime", "default", DESKTOP_NAME, f"x-scheme-handler/{SCHEME}"], check=True)
    else:
        raise ValueError("Supported systems: macOS, Windows and Ubuntu/Linux desktop.")


def install(url: str) -> Path:
    """Copy the launcher and bridge so moving the checkout does not break registered links."""
    backend = validate_backend(url)
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("Install uv before installing the Stream Deck host launcher.")
    root = install_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("host_launcher.py", "streamdeck_bridge.py", "dictation.py"):
        source = Path(__file__).resolve().with_name(name)
        destination = root / name
        if source != destination.resolve():
            shutil.copyfile(source, destination)
    (root / "config.json").write_text(json.dumps({"url": backend, "uv": uv}), encoding="utf-8")
    register(root, uv)
    return root


def uninstall() -> None:
    """Remove only this launcher's registration and known files; keep diagnostic logs."""
    root = install_dir()
    if sys.platform == "darwin":
        app = Path.home() / "Applications" / f"{APP_NAME}.app"
        if app.exists():
            subprocess.run(
                [
                    "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister",
                    "-u",
                    str(app),
                ],
                check=True,
            )
            shutil.rmtree(app)
    elif sys.platform == "win32":
        import winreg

        for suffix in (r"\shell\open\command", r"\shell\open", r"\shell", ""):
            with contextlib.suppress(FileNotFoundError):
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{SCHEME}" + suffix)
    elif sys.platform == "linux":
        (root.parents[1] / "applications" / DESKTOP_NAME).unlink(missing_ok=True)
    for name in ("config.json", "host_launcher.py", "streamdeck_bridge.py", "dictation.py"):
        (root / name).unlink(missing_ok=True)


def open_link(link: str) -> None:
    """Accept only the fixed start action, and hold a per-install lock for the helper's lifetime."""
    if link not in {LINK, LINK + "/"}:
        raise ValueError("Only physicalai-streamdeck://start is supported; links cannot supply arguments.")
    root = Path(__file__).resolve().parent
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    backend = validate_backend(config["url"])
    with (root / "helper.lock").open("a+b") as lock:
        if sys.platform == "win32":
            import msvcrt

            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return
        else:
            import fcntl

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
        with (root / "helper.log").open("ab") as log:
            subprocess.run(
                [config["uv"], "run", "--no-project", str(root / "streamdeck_bridge.py"), "--url", backend],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def main() -> None:
    """Install/uninstall explicitly; normal browser invocation only starts the configured helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--install", action="store_true", help="Register the host launcher for this user")
    actions.add_argument("--uninstall", action="store_true", help="Remove this user's launcher registration")
    actions.add_argument("--open", metavar="URL", help=argparse.SUPPRESS)
    parser.add_argument("--url", default="http://localhost:7860", help="Trusted backend URL to save at installation")
    args = parser.parse_args()
    try:
        if args.install:
            root = install(args.url)
            print(f"Installed in {root}. Use Start Stream Deck helper in Studio. USB prerequisites are still required.")
        elif args.uninstall:
            uninstall()
            print("Host launcher removed. Diagnostic logs were retained.")
        else:
            open_link(args.open)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
