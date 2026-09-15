#!/usr/bin/env python3
"""Maintain Physical AI Studio agent skills (adapters, layout, frontmatter)."""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import re
import subprocess  # nosec B404 - only used below for a fixed, non-shell mklink fallback
import sys
from pathlib import Path

_SCRIPT = "python3 .github/scripts/skills/agent_skills.py"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def skill_dirs(root: Path) -> list[tuple[str, str]]:
    """Return (bucket, skill_name) for each canonical skill directory."""
    found: list[tuple[str, str]] = []
    for bucket in ("library", "application"):
        bucket_path = root / "skills" / bucket
        if not bucket_path.is_dir():
            continue
        for child in sorted(bucket_path.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file():
                found.append((bucket, child.name))
    return found


def adapter_target(bucket: str, name: str) -> str:
    return f"../../skills/{bucket}/{name}"


def read_link(path: Path) -> str | None:
    if path.is_symlink():
        return os.readlink(path)
    return None


def resolves_to(link: Path, expected_target: str) -> bool:
    """Check if a symlink or junction points at the expected target.

    On non-Windows, adapters are symlinks and ``read_link`` suffices.
    On Windows, ``sync`` may fall back to a directory junction when
    symlinks are unavailable; ``read_link`` returns ``None`` for those,
    so resolve the link and compare against the resolved expected target.
    """
    if link.is_symlink():
        return read_link(link) == expected_target
    if link.is_dir():
        return link.resolve() == (link.parent / expected_target).resolve()
    return False


def remove_adapter(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir() and not path.is_symlink():
        raise RuntimeError(f"{path} is a real directory; remove it manually")


def create_adapter(link: Path, target: str) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    remove_adapter(link)
    abs_target = (link.parent / target).resolve()
    if not abs_target.is_dir():
        raise RuntimeError(f"canonical skill missing: {abs_target}")

    if platform.system() == "Windows":
        _create_windows_link(link, abs_target)
    else:
        link.symlink_to(target, target_is_directory=True)


def _resolve_cmd_exe() -> Path:
    """Resolve cmd.exe via the Win32 API, not env vars, so the path is never tainted."""
    buf = ctypes.create_unicode_buffer(260)
    if not ctypes.windll.kernel32.GetSystemDirectoryW(buf, len(buf)):  # type: ignore[attr-defined]
        raise RuntimeError("failed to resolve the Windows system directory")
    return Path(buf.value) / "cmd.exe"


# cmd.exe re-parses its /c argument itself even with shell=False, so these are unsafe
# regardless of quoting. A denylist (not an ASCII allowlist) is used deliberately here
# so legitimate non-ASCII/`+`/`@`/`$` paths still work; this list is the full set of
# cmd.exe metacharacters (including "(", ")", which earlier revisions of this check
# omitted), not just the common ones.
_CMD_METACHARACTERS = set('&|<>^"%!(),;\n\r\t')


def _reject_unsafe_for_cmd(path: Path) -> None:
    text = str(path)
    if any(ch in _CMD_METACHARACTERS for ch in text):
        raise RuntimeError(f"path contains characters unsafe for cmd.exe: {path}")


def _create_windows_link(link: Path, abs_target: Path) -> None:
    try:
        link.symlink_to(abs_target, target_is_directory=True)
        return
    except OSError:
        pass

    cmd_exe = _resolve_cmd_exe()
    if not cmd_exe.is_file():
        raise RuntimeError(f"cmd.exe not found at expected path: {cmd_exe}")
    # link/abs_target are built from skill directory names, which can be introduced
    # by an unreviewed PR and hit by `sync` locally (e.g. via pre-commit) before any
    # human review completes, so they are not trustworthy inputs; validate explicitly
    # rather than relying on review having already screened them.
    _reject_unsafe_for_cmd(link)
    _reject_unsafe_for_cmd(abs_target)

    subprocess.run(  # nosec B603 - absolute cmd.exe path, no shell, paths validated above
        [str(cmd_exe), "/c", "mklink", "/J", str(link), str(abs_target)],
        check=True,
        capture_output=True,
        text=True,
    )


def cmd_sync(root: Path) -> int:
    adapters = (root / ".claude" / "skills", root / ".agents" / "skills")

    for bucket, name in skill_dirs(root):
        target = adapter_target(bucket, name)
        for adapter_root in adapters:
            link = adapter_root / name
            current = read_link(link)
            if current == target and link.exists():
                continue
            create_adapter(link, target)

    names = sorted({name for _, name in skill_dirs(root)})
    for adapter_root in adapters:
        print(f"{adapter_root.relative_to(root)}:")
        for name in names:
            if (adapter_root / name).exists():
                print(f"  {name}")
    return 0


def cmd_check_adapters(root: Path) -> int:
    errors: list[str] = []
    adapters = (root / ".claude" / "skills", root / ".agents" / "skills")

    for bucket, name in skill_dirs(root):
        target = adapter_target(bucket, name)
        for adapter_root in adapters:
            link = adapter_root / name
            if resolves_to(link, target):
                continue
            current = read_link(link)
            errors.append(
                f"{link}: expected adapter to {target!r}, got {current!r}"
            )

    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        print(f"Run: {_SCRIPT} sync", file=sys.stderr)
        return 1
    return 0


def frontmatter_name(skill_md: Path) -> str | None:
    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None
    for line in match.group(1).splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def cmd_validate(root: Path) -> int:
    errors: list[str] = []

    for bucket in ("library", "application"):
        bucket_dir = root / "skills" / bucket
        if not bucket_dir.is_dir():
            errors.append(f"Missing {bucket_dir}")
            continue

        for child in sorted(bucket_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue

            name = child.name
            fm_name = frontmatter_name(skill_md)
            if not fm_name:
                errors.append(f"{skill_md}: missing frontmatter name")
            elif fm_name != name:
                errors.append(
                    f"{skill_md}: frontmatter name {fm_name!r} "
                    f"must match directory {name!r}"
                )

            for entry in child.iterdir():
                if entry.is_symlink():
                    errors.append(
                        f"{child}: must not contain symlinks "
                        "(canonical skill content only)"
                    )
                    break

            target = adapter_target(bucket, name)
            for adapter_root in (root / ".claude" / "skills", root / ".agents" / "skills"):
                link = adapter_root / name
                if not link.exists():
                    errors.append(f"Missing adapter {link} (run: {_SCRIPT} sync)")
                    continue
                if not resolves_to(link, target):
                    errors.append(
                        f"{link}: expected adapter to {target!r}, "
                        f"got {read_link(link)!r}"
                    )
                    continue
                if not (link / "SKILL.md").is_file():
                    errors.append(f"Broken adapter {link}")

    if errors:
        for msg in errors:
            print(f"::error::{msg}", file=sys.stderr)
        print(f"Skills validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print("Skills validation passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Maintain agent skills: sync .claude/.agents adapters and validate layout.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync", help="create or refresh client adapter symlinks")
    sub.add_parser(
        "validate",
        help="check SKILL.md frontmatter, layout, and adapters (no writes)",
    )
    sub.add_parser(
        "check-adapters",
        help="verify adapter symlinks only (no writes)",
    )
    args = parser.parse_args()
    root = repo_root()

    if args.command == "sync":
        return cmd_sync(root)
    if args.command == "validate":
        return cmd_validate(root)
    if args.command == "check-adapters":
        return cmd_check_adapters(root)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
