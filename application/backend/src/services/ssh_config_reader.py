# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Read-only parser for ``~/.ssh/config``-style files.

Never dials SSH. Detects but never surfaces credential-adjacent directives -
``IdentityFile``, ``IdentityAgent``, ``CertificateFile``, and any ``Password*``
keyword - so a resolved host can be shown for display without ever risking a
key path or secret reaching an API response. ``tests/services/test_ssh_config_reader.py``
asserts this with a fixture where every one of those directives is set.
"""

import re
import shlex
from glob import glob
from pathlib import Path

from schemas.remote_server import ResolvedSshHost, SshHostAliasOption

# "Keyword value", "Keyword=value", and "Keyword = value" are all valid.
_DIRECTIVE_PATTERN = re.compile(r"^(\S+?)(?:\s+|\s*=\s*)(.*)$")

# Keywords that name credential material. Matched by prefix (already
# lowercased) so both "Password" and "PasswordAuthentication" are caught, not
# just an exact keyword. Detected only to be skipped, never stored or read.
_CREDENTIAL_DIRECTIVE_PREFIXES = ("identityfile", "identityagent", "certificatefile", "password")


class _HostBlock:
    """One ``Host`` stanza: its patterns plus the directives this reader cares about."""

    __slots__ = ("hostname", "patterns", "port", "user")

    def __init__(self, patterns: list[str]) -> None:
        self.patterns = patterns
        self.hostname: str | None = None
        self.port: int | None = None
        self.user: str | None = None


def _is_literal_pattern(pattern: str) -> bool:
    """Return True for a plain host name, False for a wildcard or negated pattern."""
    return not pattern.startswith("!") and "*" not in pattern and "?" not in pattern


def _parse_line(line: str) -> tuple[str, list[str]] | None:
    """Split one config line into a lowercased keyword and its arguments.

    Returns None for blank lines, comments, and lines shlex cannot tokenize
    (e.g. unbalanced quotes) - malformed input is skipped, never raised.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    match = _DIRECTIVE_PATTERN.match(stripped)
    if match is None:
        return None
    keyword = match.group(1).lower()
    try:
        args = shlex.split(match.group(2), comments=True)
    except ValueError:
        return None
    return keyword, args


def _resolve_include_paths(pattern: str, including_dir: Path) -> list[Path]:
    """Expand one ``Include`` argument to the files it names.

    A relative pattern is resolved against the directory of the file doing the
    including, so a config and the files it includes can be moved together.
    """
    candidate = Path(pattern).expanduser()
    if not candidate.is_absolute():
        candidate = including_dir / candidate
    return sorted(Path(match) for match in glob(str(candidate)))


def _apply_directive(current: _HostBlock, keyword: str, args: list[str]) -> None:
    """Apply one directive to the ``Host`` block it belongs to.

    Silently ignores anything this reader does not need, including every
    credential-adjacent keyword - those are recognized only so they can be
    skipped, never so their value can be read.
    """
    if keyword.startswith(_CREDENTIAL_DIRECTIVE_PREFIXES):
        return
    if keyword == "hostname":
        current.hostname = args[0]
    elif keyword == "port":
        try:
            current.port = int(args[0])
        except ValueError:
            pass
    elif keyword == "user":
        current.user = args[0]


def _iter_blocks(path: Path, _visited: set[Path] | None = None) -> list[_HostBlock]:
    """Parse one config file into an ordered list of ``Host`` blocks.

    Follows ``Include`` one or more levels deep. A missing or empty file
    yields no blocks rather than raising, so an absent SSH config behaves like
    an empty one. ``_visited`` guards against an ``Include`` cycle.
    """
    visited = _visited if _visited is not None else set()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved in visited:
        return []
    visited.add(resolved)

    try:
        text = path.read_text()
    except OSError:
        return []

    blocks: list[_HostBlock] = []
    current: _HostBlock | None = None

    for line in text.splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        keyword, args = parsed
        if not args:
            continue

        if keyword == "host":
            current = _HostBlock(patterns=args)
            blocks.append(current)
        elif keyword == "include":
            for pattern in args:
                for include_path in _resolve_include_paths(pattern, path.parent):
                    blocks.extend(_iter_blocks(include_path, visited))
        elif current is not None:
            # A directive outside any Host stanza is only informative to real
            # ssh, never to this reader, so it is skipped when `current` is None.
            _apply_directive(current, keyword, args)

    return blocks


def list_host_aliases(config_path: Path) -> list[SshHostAliasOption]:
    """Return every literal ``Host`` alias in the config as a selectable option.

    Wildcard and negated patterns are excluded: an alias is created by a
    literal ``Host`` name, not a pattern that happens to match one. When
    ``HostName`` is unset, the alias itself is the effective hostname - real
    ssh behavior, not an invented default.

    If an alias is defined by more than one stanza (common with ``Include``),
    the stanzas are merged field-by-field with the same last-stanza-wins rule
    as ``resolve_alias``, so each alias appears exactly once and both
    functions agree on its resolved fields.
    """
    merged: dict[str, _HostBlock] = {}
    order: list[str] = []
    for block in _iter_blocks(config_path):
        for pattern in block.patterns:
            if not _is_literal_pattern(pattern):
                continue
            existing = merged.get(pattern)
            if existing is None:
                order.append(pattern)
                existing = _HostBlock(patterns=[pattern])
                merged[pattern] = existing
            if block.hostname is not None:
                existing.hostname = block.hostname
            if block.port is not None:
                existing.port = block.port
            if block.user is not None:
                existing.user = block.user

    return [
        SshHostAliasOption(
            alias=alias,
            hostname=merged[alias].hostname or alias,
            port=merged[alias].port,
            user=merged[alias].user,
        )
        for alias in order
    ]


def resolve_alias(config_path: Path, alias: str) -> ResolvedSshHost:
    """Resolve one alias to its effective hostname/port/user, for display only.

    Matches only a literal ``Host`` pattern equal to ``alias``: a wildcard
    stanza that would match it via glob is not a usable target, since aliases
    are created by literal name, and resolution here does no glob matching.

    If the alias is defined by more than one stanza (common with ``Include``),
    every matching stanza is scanned in file order and a later one overrides an
    earlier one field-by-field - only a field the later stanza actually sets is
    overridden, not the whole result. This is last-stanza-wins, the opposite of
    real ssh's first-obtained-value-wins rule: it is deliberate here so an
    ``Include``d override file takes effect, since display-only resolution has
    no reason to replicate ssh's actual precedence.
    """
    found = False
    hostname: str | None = None
    port: int | None = None
    user: str | None = None
    for block in _iter_blocks(config_path):
        for pattern in block.patterns:
            if _is_literal_pattern(pattern) and pattern == alias:
                found = True
                if block.hostname is not None:
                    hostname = block.hostname
                if block.port is not None:
                    port = block.port
                if block.user is not None:
                    user = block.user
                break
    if not found:
        return ResolvedSshHost(alias=alias, found=False)
    return ResolvedSshHost(
        alias=alias,
        hostname=hostname or alias,
        port=port,
        user=user,
        found=True,
    )
