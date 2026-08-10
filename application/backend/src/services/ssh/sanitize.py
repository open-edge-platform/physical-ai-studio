# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sanitize output streamed from remote commands.

Remote stdout/stderr is environment-influenced content, not trusted text. A
hostile or merely noisy remote process can emit ANSI/OSC escape sequences that
rewrite what the operator sees, bidi overrides that reorder rendered text, or
megabytes of output that blow the ``extra_info`` cap. Everything crossing the
SSH boundary into an API response or a job message passes through
:func:`sanitize_output` first.

The stripped set is exactly the one the design doc names:

* Unicode ``Cc`` characters, except ``\\n``, which survives as a line separator;
* the ``Cf`` bidi-override code points (``U+200F``, ``U+202A``-``U+202E``,
  ``U+2066``-``U+2069``, ``U+FEFF``) - not all of ``Cf``, which also holds
  harmless formatting characters;
* every ESC-introduced sequence, from ``\\x1b`` through its terminator.
"""

import unicodedata
from typing import Final

# Cf code points that can reorder or hide rendered text. Deliberately not all of
# Cf: a soft hyphen or a joiner is harmless and stripping it would corrupt
# legitimate output.
# Written as escapes, not literals: a source file containing real bidi overrides
# is itself unreviewable, which is the whole reason these are being stripped.
_BIDI_OVERRIDES: Final[frozenset[str]] = frozenset(
    {
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u202a",  # LEFT-TO-RIGHT EMBEDDING
        "\u202b",  # RIGHT-TO-LEFT EMBEDDING
        "\u202c",  # POP DIRECTIONAL FORMATTING
        "\u202d",  # LEFT-TO-RIGHT OVERRIDE
        "\u202e",  # RIGHT-TO-LEFT OVERRIDE
        "\u2066",  # LEFT-TO-RIGHT ISOLATE
        "\u2067",  # RIGHT-TO-LEFT ISOLATE
        "\u2068",  # FIRST STRONG ISOLATE
        "\u2069",  # POP DIRECTIONAL ISOLATE
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
    }
)

_ESC: Final = "\x1b"
_BEL: Final = "\x07"
_LINE_SEPARATOR: Final = "\n"

# Appended when the total cap forces a cut, so an operator can tell a truncated
# message from a command that simply produced short output.
TRUNCATION_MARKER: Final = "...[truncated]"


def _skip_escape_sequence(text: str, start: int) -> int:  # noqa: PLR0911 - one return per terminator rule.
    """Return the index just past the ESC sequence beginning at ``start``.

    ``start`` indexes the ESC itself. Returns ``len(text)`` for an unterminated
    sequence so a dangling ESC can never survive into rendered output.

    Args:
        text: The string being scanned.
        start: Index of the ESC character.

    Returns:
        The index of the first character after the sequence.
    """
    length = len(text)
    index = start + 1
    if index >= length:
        return length

    introducer = text[index]

    # CSI: ESC [ params... final-byte, where the final byte is in 0x40-0x7E.
    if introducer == "[":
        index += 1
        while index < length and not ("\x40" <= text[index] <= "\x7e"):
            index += 1
        # Past the final byte, or end-of-string for an unterminated sequence.
        return min(index + 1, length)

    # OSC (and the other string-terminated introducers: DCS, SOS, PM, APC).
    # Terminated by BEL or by the two-byte ST (ESC \).
    if introducer in {"]", "P", "X", "^", "_"}:
        index += 1
        while index < length:
            char = text[index]
            if char == _BEL:
                return index + 1
            if char == _ESC:
                # ST is ESC \. A bare ESC starts a new sequence, so stop here
                # and let the outer scanner handle it rather than swallowing it.
                if index + 1 < length and text[index + 1] == "\\":
                    return index + 2
                return index
            index += 1
        return length

    # nF escape sequences: ESC, zero or more intermediate bytes (0x20-0x2F), then
    # one final byte (0x30-0x7E). Character-set designations (e.g. `ESC ( B`) are
    # the common case; a plain two-byte sequence (e.g. `ESC c`) is just the
    # zero-intermediate-bytes case of this same rule.
    if "\x20" <= introducer <= "\x2f":
        index += 1
        while index < length and "\x20" <= text[index] <= "\x2f":
            index += 1
        # Past the final byte, or end-of-string for an unterminated sequence.
        return min(index + 1, length)

    # Any other ESC sequence is two bytes: ESC plus the following character.
    return index + 1


def _strip_control_sequences(text: str) -> str:
    """Remove ESC sequences, ``Cc`` characters, and bidi overrides.

    ``\\n`` survives; every other control character does not.

    Args:
        text: Raw text from a remote command.

    Returns:
        The text with control sequences removed.
    """
    kept: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char == _ESC:
            index = _skip_escape_sequence(text, index)
            continue

        if char == _LINE_SEPARATOR:
            kept.append(char)
        elif char in _BIDI_OVERRIDES:
            pass
        elif unicodedata.category(char) != "Cc":
            kept.append(char)

        index += 1

    return "".join(kept)


def sanitize_output(text: str, *, max_line_chars: int, max_total_chars: int) -> str:
    """Strip control sequences from remote output and cap its length.

    Applies the per-line cap first, then the total cap, so one pathological line
    cannot consume the whole budget and hide every following line.

    Args:
        text: Raw stdout/stderr from a remote command.
        max_line_chars: Maximum characters kept per line. Non-positive drops all
            line content.
        max_total_chars: Maximum characters of the joined result. Non-positive
            returns an empty string.

    Returns:
        Sanitized, length-capped text safe to place in an API response or a job
        message.
    """
    if max_total_chars <= 0 or not text:
        return ""

    stripped = _strip_control_sequences(text)
    capped_lines = [line[:max_line_chars] if max_line_chars > 0 else "" for line in stripped.split(_LINE_SEPARATOR)]
    joined = _LINE_SEPARATOR.join(capped_lines)

    if len(joined) <= max_total_chars:
        return joined

    # Keep the marker inside the cap so the result never exceeds the budget the
    # caller sized its storage against.
    if max_total_chars <= len(TRUNCATION_MARKER):
        return joined[:max_total_chars]
    return joined[: max_total_chars - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
