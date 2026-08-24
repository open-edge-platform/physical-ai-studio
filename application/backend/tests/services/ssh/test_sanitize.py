# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for remote-output sanitization.

Remote stdout/stderr is environment-influenced content that ends up in an API
response and in job messages, so these tests pin the whole contract: escape
sequences and control characters are removed, ``\\n`` survives, bidi overrides
are removed, and both length caps hold. The hostile cases at the end combine all
of them in one payload, which is how they would actually arrive.
"""

from services.ssh.sanitize import TRUNCATION_MARKER, sanitize_output

_LINE_CAP = 512
_TOTAL_CAP = 4096


def _sanitize(text: str, line_cap: int = _LINE_CAP, total_cap: int = _TOTAL_CAP) -> str:
    return sanitize_output(text, max_line_chars=line_cap, max_total_chars=total_cap)


def test_plain_text_is_unchanged() -> None:
    assert _sanitize("Docker version 27.3.1") == "Docker version 27.3.1"


def test_empty_text_returns_empty() -> None:
    assert _sanitize("") == ""


def test_newlines_are_preserved() -> None:
    assert _sanitize("first\nsecond\nthird") == "first\nsecond\nthird"


def test_trailing_newline_is_preserved() -> None:
    assert _sanitize("only line\n") == "only line\n"


# --------------------------------------------------------------------------- #
# Escape sequences                                                            #
# --------------------------------------------------------------------------- #


def test_ansi_sgr_colour_codes_are_removed() -> None:
    assert _sanitize("\x1b[31mERROR\x1b[0m: disk full") == "ERROR: disk full"


def test_ansi_cursor_movement_is_removed() -> None:
    # A remote process can use cursor movement to overwrite a line it already
    # printed, so the visible text no longer matches the recorded text.
    assert _sanitize("real message\x1b[2K\x1b[1Afake message") == "real messagefake message"


def test_osc_8_hyperlink_is_removed_but_label_survives() -> None:
    # OSC 8 makes arbitrary text a clickable link to an arbitrary URL - the exact
    # trick for rendering "docs.example.com" as a link to somewhere else.
    text = "\x1b]8;;https://evil.example.com\x07Intel docs\x1b]8;;\x07"

    assert _sanitize(text) == "Intel docs"


def test_osc_terminated_by_string_terminator_is_removed() -> None:
    assert _sanitize("before\x1b]0;new window title\x1b\\after") == "beforeafter"


def test_unterminated_csi_sequence_is_dropped_entirely() -> None:
    # A dangling ESC must never survive, so an unterminated sequence consumes the
    # remainder rather than leaking the introducer.
    assert _sanitize("visible\x1b[38;5;") == "visible"


def test_unterminated_osc_sequence_is_dropped_entirely() -> None:
    assert _sanitize("visible\x1b]8;;https://evil.example.com") == "visible"


def test_two_byte_escape_sequence_is_removed() -> None:
    # ESC c is a full terminal reset.
    assert _sanitize("before\x1bcafter") == "beforeafter"


def test_charset_designation_escape_sequence_is_fully_removed() -> None:
    # ESC ( B designates ASCII as G0: an intermediate byte (`(`) followed by a
    # final byte (`B`), not a plain two-byte sequence. The final byte must not
    # leak into the output as literal text.
    assert _sanitize("before\x1b(Bafter") == "beforeafter"


def test_multiple_intermediate_bytes_escape_sequence_is_fully_removed() -> None:
    assert _sanitize("before\x1b%%Gafter") == "beforeafter"


def test_unterminated_intermediate_byte_escape_sequence_is_dropped_entirely() -> None:
    assert _sanitize("visible\x1b(") == "visible"


def test_bare_escape_at_end_is_removed() -> None:
    assert _sanitize("tail\x1b") == "tail"


def test_escape_inside_osc_is_not_swallowed() -> None:
    # A bare ESC inside an OSC body ends the OSC and starts a new sequence; the
    # scanner must hand it back rather than consuming the CSI that follows.
    assert _sanitize("a\x1b]0;title\x1b[31mred\x1b[0mb") == "aredb"


# --------------------------------------------------------------------------- #
# Control characters and bidi overrides                                       #
# --------------------------------------------------------------------------- #


def test_nul_and_other_control_bytes_are_removed() -> None:
    assert _sanitize("a\x00b\x07c\x08d\x1fe") == "abcde"


def test_carriage_return_is_removed() -> None:
    # CR alone returns the cursor to column zero, letting later text overwrite
    # earlier text on the same rendered line.
    assert _sanitize("harmless text\rmalicious text") == "harmless textmalicious text"


def test_tab_is_removed_as_a_control_character() -> None:
    assert _sanitize("name\tvalue") == "namevalue"


def test_bidi_override_characters_are_removed() -> None:
    assert _sanitize("start\u202eresrever\u202cend") == "startresreverend"


def test_all_bidi_override_code_points_are_removed() -> None:
    overrides = "\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff"

    assert _sanitize(f"a{overrides}b") == "ab"


def test_harmless_format_characters_are_preserved() -> None:
    # Only the reordering subset of Cf is stripped: a soft hyphen and a zero-width
    # joiner are legitimate content, and removing them would corrupt output.
    assert _sanitize("soft\u00adhyphen\u200djoiner") == "soft\u00adhyphen\u200djoiner"


def test_non_ascii_text_is_preserved() -> None:
    assert _sanitize("GPU wärmt: 日本語 ✓") == "GPU wärmt: 日本語 ✓"


# --------------------------------------------------------------------------- #
# Length caps                                                                 #
# --------------------------------------------------------------------------- #


def test_over_long_line_is_truncated_to_the_line_cap() -> None:
    result = _sanitize("x" * 900, line_cap=512)

    assert result == "x" * 512


def test_line_cap_applies_per_line_not_to_the_whole_text() -> None:
    result = _sanitize("\n".join(["y" * 20] * 3), line_cap=5)

    assert result == "yyyyy\nyyyyy\nyyyyy"


def test_line_cap_is_applied_before_the_total_cap() -> None:
    # One pathological first line must not consume the whole budget and hide every
    # following line.
    text = "\n".join(["z" * 5_000, "second", "third"])

    result = _sanitize(text, line_cap=10, total_cap=100)

    assert result == "zzzzzzzzzz\nsecond\nthird"


def test_non_positive_line_cap_drops_all_line_content() -> None:
    assert _sanitize("content\nmore", line_cap=0) == "\n"


def test_total_cap_truncates_and_marks_the_result() -> None:
    result = _sanitize("\n".join(f"line {index}" for index in range(500)), total_cap=200)

    assert len(result) == 200
    assert result.endswith(TRUNCATION_MARKER)


def test_result_never_exceeds_the_total_cap() -> None:
    result = _sanitize("q" * 10_000, line_cap=10_000, total_cap=64)

    assert len(result) == 64


def test_total_cap_shorter_than_the_marker_still_holds() -> None:
    result = _sanitize("w" * 100, total_cap=4)

    assert result == "wwww"


def test_non_positive_total_cap_returns_empty() -> None:
    assert _sanitize("anything", total_cap=0) == ""
    assert _sanitize("anything", total_cap=-1) == ""


def test_text_at_exactly_the_total_cap_is_not_marked() -> None:
    result = _sanitize("e" * 64, total_cap=64)

    assert result == "e" * 64


def test_stripping_happens_before_measuring_the_caps() -> None:
    # 400 escape sequences must not count against the caller's budget, otherwise a
    # remote process could hide real output behind invisible padding.
    text = "\x1b[31m" * 400 + "the real message"

    assert _sanitize(text, total_cap=64) == "the real message"


# --------------------------------------------------------------------------- #
# Combined hostile payload                                                    #
# --------------------------------------------------------------------------- #


def test_hostile_nested_payload_is_fully_neutralized() -> None:
    text = (
        "\x1b[2J\x1b[H"  # clear screen, home cursor
        "\x1b]8;;https://evil.example.com\x07"  # hyperlink to elsewhere
        "\u202eLOOKS SAFE\u202c"  # reversed rendering
        "\x1b]8;;\x07"
        "\x00\x07\r"  # raw control bytes
        "\n"
        "\x1b[31mfailure: pull denied\x1b[0m"
        "\x1b[999"  # unterminated tail
    )

    result = _sanitize(text)

    assert result == "LOOKS SAFE\nfailure: pull denied"
    assert "\x1b" not in result
    assert "\x00" not in result
    assert "evil.example.com" not in result


def test_sanitized_output_contains_no_escape_or_control_characters() -> None:
    text = "".join(chr(code) for code in range(0x20)) + "\x1b[1;2;3mtext\x1b]0;title\x07"

    result = _sanitize(text)

    assert result == "\ntext"
