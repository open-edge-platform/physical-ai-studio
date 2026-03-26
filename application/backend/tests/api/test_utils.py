from api.utils import safe_archive_name


def test_safe_archive_name_preserves_valid_characters() -> None:
    assert safe_archive_name("model-v1.2") == "model-v1.2"


def test_safe_archive_name_replaces_unsafe_characters() -> None:
    assert safe_archive_name("My Robot ACT Model @ v2") == "My-Robot-ACT-Model-v2"


def test_safe_archive_name_strips_leading_and_trailing_dots_and_hyphens() -> None:
    assert safe_archive_name(".-model-export-.") == "model-export"


def test_safe_archive_name_returns_default_fallback_for_empty_output() -> None:
    assert safe_archive_name("@@@") == "archive"


def test_safe_archive_name_returns_custom_fallback_for_empty_output() -> None:
    assert safe_archive_name("@@@", fallback="dataset") == "dataset"
