import re


def safe_archive_name(name: str, fallback: str = "archive") -> str:
    """Create a filesystem-safe archive filename stem."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return sanitized or fallback
