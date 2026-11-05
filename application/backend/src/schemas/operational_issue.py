"""Provider-independent problems and recovery instructions exposed to clients."""

from pydantic import BaseModel


class OperationalIssue(BaseModel):
    """A current problem identified consistently across reporting resources."""

    id: str
    title: str
    detail: str
    copy_text: str | None = None
