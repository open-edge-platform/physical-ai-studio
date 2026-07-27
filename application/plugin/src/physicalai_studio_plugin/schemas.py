"""Pydantic schemas shared by plugin protocols."""

from __future__ import annotations

from pydantic import BaseModel


class SerialPortInfo(BaseModel):
    """Connection metadata for a discovered serial or network robot."""

    connection_string: str | None
    serial_number: str | None
