"""normalize robot types and remove widowx serial_number

Revision ID: d4e5f6a7b8c9
Revises: c9d8e7f6a5b4
Create Date: 2026-07-27 00:00:00.000000
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "c9d8e7f6a5b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TYPE_NORMALIZATION = {
    "SO101_FOLLOWER": "SO101_Follower",
    "SO101_LEADER": "SO101_Leader",
    "TROSSEN_WIDOWXAI_FOLLOWER": "Trossen_WidowXAI_Follower",
    "TROSSEN_WIDOWXAI_LEADER": "Trossen_WidowXAI_Leader",
    "TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER": "Trossen_Bimanual_WidowXAI_Follower",
    "TROSSEN_BIMANUAL_WIDOWXAI_LEADER": "Trossen_Bimanual_WidowXAI_Leader",
}

_TYPE_DENORMALIZATION = {new: old for old, new in _TYPE_NORMALIZATION.items()}

_WIDOWX_TYPES = (
    "Trossen_WidowXAI_Follower",
    "Trossen_WidowXAI_Leader",
    "Trossen_Bimanual_WidowXAI_Follower",
    "Trossen_Bimanual_WidowXAI_Leader",
)


def _parse_payload(payload_raw: object) -> dict | None:
    if isinstance(payload_raw, dict):
        return payload_raw
    if isinstance(payload_raw, str):
        try:
            parsed = json.loads(payload_raw)
        except ValueError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_robot_types(conn: sa.Connection) -> None:
    for legacy_type, canonical_type in _TYPE_NORMALIZATION.items():
        conn.execute(
            sa.text("UPDATE project_robots SET type = :canonical WHERE type = :legacy"),
            {"canonical": canonical_type, "legacy": legacy_type},
        )


def _denormalize_robot_types(conn: sa.Connection) -> None:
    for canonical_type, legacy_type in _TYPE_DENORMALIZATION.items():
        conn.execute(
            sa.text("UPDATE project_robots SET type = :legacy WHERE type = :canonical"),
            {"legacy": legacy_type, "canonical": canonical_type},
        )


def _remove_widowx_serial_number(conn: sa.Connection) -> None:
    rows = conn.execute(
        sa.text("SELECT id, payload FROM project_robots WHERE type IN :types").bindparams(
            sa.bindparam("types", expanding=True)
        ),
        {"types": _WIDOWX_TYPES},
    ).fetchall()

    for robot_id, payload_raw in rows:
        payload = _parse_payload(payload_raw)
        if payload is None or "serial_number" not in payload:
            continue

        payload.pop("serial_number", None)
        conn.execute(
            sa.text("UPDATE project_robots SET payload = :payload WHERE id = :id"),
            {"payload": json.dumps(payload), "id": robot_id},
        )


def _add_widowx_serial_number(conn: sa.Connection) -> None:
    rows = conn.execute(
        sa.text("SELECT id, payload FROM project_robots WHERE type IN :types").bindparams(
            sa.bindparam("types", expanding=True)
        ),
        {"types": _WIDOWX_TYPES},
    ).fetchall()

    for robot_id, payload_raw in rows:
        payload = _parse_payload(payload_raw)
        if payload is None or "serial_number" in payload:
            continue

        payload["serial_number"] = ""
        conn.execute(
            sa.text("UPDATE project_robots SET payload = :payload WHERE id = :id"),
            {"payload": json.dumps(payload), "id": robot_id},
        )


def upgrade() -> None:
    """Normalize legacy robot types and remove widowx serial_number from payload."""
    conn = op.get_bind()
    _normalize_robot_types(conn)
    _remove_widowx_serial_number(conn)


def downgrade() -> None:
    """Re-add widowx serial_number and restore legacy uppercase type values."""
    conn = op.get_bind()
    _add_widowx_serial_number(conn)
    _denormalize_robot_types(conn)
