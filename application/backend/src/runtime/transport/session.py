from __future__ import annotations

from typing import Any

from runtime.transport.ids import derive_endpoint_port, validate_session_name


def build_session_config(name: str, *, listen: bool) -> Any:
    """Build a peer-only, loopback-only Zenoh configuration."""
    import zenoh

    name = validate_session_name(name)
    port = derive_endpoint_port(name)
    config = zenoh.Config()
    config.insert_json5("mode", '"peer"')
    config.insert_json5("scouting/multicast/enabled", "false")
    config.insert_json5("scouting/gossip/enabled", "false")
    endpoint_kind = "listen" if listen else "connect"
    config.insert_json5(f"{endpoint_kind}/endpoints", f'["tcp/127.0.0.1:{port}"]')
    return config


def open_session(name: str, *, listen: bool) -> Any:
    """Open a secure Studio-owned Zenoh session."""
    import zenoh

    return zenoh.open(build_session_config(name, listen=listen))
