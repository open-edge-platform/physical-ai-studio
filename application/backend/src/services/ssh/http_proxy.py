# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""HTTP CONNECT proxy support for direct SSH connections."""

import asyncio
import socket
from typing import Final
from urllib.parse import urlsplit
from urllib.request import getproxies_environment

_MAX_PROXY_RESPONSE_SIZE: Final = 8192


def resolve_http_connect_proxy(host: str, port: int) -> tuple[str, int] | None:
    """Return the configured HTTP CONNECT proxy unless the target bypasses it."""
    proxies = getproxies_environment()
    if _matches_no_proxy(host, port, proxies.get("no")):
        return None

    proxy_url = proxies.get("https") or proxies.get("http")
    if proxy_url is None:
        return None
    parsed = urlsplit(proxy_url)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.username is not None or parsed.password is not None:
        raise ValueError("SSH proxy must be an unauthenticated http:// URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("SSH proxy URL must not contain a path, query, or fragment")
    return parsed.hostname, parsed.port or 80


async def open_http_connect_socket(
    proxy: tuple[str, int],
    host: str,
    port: int,
    timeout_s: float,
) -> socket.socket:
    """Open a socket to an SSH target through an HTTP CONNECT proxy."""
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in host):
        raise ValueError("SSH host contains invalid control characters")
    connect_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    target = f"{connect_host}:{port}"
    request = f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode("ascii")
    loop = asyncio.get_running_loop()
    proxy_socket: socket.socket | None = None
    try:
        async with asyncio.timeout(timeout_s):
            addresses = await loop.getaddrinfo(*proxy, type=socket.SOCK_STREAM)
            last_error: OSError | None = None
            for family, socket_type, protocol, _, address in addresses:
                candidate = socket.socket(family, socket_type, protocol)
                candidate.setblocking(False)
                try:
                    await loop.sock_connect(candidate, address)
                except OSError as error:
                    candidate.close()
                    last_error = error
                    continue
                except BaseException:
                    candidate.close()
                    raise
                proxy_socket = candidate
                break
            if proxy_socket is None:
                raise last_error or OSError("HTTP CONNECT proxy has no reachable address")

            await loop.sock_sendall(proxy_socket, request)
            response = bytearray()
            while not response.endswith(b"\r\n\r\n"):
                chunk = await loop.sock_recv(proxy_socket, 1)
                if not chunk or len(response) >= _MAX_PROXY_RESPONSE_SIZE:
                    raise OSError("Invalid HTTP CONNECT response")
                response.extend(chunk)

        status_line = bytes(response).split(b"\r\n", 1)[0].split()
        if len(status_line) < 2 or not status_line[1].isdigit() or not 200 <= int(status_line[1]) < 300:
            raise OSError("HTTP CONNECT proxy rejected the connection")
        return proxy_socket
    except BaseException:
        if proxy_socket is not None:
            proxy_socket.close()
        raise


def _matches_no_proxy(host: str, port: int, no_proxy: str | None) -> bool:
    if no_proxy is None:
        return False
    normalized_host = host.rstrip(".").lower()
    host_with_port = f"{normalized_host}:{port}"
    for entry in no_proxy.split(","):
        candidate = entry.strip().lstrip(".").rstrip(".").lower()
        if candidate == "*" or candidate in {normalized_host, host_with_port}:
            return True
        if ":" not in candidate and normalized_host.endswith(f".{candidate}"):
            return True
    return False
