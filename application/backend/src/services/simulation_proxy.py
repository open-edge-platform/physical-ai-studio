# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pass a simulation's viewer and camera frames on to the browser through Studio's own address.

A simulation serves its HTTP API and its 3D viewer (viser) on ports of its own, on this machine.
The browser may not reach them: in Docker only Studio's port is published. Studio passes on what
the browser needs, and only to the addresses it knows for the robot's simulation (see
``RobotSimulationService.proxy_addresses``), never to an address from the request:

- the viewer, page and WebSocket, under ``<simulation path>/viewer/``. The viser client connects
  its WebSocket to the page's own URL without its trailing slash, so it works under any path;
- the camera frames and streams of the HTTP API (``PROXIED_HTTP_PATH``) under ``<simulation path>/http/``.
"""

import asyncio
import contextlib
import ipaddress
import re
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx
import websockets.asyncio.client
import websockets.exceptions
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from starlette.responses import Response, StreamingResponse
from websockets.typing import Subprotocol

PROXIED_HTTP_PATH = re.compile(r"/cameras/[A-Za-z0-9][A-Za-z0-9_.-]*/(frame\.jpg|mjpeg)")
"""The paths of a simulation's HTTP API the browser gets: a camera's latest frame and its stream."""
_VIEWER_PATH = re.compile(r"[A-Za-z0-9_.~/-]*")
"""Paths of the viewer's page and files: no ``..`` segment, no query, nothing to escape."""
CONNECT_TIMEOUT_S = 5.0
HEADERS_TIMEOUT_S = 5.0
"""How long the simulation may take to start its answer (connecting included)."""
_PASSED_HEADERS = ("content-type", "cache-control", "last-modified", "etag")
_PASSED_CLOSE_CODES = {1000, 1001, 1002, 1003, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, *range(3000, 5000)}
"""Close codes that may be sent on (1005, 1006 and 1015 only say how a connection ended)."""


def local_base_url(url: str | None) -> str | None:
    """Return *url* without a trailing slash when it is an HTTP address on this machine, else ``None``."""
    if not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url)
        host, _ = parts.hostname or "", parts.port
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or parts.query or parts.fragment:
        return None
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                return None
        except ValueError:
            return None
    return url.rstrip("/")


def is_viewer_path(path: str) -> bool:
    """Return whether *path* (relative to the viewer) may be passed on to the viewer."""
    return _VIEWER_PATH.fullmatch(path) is not None and ".." not in path.split("/")


def is_proxied_http_path(path: str) -> bool:
    """Return whether *path* of a simulation's HTTP API may be passed on to the browser."""
    return PROXIED_HTTP_PATH.fullmatch(path) is not None


async def proxy_get(url: str, *, read_timeout_s: float | None) -> Response:
    """GET *url* and stream the response on; the upstream closes when the browser goes away.

    The response's headers must arrive within ``HEADERS_TIMEOUT_S``; *read_timeout_s* bounds each
    read of the body after that (``None`` for a stream). A redirect is not followed, nor passed on:
    it could lead off this machine.
    """
    client = httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT_S, read=read_timeout_s), follow_redirects=False)
    try:
        upstream = await asyncio.wait_for(client.send(client.build_request("GET", url), stream=True), HEADERS_TIMEOUT_S)
    except TimeoutError:
        await client.aclose()
        logger.debug("Simulation proxy GET {}: no response within {} s", url, HEADERS_TIMEOUT_S)
        return Response("The simulation does not answer in time.", status_code=504, media_type="text/plain")
    except httpx.HTTPError as error:
        await client.aclose()
        logger.debug("Simulation proxy GET {} failed: {}", url, error)
        return Response("The simulation does not answer.", status_code=502, media_type="text/plain")
    except BaseException:
        await client.aclose()
        raise
    if upstream.is_redirect:
        await upstream.aclose()
        await client.aclose()
        logger.warning("Simulation proxy GET {} answered with a redirect; not following it", url)
        return Response("The simulation redirected elsewhere.", status_code=502, media_type="text/plain")

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        except httpx.HTTPError as error:
            logger.debug("Simulation proxy GET {} broke off: {}", url, error)
        finally:
            await upstream.aclose()
            await client.aclose()

    headers = {name: upstream.headers[name] for name in _PASSED_HEADERS if name in upstream.headers}
    return StreamingResponse(body(), status_code=upstream.status_code, headers=headers)


async def proxy_websocket(browser: WebSocket, url: str) -> None:
    """Connect *browser* to the WebSocket at *url* and pass messages both ways until one side closes.

    The upstream is opened first, with the subprotocols the browser asked for, so that the browser
    is accepted with the one the upstream chose (viser checks its version that way).
    """
    requested = [Subprotocol(protocol) for protocol in browser.scope.get("subprotocols", [])]
    try:
        upstream = await _ConnectWithoutRedirects(
            url, subprotocols=requested or None, max_size=None, open_timeout=CONNECT_TIMEOUT_S, compression=None
        )
    except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as error:
        logger.debug("Simulation proxy WebSocket {} failed: {}", url, error)
        await browser.close(code=1011)
        return

    async with upstream:
        await browser.accept(subprotocol=upstream.subprotocol)
        to_upstream = asyncio.create_task(_browser_to_upstream(browser, upstream))
        to_browser = asyncio.create_task(_upstream_to_browser(upstream, browser))
        try:
            await asyncio.wait({to_upstream, to_browser}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (to_upstream, to_browser):
                task.cancel()
            await asyncio.gather(to_upstream, to_browser, return_exceptions=True)


class _ConnectWithoutRedirects(websockets.asyncio.client.connect):
    """``websockets``' ``connect``, which takes a redirect of the handshake for a failure.

    It would follow one, to another host or port too: the simulation's address is checked to be on
    this machine, a redirect's target is not.
    """

    def process_redirect(self, exc: Exception) -> Exception | str:
        return exc


async def _browser_to_upstream(browser: WebSocket, upstream: websockets.asyncio.client.ClientConnection) -> None:
    while True:
        message = await browser.receive()
        if message["type"] == "websocket.disconnect":
            code = message.get("code", 1000)
            await upstream.close(code=code if code in _PASSED_CLOSE_CODES else 1000)
            return
        data = message.get("bytes")
        if data is None:
            data = message.get("text")
        if data is not None:
            try:
                await upstream.send(data)
            except websockets.exceptions.ConnectionClosed:
                return


async def _upstream_to_browser(upstream: websockets.asyncio.client.ClientConnection, browser: WebSocket) -> None:
    with contextlib.suppress(websockets.exceptions.ConnectionClosed):
        async for message in upstream:
            if isinstance(message, bytes):
                await browser.send_bytes(message)
            else:
                await browser.send_text(message)
    code = upstream.close_code
    if code is None or code not in _PASSED_CLOSE_CODES:
        code = 1000
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await browser.close(code=code, reason=upstream.close_reason or "")
