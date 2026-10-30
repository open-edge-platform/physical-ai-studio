# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared human/JSON output and credential-safe CLI errors."""

import json
from collections.abc import Callable
from typing import Any

import click


def invoke(callback: Callable[..., dict[str, Any]], *, as_json: bool, **kwargs: Any) -> None:
    """Run a command's workflow with consistent output and sanitized errors."""
    try:
        result = callback(**kwargs)
    except (Exception, KeyboardInterrupt) as exc:
        # SDK exceptions can contain request details. Never echo credentials.
        from cli.hf.inspection import redact

        message = redact(str(exc) or "Transfer interrupted")
        if as_json:
            error = {"code": "hf_operation_failed", "message": message}
            if hasattr(exc, "resume_command"):
                error.update(resume_command=exc.resume_command, transfer_id=getattr(exc, "transfer_id", ""))
            click.echo(json.dumps({"error": error}))
            raise click.exceptions.Exit(1) from None
        raise click.ClickException(message) from None
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return
    for key, item in result.items():
        click.echo(f"{key}: {json.dumps(item) if isinstance(item, (dict, list)) else item}")
