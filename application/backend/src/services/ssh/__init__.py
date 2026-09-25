# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SSH connection and output-sanitization utilities for remote trainers."""

from services.ssh.connection import AliasTarget, DirectTarget
from services.ssh.sanitize import sanitize_output
from services.ssh.transport import CommandFailure, CommandResult, SshTransport, open_transport, reset_alias_gates

__all__ = [
    "AliasTarget",
    "CommandFailure",
    "CommandResult",
    "DirectTarget",
    "SshTransport",
    "open_transport",
    "reset_alias_gates",
    "sanitize_output",
]
