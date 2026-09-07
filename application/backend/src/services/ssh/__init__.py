# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SSH remote-execution boundary for SSH-provisioned remote training servers.

Everything that runs a command on a user's GPU box goes through this package:

* :mod:`services.ssh.transport` owns the connection, argument quoting, the
  bounded timeouts, and the mapping from ``asyncssh`` failures to actionable
  Studio exceptions that carry no host or key detail.
* :mod:`services.ssh.preflight` owns the two-tier verification built on top of it.
* :mod:`services.ssh.sanitize` strips and caps remote output before any of it
  reaches an API response.

Studio never receives, reads, stores, or transports SSH key material. A remote
server is identified only by a non-secret ``ssh_host_alias``, and ``asyncssh``
resolves that alias against the user's own ``~/.ssh/config`` and verifies the host
against their own ``~/.ssh/known_hosts``.
"""

from services.ssh.preflight import (
    DEFAULT_PROTOCOL_VERSION,
    reset_transport_factory,
    run_tier1_preflight,
    run_tier2_preflight,
    set_transport_factory,
)
from services.ssh.sanitize import sanitize_output
from services.ssh.transport import CommandFailure, CommandResult, SshTransport, open_transport, reset_alias_gates

__all__ = [
    "DEFAULT_PROTOCOL_VERSION",
    "CommandFailure",
    "CommandResult",
    "SshTransport",
    "open_transport",
    "reset_alias_gates",
    "reset_transport_factory",
    "run_tier1_preflight",
    "run_tier2_preflight",
    "sanitize_output",
    "set_transport_factory",
]
