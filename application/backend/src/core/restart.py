# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared graceful-restart request helper.

Used by any endpoint whose change only takes effect after the process is
replaced - e.g. picking up installed catalog plugin changes
(`POST /api/system/restart`), or flipping a setting that is cached for the
life of the process, such as the SSH remote-trainer feature's master switch
(`core.security.get_ssh_feature_availability`).
"""

import os
import signal


def request_graceful_restart() -> None:
    """Send this process SIGTERM so the process supervisor restarts it.

    The FastAPI lifespan (`core.lifecycle.lifespan`) stops workers on
    receiving the signal, then re-executes the process
    (`core.lifecycle._restart_process`) if
    `HealthService.plugin_restart_required` was set beforehand. Callers must
    set that flag first - this function only requests the shutdown half of
    that sequence.
    """
    os.kill(os.getpid(), signal.SIGTERM)
