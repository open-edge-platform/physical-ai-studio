# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``physicalai-trainer-batch``: run one batch training job and exit.

The cloud provider is selected with ``PHYSICALAI_BATCH_PROVIDER``; each
provider contributes a :class:`~trainer.batch.storage.JobStorage` factory.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable

from trainer.batch import BatchSettings, install_sigterm_handler, run
from trainer.batch.storage import JobStorage


def _s3(settings: BatchSettings) -> JobStorage:
    from trainer.batch.s3 import S3JobStorage

    return S3JobStorage(settings)


_PROVIDERS: dict[str, Callable[[BatchSettings], JobStorage]] = {
    "aws": _s3,
}


def main() -> int:
    provider = os.environ.get("PHYSICALAI_BATCH_PROVIDER", "aws").lower()
    try:
        make_storage = _PROVIDERS[provider]
    except KeyError:
        sys.stderr.write(f"Unknown PHYSICALAI_BATCH_PROVIDER={provider!r}; known: {sorted(_PROVIDERS)}\n")
        return 2
    settings = BatchSettings()  # type: ignore[call-arg]
    stop = threading.Event()
    install_sigterm_handler(stop)
    return run(make_storage(settings), settings, stop=stop)


if __name__ == "__main__":
    sys.exit(main())
