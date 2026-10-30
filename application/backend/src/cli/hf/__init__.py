# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI-local Hugging Face inspection and resumable transfers."""

import click

from cli.hf.dataset import dataset
from cli.hf.model import model


@click.group()
def hf() -> None:
    """Inspect and transfer Studio assets on Hugging Face Hub.

    Commands operate on local Studio storage, not a remote Studio HTTP server.
    Use the same STORAGE_DIR and SETTINGS_FILE as the backend.

    Inspect first, then import into a project or export a local UUID. REPO_ID is
    OWNER/REPO on the Hub, not a Studio UUID. Import/export support --dry-run and
    print a --resume command before payload transfer. Each subcommand has --json.

    Authentication: saved Studio token, then HF_TOKEN, otherwise anonymous access.
    Cached hf auth login credentials are not used automatically. New export
    repositories are private unless --public is supplied.

    \b
    Start here:
      physicalai-studio hf model inspect OWNER/MODEL --json
      physicalai-studio hf dataset import --help
    """


hf.add_command(model)
hf.add_command(dataset)
