#!/usr/bin/env bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Check that the Alembic migration graph has exactly one head revision.
# Multiple heads indicate a migration conflict that must be resolved
# before merging. See application/backend/MIGRATIONS.md for details.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${BACKEND_DIR}"

# Capture alembic heads output
heads_output=$(uv run alembic -c src/alembic.ini heads 2>&1) || {
	echo "ERROR: Failed to run 'alembic heads'."
	echo "${heads_output}"
	exit 1
}

# Count the number of head revisions (non-empty lines)
head_count=$(echo "${heads_output}" | grep -c '(head)')

if [ "${head_count}" -eq 1 ]; then
	echo "OK: Single migration head detected."
	echo "${heads_output}"
	exit 0
elif [ "${head_count}" -eq 0 ]; then
	echo "ERROR: No migration heads found. This likely indicates a broken migration chain."
	echo "${heads_output}"
	exit 1
else
	echo "ERROR: Multiple migration heads detected (${head_count} heads)."
	echo ""
	echo "This means there are conflicting migrations with the same down_revision."
	echo "Detected heads:"
	echo "${heads_output}"
	echo ""
	echo "To fix this:"
	echo "  1. Pull the latest 'main' branch"
	echo "  2. Identify the current HEAD revision: uv run alembic -c src/alembic.ini heads"
	echo "  3. Update your migration's 'down_revision' to point to the latest head from main"
	echo "  4. Verify: uv run alembic -c src/alembic.ini heads (should show exactly 1 head)"
	echo ""
	echo "See application/backend/MIGRATIONS.md for detailed instructions."
	exit 1
fi
