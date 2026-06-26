#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
APPLICATION_DIR="$(cd -- "${BACKEND_DIR}/.." && pwd)"
UI_DIR="${APPLICATION_DIR}/ui"

cd "${BACKEND_DIR}"

uv sync --frozen --extra xpu
uv run physicalai-studio gen-api --target-path openapi-spec.json

cd "${UI_DIR}"
npm ci
cp "${BACKEND_DIR}/openapi-spec.json" src/api/openapi-spec.json
npm run build:api
npm run build

if [[ ! -f "${UI_DIR}/dist/index.html" ]]; then
    echo "Missing ${UI_DIR}/dist/index.html after UI build" >&2
    exit 1
fi

cd "${BACKEND_DIR}"
rm -rf dist

# Dev/test builds: override version so TestPyPI accepts re-uploads
if [[ -n "${VERSION_OVERRIDE:-}" ]]; then
    sed -i 's/^version = .*/version = "'"${VERSION_OVERRIDE}"'"/' pyproject.toml
fi

uv run --with build==1.5.0 --with twine==6.2.0 python -m build --wheel
uv run --with twine==6.2.0 python -m twine check dist/*
