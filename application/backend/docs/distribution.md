# Physical AI Studio Distribution

This document describes how the backend is packaged as the `physicalai-studio` Python distribution, how to test the wheel locally before publishing, and how the GitHub Actions publishing workflows behave.

## Usage Overview

Physical AI Studio can be deployed with an XPU, CUDA, or CPU training backend. The matching PyTorch wheel index must be passed via `--index` because wheel metadata cannot carry the source project's index configuration.

### Intel XPU

Use on Intel GPU systems:

```bash
uvx \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[xpu]" \
  physicalai-studio serve
```

### NVIDIA CUDA

Use on NVIDIA GPU systems:

```bash
uvx \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[cuda]" \
  physicalai-studio serve
```

### CPU

Use on systems without a supported GPU, or for simple smoke testing:

```bash
uvx \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[cpu]" \
  physicalai-studio serve
```

## Package Shape

The Python package is built from `application/backend`.

Important files:

```text
application/backend/
├── pyproject.toml
├── scripts/build_package.sh
├── scripts/hatch_build.py
├── src/
│   ├── cli/
│   ├── main.py
│   ├── alembic.ini
│   ├── alembic/
│   └── ...
└── dist/
```

The wheel contains:

- Backend Python modules from `application/backend/src`.
- Alembic configuration and migrations.
- A `physicalai-studio` console script with a `serve` command that runs the backend and UI together.
- The production UI build from `application/ui/dist`, packaged into the wheel as `webui/`.

Supported options: `--host 127.0.0.1 --port 7860`.

The UI is included through a `force_include` entry that `scripts/hatch_build.py` adds to `build_data` during non-editable builds. The hatch hook also prevents publishing a wheel without the frontend and transforms relative URLs in the app README to absolute GitHub URLs for the PyPI description.

## GitHub Actions

The wheel is build and published using GitHub Actions.

- **TestPyPI** — Publishes on every push to `main` (or manual dispatch). Appends `.dev<timestamp>` to the version for unique uploads.
- **PyPI** — Publishes when an `app/vX.Y.Z` tag is pushed. Validates the tag matches `application/VERSION` and `pyproject.toml` before building.

### Test From TestPyPI

When validating a release candidate from TestPyPI, include all three indexes:

- TestPyPI for `physicalai-studio`
- PyPI for normal dependencies (`torchao`, `fastapi`, etc.)
- PyTorch hardware index for `torch`/`torchvision` wheels

```bash
uvx \
  --index https://test.pypi.org/simple/ \
  --index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[xpu]==0.1.0" \
  physicalai-studio serve
```

## Build The Wheel Locally

From the repository root:

```bash
bash application/backend/scripts/build_package.sh
```

This syncs dependencies, generates the OpenAPI spec, builds the production UI, optionally patches the version (if `VERSION_OVERRIDE` is set), builds the wheel, and runs `twine check`.

The generated wheel is written to:

```text
application/backend/dist/
```

### Inspect Wheel Contents

Confirm that the wheel contains UI assets and migrations:

```python
from pathlib import Path
from zipfile import ZipFile

wheel = next(Path("application/backend/dist").glob("*.whl"))
with ZipFile(wheel) as zf:
    names = set(zf.namelist())

for name in (
    "webui/index.html",
    "alembic.ini",
    "alembic/env.py",
):
    print(f"{name}: {name in names}")

print("webui file count:", sum(name.startswith("webui/") for name in names))
```

Expected:

```text
webui/index.html: True
alembic.ini: True
alembic/env.py: True
webui file count: <non-zero>
```

### Test The Wheel Locally With uvx

Use the wheel directly, without publishing to PyPI:

```bash
WHEEL="/home/intel/physical-ai-studio/application/backend/dist/physicalai_studio-0.1.0-py3-none-any.whl"

uvx --isolated --no-cache \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[xpu] @ file://${WHEEL}" \
  physicalai-studio serve
```

Use `--no-cache` and `--isolated` to avoid reusing an installed tool environment.

## Common Issues

### `No module named 'pytorch-triton-xpu'`

The PyTorch XPU index was not supplied.

Add:

```bash
--index https://download.pytorch.org/whl/xpu \
--index-strategy unsafe-best-match
```

### Stale Local Wheel Is Used

If you rebuild the wheel without changing the version, `uv` may reuse a cached artifact.

Use:

```bash
--no-cache --isolated
```

Or point directly to the wheel file:

```bash
--from "physicalai-studio[xpu] @ file://${WHEEL}"
```

### Missing UI Assets In The Wheel

Run:

```bash
bash application/backend/scripts/build_package.sh
```

Do not build the app wheel before `application/ui/dist/index.html` exists. The Hatch build hook should fail wheel builds when the UI production build is missing.
