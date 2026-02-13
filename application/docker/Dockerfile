# syntax=docker/dockerfile:1

# Build arguments
ARG PYTHON_VERSION=3.12

# ===========================================================================
# Stage 1: Clone Geti UI packages and URDF robot models
# ===========================================================================
FROM node:24-alpine3.22@sha256:bb089be859f2741e5ede9d85f47dc7daca754015b50e9642a7a45c5252807d2c AS ui-packages

WORKDIR /home/app/web_ui/
RUN apk add --no-cache git
COPY --link ./application/ui/package.json ./
RUN npm run clone-geti-ui-packages

# Clone URDF files
RUN npm run clone-so101
RUN npm run clone-widowx

# ===========================================================================
# Stage 2: Build Web UI
# ===========================================================================
FROM node:24-alpine3.22@sha256:bb089be859f2741e5ede9d85f47dc7daca754015b50e9642a7a45c5252807d2c AS web-ui

WORKDIR /home/app/web_ui/

COPY --link application/ui/package.json ./
COPY --link application/ui/package-lock.json ./
COPY --from=ui-packages /home/app/web_ui/packages/ /home/app/web_ui/packages/
COPY --from=ui-packages /home/app/web_ui/public/ /home/app/web_ui/public/
RUN npm ci --audit=false --ignore-scripts

COPY --link application/ui/tsconfig.json ./
COPY --link application/ui/rsbuild.config.ts ./
COPY --link application/ui/src/ src/

RUN npm run build

# ===========================================================================
# Stage 3: Python builder base
#
# Contains build-time-only packages (g++, build-essential, etc.)
# and is NOT shipped in the final image. The .venv it produces is
# copied into the slim runtime stages below.
# ===========================================================================
FROM python:${PYTHON_VERSION}-slim@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab AS builder-base

COPY --from=docker.io/astral/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

# Build-time system dependencies (not shipped in final image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ \
        build-essential \
        libclang-dev \
        libusb-1.0-0-dev \
        pkg-config \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY --link pyproject.toml uv.lock LICENSE README.md ./

# Copy library source
COPY --link --from=libs ./src /app/library/src
COPY --link --from=libs ./LICENSE /app/library/LICENSE
COPY --link --from=libs ./README.md /app/library/README.md
COPY --link --from=libs ./pyproject.toml /app/library/pyproject.toml

# Copy backend application
COPY --link application/backend /app/application/backend

# Copy built UI assets
ARG STATIC_FILES_DIR="/app/application/ui/"
COPY --link --from=web-ui /home/app/web_ui/dist ${STATIC_FILES_DIR}

WORKDIR /app/application/backend

# Install hardware-independent dependencies (FastAPI, SQLAlchemy, etc.)
# first so this layer is shared across all device-specific builders.
RUN --mount=type=cache,id=uv-base,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ===========================================================================
# Stage 3a: CPU builder
# ===========================================================================
FROM builder-base AS builder-cpu

RUN --mount=type=cache,id=uv-cpu,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cpu

# ===========================================================================
# Stage 3b: XPU builder
# ===========================================================================
FROM builder-base AS builder-xpu

RUN --mount=type=cache,id=uv-xpu,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra xpu

# ===========================================================================
# Stage 3c: CUDA builder
# ===========================================================================
FROM builder-base AS builder-cuda

RUN --mount=type=cache,id=uv-cuda,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cuda

# ===========================================================================
# Stage 4: Runtime base — shared across all hardware targets
#
# Provides the non-root user, common runtime libraries, environment
# variables, and directory structure. Healthcheck is defined in
# docker-compose.yaml (not here) so it can reference runtime PORT.
# ===========================================================================
FROM python:${PYTHON_VERSION}-slim@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab AS runtime-base

# OCI image metadata
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.source="https://github.com/open-edge-platform/geti-action" \
      org.opencontainers.image.revision="${GIT_SHA}"

# uv is needed at runtime for `uv run` in run.sh
COPY --from=docker.io/astral/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /uvx /bin/

# Non-root user — override at build time to match host UID/GID for
# bind-mounted volumes (e.g. calibration data, HuggingFace cache).
ARG APP_UID=1000
ARG APP_GID=1000

RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home appuser

# Common runtime system dependencies (OpenCV, USB access)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create directories for mounted volumes and uv cache.
# /home/appuser/.cache must be pre-created so that bind mounts for
# subdirectories (e.g. huggingface calibration) don't cause Docker
# to create .cache as root, which blocks uv from writing its cache.
RUN mkdir -p /app/data /app/storage /app/tmp /home/appuser/.cache \
    && chown -R "${APP_UID}:${APP_GID}" /app /home/appuser/.cache

# Environment configuration
ENV PATH="/app/application/backend/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    STATIC_FILES_DIR="/app/application/ui/" \
    DATA_DIR="/app/data" \
    STORAGE_DIR="/app/storage" \
    PYTHONPATH="/app/application/backend"

EXPOSE 8000

# ===========================================================================
# Stage 5a: CPU runtime target
# ===========================================================================
FROM runtime-base AS geti-action-cpu

ARG APP_UID=1000
ARG APP_GID=1000
COPY --link --from=builder-cpu --chown=${APP_UID}:${APP_GID} /app /app

WORKDIR /app/application/backend
USER ${APP_UID}:${APP_GID}
CMD ["./run.sh"]

# ===========================================================================
# Stage 5b: XPU (Intel) runtime target
# ===========================================================================
FROM runtime-base AS runtime-xpu

# Intel GPU compute & media runtime from Kobuk PPA (Ubuntu Noble).
# The PPA packages work on Debian 13 (trixie); trusted=yes is required
# because Debian 13's sqv GPG verifier cannot validate the Launchpad key.
# Based on: https://dgpu-docs.intel.com/driver/client/overview.html
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo "deb [arch=amd64 trusted=yes] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" \
        > /etc/apt/sources.list.d/kobuk-intel.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        libze-intel-gpu1=26.01.36711.4-1~24.04~ppa1 \
        libze1=1.26.2-1~24.04~ppa1 \
        libze-dev=1.26.2-1~24.04~ppa1 \
        intel-opencl-icd=26.01.36711.4-1~24.04~ppa1 \
        intel-metrics-discovery=1.14.183-1~24.04~ppa1 \
        intel-gsc=0.9.5-1~24.04~ppa2 \
        intel-ocloc=26.01.36711.4-1~24.04~ppa1 \
        clinfo=3.0.25.02.14-1 \
        intel-media-va-driver-non-free=25.4.6-1~24.04~ppa1 \
        libmfx-gen1=25.4.0-0ubuntu1~24.04~ppa1 \
        libvpl2=1:2.16.0-1~24.04~ppa1 \
        libvpl-tools=1.5.0-1~24.04~ppa1 \
        libva-glx2=2.22.0-3 \
        va-driver-all=2.22.0-3 \
        vainfo=2.22.0+ds1-2 \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/sh", "-c"]

FROM runtime-xpu AS geti-action-xpu

ARG APP_UID=1000
ARG APP_GID=1000
COPY --link --from=builder-xpu --chown=${APP_UID}:${APP_GID} /app /app

WORKDIR /app/application/backend
USER ${APP_UID}:${APP_GID}
CMD ["./run.sh"]

# ===========================================================================
# Stage 5c: CUDA runtime target
# ===========================================================================
FROM runtime-base AS runtime-cuda

# NVIDIA CUDA 12.8 runtime libraries (matches cu128 PyTorch wheels).
# Requires NVIDIA Container Toolkit on the host for GPU passthrough.
#
# Uses the debian12 repo on Debian 13 (trixie) because the debian13 repo
# only has CUDA 13.1 packages (not 12.8). trusted=yes is required because
# Debian 13's sqv GPG verifier cannot validate the NVIDIA signing key.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo "deb [arch=amd64 trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64 /" \
        > /etc/apt/sources.list.d/nvidia-cuda.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-12-8=12.8.90-1 \
        libcublas-12-8=12.8.4.1-1 \
        libcudnn9-cuda-12=9.19.0.56-1 \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/sh", "-c"]

# ---------------------------------------------------------------------------
# EULA compliance cleanup — remove non-redistributable CUDA components.
# Keeps only the runtime shared libraries needed for inference.
# Based on: https://docs.nvidia.com/cuda/eula/index.html
# ---------------------------------------------------------------------------
RUN set -eux; \
    # Remove development tools (compilers, debuggers, profilers)
    apt-get remove -y --allow-remove-essential \
        cuda-compiler-12-8 cuda-cudart-dev-12-8 cuda-nvcc-12-8 \
        cuda-gdb-12-8 cuda-nsight-12-8 cuda-nsight-compute-12-8 \
        cuda-nsight-systems-12-8 || true; \
    # Remove headers and static libraries
    find /usr/local/cuda* /usr/lib/x86_64-linux-gnu/ \
        \( -path '*/include' -o -path '*/headers' \) -type d \
        -exec rm -rf {} + 2>/dev/null || true; \
    find /usr/local/cuda* /usr/lib/x86_64-linux-gnu/ \
        -name '*.a' -delete 2>/dev/null || true; \
    # Remove documentation and samples
    rm -rf /usr/local/cuda*/doc /usr/local/cuda*/samples \
           /usr/local/cuda*/extras/demo_suite; \
    # Remove CUPTI profiling libraries (non-redistributable)
    rm -rf /usr/local/cuda*/extras/CUPTI; \
    # Remove cuSolver multi-GPU component (requires separate license)
    find /usr/lib/x86_64-linux-gnu/ -name 'libcusolverMg*' -delete 2>/dev/null || true; \
    # Clean up
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

FROM runtime-cuda AS geti-action-cuda

ARG APP_UID=1000
ARG APP_GID=1000
COPY --link --from=builder-cuda --chown=${APP_UID}:${APP_GID} /app /app

WORKDIR /app/application/backend
USER ${APP_UID}:${APP_GID}
CMD ["./run.sh"]
