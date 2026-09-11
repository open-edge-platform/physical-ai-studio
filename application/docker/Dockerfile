# syntax=docker/dockerfile:1@sha256:b6afd42430b15f2d2a4c5a02b919e98a525b785b1aaff16747d2f623364e39b6

# Build arguments
ARG PYTHON_VERSION=3.13

# ===========================================================================
# Stage 1: Build Web UI
# ===========================================================================
FROM node:24-alpine3.22@sha256:191c9f0080fcbbc6547a85dc0ff7988072214a355aabdc1d2ec55a7dae5eea8a AS web-ui

WORKDIR /home/app/web_ui/

COPY --link application/ui/package.json ./
COPY --link application/ui/package-lock.json ./
RUN npm ci --audit=false --ignore-scripts

COPY --link application/ui/tsconfig.base.json ./
COPY --link application/ui/tsconfig.json ./
COPY --link application/ui/rsbuild.config.ts ./
COPY --link application/ui/src/ src/

RUN npm run build

# ===========================================================================
# Stage 3: App source assembly
#
# Collects all non-.venv application files into a single layer so that
# the final runtime stages can COPY them in one instruction. Built in
# parallel with the Python builder stages (no dependency between them).
# ===========================================================================
FROM scratch AS app-source

# Root workspace manifests (needed by uv run for workspace resolution)
COPY --link pyproject.toml uv.lock LICENSE README.md /app/

# Application-level README (needed by the hatch build hook to transform
# relative URLs to absolute GitHub URLs for the PyPI description)
COPY --link application/README.md /app/application/README.md

# Backend source, entrypoint, manifests, and build hook
COPY --link application/backend/src /app/application/backend/src
COPY --link application/backend/scripts /app/application/backend/scripts
COPY --link application/backend/pyproject.toml application/backend/uv.lock \
    /app/application/backend/
COPY --link application/backend/README.md /app/application/backend/README.md
COPY --link application/backend/run.sh /app/application/backend/run.sh
COPY --link application/backend/scripts/hatch_build.py \
    /app/application/backend/scripts/hatch_build.py

# Built UI assets (subdirectory matches the hatch hook's expected path: ../ui/dist/)
COPY --link --from=web-ui /home/app/web_ui/dist /app/application/ui/dist

# ===========================================================================
# Stage 3b: LGPL FFmpeg + PyAV builder
#
# Builds an LGPL-licensed FFmpeg (no --enable-gpl / --enable-nonfree) with
# Intel oneVPL/QSV + VAAPI hardware acceleration and permissively licensed
# software encoders (libvpx, libopenh264, libsvtav1), then compiles a PyAV
# wheel against it. The stock Debian `ffmpeg` package is GPL-licensed and the
# PyPI `av` wheel bundles its own FFmpeg, so both are replaced by these
# artifacts in the runtime stages. Built in parallel with the other stages.
# ===========================================================================
FROM python:${PYTHON_VERSION}-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS ffmpeg-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# Pinned source revisions for reproducible builds.
ARG LIBVPL_REF=778a66d6c6537f08eabb91955dbbf1bce3812894
ARG FFMPEG_REF=9dd0fe0225c2c3406ca49430a74c8660b1713889
ARG OPENH264_REF=e3f5b10438e2bacc155cf54578222bd4236c9f06
ARG SVTAV1_REF=349f4eb9c369f68d463029d1cd37551c90b7de51
# Must match the `av` version resolved in application/backend/uv.lock.
ARG PYAV_VERSION=15.1.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        build-essential \
        cmake \
        meson \
        ninja-build \
        pkg-config \
        yasm \
        nasm \
        libdrm-dev \
        libva-dev \
        libnuma-dev \
        libvpx-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packaging tools for building the PyAV wheel.
RUN python -m pip install --no-cache-dir \
        "pip==25.*" \
        "setuptools==75.*" \
        "wheel==0.45.*" \
        "Cython==3.0.*"

ENV VPL_INSTALL_DIR=/opt/vpl
ENV FFMPEG_INSTALL_DIR=/opt/ffmpeg-vpl
WORKDIR /tmp

# oneVPL dispatcher + headers (enables the *_qsv codecs in FFmpeg).
RUN git init libvpl \
    && git -C libvpl remote add origin https://github.com/intel/libvpl.git \
    && git -C libvpl fetch --depth 1 origin "${LIBVPL_REF}" \
    && git -C libvpl checkout FETCH_HEAD \
    && cmake -B libvpl/build -S /tmp/libvpl \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${VPL_INSTALL_DIR}" \
    && cmake --build libvpl/build -j"$(nproc)" \
    && cmake --install libvpl/build

# Make the custom libvpl and FFmpeg visible to later build steps.
ENV PKG_CONFIG_PATH="/opt/vpl/lib/pkgconfig:/opt/vpl/lib64/pkgconfig:/opt/ffmpeg-vpl/lib/pkgconfig:/usr/local/lib/pkgconfig" \
    LD_LIBRARY_PATH="/opt/vpl/lib:/opt/vpl/lib64:/opt/ffmpeg-vpl/lib:/usr/local/lib" \
    PATH="/opt/ffmpeg-vpl/bin:${PATH}"

# Software fallback encoder: OpenH264 (BSD, Cisco).
RUN git init openh264 \
    && git -C openh264 remote add origin https://github.com/cisco/openh264.git \
    && git -C openh264 fetch --depth 1 origin "${OPENH264_REF}" \
    && git -C openh264 checkout FETCH_HEAD \
    && make -C openh264 -j"$(nproc)" \
    && make -C openh264 install \
    && ldconfig

# Software fallback encoder: SVT-AV1 (BSD).
RUN git init svtav1 \
    && git -C svtav1 remote add origin https://gitlab.com/AOMediaCodec/SVT-AV1.git \
    && git -C svtav1 fetch --depth 1 origin "${SVTAV1_REF}" \
    && git -C svtav1 checkout FETCH_HEAD \
    && cmake -B svtav1/build -S /tmp/svtav1 -DCMAKE_BUILD_TYPE=Release \
    && cmake --build svtav1/build -j"$(nproc)" \
    && cmake --install svtav1/build \
    && ldconfig

# FFmpeg — LGPL build: no --enable-gpl and no --enable-nonfree. Every enabled
# external library (libvpl/MIT, libvpx/BSD, libopenh264/BSD, libsvtav1/BSD,
# vaapi/MIT) is LGPL-compatible.
RUN git init ffmpeg \
    && git -C ffmpeg remote add origin https://github.com/FFmpeg/FFmpeg.git \
    && git -C ffmpeg fetch --depth 1 origin "${FFMPEG_REF}" \
    && git -C ffmpeg checkout FETCH_HEAD

WORKDIR /tmp/ffmpeg
RUN ./configure \
        --prefix="${FFMPEG_INSTALL_DIR}" \
        --extra-cflags="-I${VPL_INSTALL_DIR}/include" \
        --extra-ldflags="-L${VPL_INSTALL_DIR}/lib -L${VPL_INSTALL_DIR}/lib64" \
        --extra-libs="-lpthread -lm" \
        --enable-libvpl \
        --enable-vaapi \
        --enable-shared \
        --disable-static \
        --enable-libsvtav1 \
        --enable-libopenh264 \
        --enable-libvpx \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig

# Fail the build early if QSV encoders did not get wired in.
RUN ffmpeg -version \
    && ffmpeg -hide_banner -encoders | grep qsv \
    && ffmpeg -hide_banner -h encoder=h264_qsv > /dev/null

# Compile the PyAV wheel from source against the custom FFmpeg.
WORKDIR /build
RUN python -m pip wheel --no-binary av "av==${PYAV_VERSION}" -w /dist

# ===========================================================================
# Stage 4: Python builder base
#
# Contains build-time-only packages (g++, build-essential, etc.)
# and dependency manifests — but NOT backend source code or UI assets.
# This separation ensures that source-only changes do not invalidate
# the expensive uv sync layers in the device-specific builders below.
# ===========================================================================
FROM python:${PYTHON_VERSION}-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS builder-base

COPY --from=docker.io/astral/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac /uv /uvx /bin/
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

# --- Dependency manifests only (no source code, no docs) ---
#
# Only files that influence `uv sync --frozen` dependency resolution are
# copied here.  The library (physicalai-train) is a local path dependency
# that is NOT published to PyPI, but --no-install-local (used below)
# tells uv to skip building/installing all local packages — so we don't
# need library source, README, or LICENSE here.  However, uv still needs
# to *discover* the path dependency during project loading, so the
# library pyproject.toml must exist at the expected relative path.
# The real library source is overlaid via app-source in the final image.
COPY --link pyproject.toml uv.lock ./
COPY --link --from=libs ./pyproject.toml /app/library/pyproject.toml

# Backend manifests only (not src/ or run.sh) — sufficient for dep resolution.
COPY --link application/backend/pyproject.toml application/backend/uv.lock \
    /app/application/backend/

# Local plugin dependency manifest (required by backend's tool.uv.sources path).
# Keep source out of builder-base so source-only changes do not invalidate
# the shared hardware-independent sync layer.
COPY --link --from=plugin ./pyproject.toml \
    /app/application/plugin/pyproject.toml

WORKDIR /app/application/backend

# Install hardware-independent dependencies (FastAPI, SQLAlchemy, etc.)
# first so this layer is shared across all device-specific builders.
# --no-install-local: skip building all local packages (the backend
# project AND the physicalai-train path dependency).  This avoids
# invoking hatchling, which would require library source files.
# The local packages are installed at runtime via app-source + uv run.
# --no-install-package av: the PyPI `av` wheel bundles its own FFmpeg; it is
# replaced by the LGPL wheel from ffmpeg-builder in the runtime stages.
RUN --mount=type=cache,id=uv-base,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-local --no-install-package av

# FIXME: Remove this block after upgrading lerobot to >=0.5.2.
# lerobot==0.5.1 declares wandb <0.25.0 as an unconditional runtime dependency, which
# pulls in wandb-core a Go binary with several CVEs.
# Until the pin in application/backend/pyproject.toml is bumped to >=0.5.2,
# we delete the binary here to keep it out of the runtime image.
RUN find .venv -name "wandb-core" -delete

# ===========================================================================
# Stage 4a: CPU builder
# ===========================================================================
FROM builder-base AS builder-cpu

RUN --mount=type=cache,id=uv-cpu,target=/root/.cache/uv \
    --mount=type=bind,from=libs,target=/app/library \
    --mount=type=bind,from=plugin,target=/app/application/plugin \
    uv sync --frozen --no-dev --no-editable --no-install-project --no-install-package av --extra cpu

# Remove test-only packages that executorch ships as unconditional runtime dependencies.
RUN uv pip uninstall --python .venv/bin/python \
    pytest pluggy iniconfig \
    pytest-xdist execnet \
    pytest-json-report pytest-metadata \
    pytest-rerunfailures \
    expecttest hypothesis sortedcontainers kgb parameterized

# ===========================================================================
# Stage 4b: XPU builder
# ===========================================================================
FROM builder-base AS builder-xpu

RUN --mount=type=cache,id=uv-xpu,target=/root/.cache/uv \
    --mount=type=bind,from=libs,target=/app/library \
    --mount=type=bind,from=plugin,target=/app/application/plugin \
    uv sync --frozen --no-dev --no-editable --no-install-project --no-install-package av --extra xpu

# Remove test-only packages that executorch ships as unconditional runtime dependencies.
RUN uv pip uninstall --python .venv/bin/python \
    pytest pluggy iniconfig \
    pytest-xdist execnet \
    pytest-json-report pytest-metadata \
    pytest-rerunfailures \
    expecttest hypothesis sortedcontainers kgb parameterized

# Remove redundant for XPU container image Triton GPU backends (NVIDIA and AMD)
# that also contains non-redistributable files
# Based on: https://docs.nvidia.com/cuda/eula/index.html
RUN set -eu; \
    CLEANUP_ROOT="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    [ -n "$CLEANUP_ROOT" ] && [ -d "$CLEANUP_ROOT" ]; \
    find "$CLEANUP_ROOT" -type d -path "*/triton/backends/nvidia" \
        -exec rm -rf {} + 2>/dev/null || true; \
    find "$CLEANUP_ROOT" -type d -path "*/triton/backends/amd" \
        -exec rm -rf {} + 2>/dev/null || true

# ===========================================================================
# Stage 4c: CUDA builder
# ===========================================================================
FROM builder-base AS builder-cuda

RUN --mount=type=cache,id=uv-cuda,target=/root/.cache/uv \
    --mount=type=bind,from=libs,target=/app/library \
    --mount=type=bind,from=plugin,target=/app/application/plugin \
    uv sync --frozen --no-dev --no-editable --no-install-project --no-install-package av --extra cuda

# Remove test-only packages that executorch ships as unconditional runtime dependencies.
RUN uv pip uninstall --python .venv/bin/python \
    pytest pluggy iniconfig \
    pytest-xdist execnet \
    pytest-json-report pytest-metadata \
    pytest-rerunfailures \
    expecttest hypothesis sortedcontainers kgb parameterized

# ===========================================================================
# Stage 4d: Robot assets builder
#
# Sync backend-owned robot assets once so all runtime targets can copy the
# exact same files before the backend wheel install step.
# ===========================================================================
FROM builder-base AS robot-assets

COPY --link --from=app-source /app/application/backend/src /app/application/backend/src

RUN --mount=type=bind,from=plugin,target=/app/application/plugin \
    PYTHONPATH=/app/application/backend/src:/app/application/plugin/src \
    /app/application/backend/.venv/bin/python -c \
    "from robots.catalog.sync_robot_assets import sync_robot_assets; sync_robot_assets()"

# ===========================================================================
# Stage 5: Runtime base — shared across all hardware targets
#
# Provides the non-root user, common runtime libraries, environment
# variables, and directory structure. Healthcheck is defined in
# docker-compose.yaml (not here) so it can reference runtime PORT.
# ===========================================================================
FROM python:${PYTHON_VERSION}-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS runtime-base

# uv is needed at runtime for `uv run` in run.sh
COPY --from=docker.io/astral/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac /uv /uvx /bin/

# Non-root user — override at build time to match host UID/GID for
# bind-mounted volumes (e.g. calibration data, HuggingFace cache).
ARG APP_UID=1000
ARG APP_GID=1000

RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home appuser

# Common runtime system dependencies (OpenCV, USB access, video decoding)
# Includes runtime toolchain/linker deps required by torch.compile (Inductor/Triton).
# The GPL `ffmpeg` package is intentionally omitted; the LGPL FFmpeg binary and
# shared libraries from ffmpeg-builder are added below. libva*/libvpx/libnuma
# are the runtime counterparts of the -dev packages used to build them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        libusb-1.0-0 \
        libva2 \
        libva-drm2 \
        libvpx9 \
        libnuma1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Strip packaging tools, uv manages the pre-built venv, pip is never used at runtime.
RUN python -m pip uninstall -y setuptools pip \
    && rm -rf /root/.cache/pip

WORKDIR /app

# Create directories for mounted volumes and uv cache.
# /home/appuser/.cache must be pre-created so that bind mounts for
# subdirectories (e.g. huggingface calibration) don't cause Docker
# to create .cache as root, which blocks uv from writing its cache.
RUN mkdir -p /app/storage /app/tmp /home/appuser/.cache/huggingface \
    && chown -R "${APP_UID}:${APP_GID}" /app /home/appuser/.cache

# Environment configuration
ENV PATH="/app/application/backend/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    SYNC=false \
    UV_NO_SYNC=1 \
    HOST="0.0.0.0" \
    STATIC_FILES_DIR="/app/application/ui/dist/" \
    STORAGE_DIR="/app/storage" \
    AUTO_MIGRATE_STORAGE_DIR="true" \
    PYTHONPATH="/app/application/backend/src:/app/application/backend"

# Custom LGPL FFmpeg + oneVPL shared libraries and the LGPL FFmpeg CLI binary.
COPY --link --from=ffmpeg-builder /opt/vpl /opt/vpl
COPY --link --from=ffmpeg-builder /opt/ffmpeg-vpl /opt/ffmpeg-vpl
COPY --link --from=ffmpeg-builder /usr/local/lib/libSvtAv1* /usr/local/lib/
COPY --link --from=ffmpeg-builder /usr/local/lib/libopenh264* /usr/local/lib/

ENV LD_LIBRARY_PATH="/opt/vpl/lib:/opt/vpl/lib64:/opt/ffmpeg-vpl/lib:/usr/local/lib" \
    PATH="/opt/ffmpeg-vpl/bin:/app/application/backend/.venv/bin:$PATH"

EXPOSE 8000

# ===========================================================================
# Stage 6a: CPU runtime target
# ===========================================================================
FROM runtime-base AS physical-ai-studio-cpu

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.source="https://github.com/open-edge-platform/physical-ai-studio" \
      org.opencontainers.image.description="Physical AI Studio is an end-to-end framework for teaching robots to perform tasks through imitation learning from human demonstrations" \
      org.opencontainers.image.revision="${GIT_SHA}"

ARG APP_UID=1000
ARG APP_GID=1000

# Virtual environment from CPU builder; app files from app-source.
COPY --link --from=builder-cpu --chown=${APP_UID}:${APP_GID} \
    /app/application/backend/.venv /app/application/backend/.venv
COPY --link --from=app-source --chown=${APP_UID}:${APP_GID} /app /app
COPY --link --from=robot-assets --chown=${APP_UID}:${APP_GID} \
    /app/application/backend/src/static/robot-assets /app/application/backend/src/static/robot-assets
RUN uv pip install --python /app/application/backend/.venv/bin/python --no-deps /app/application/backend

# Replace PyAV with the wheel built against the custom LGPL FFmpeg.
RUN --mount=type=bind,from=ffmpeg-builder,source=/dist,target=/dist,ro \
    uv pip install --python /app/application/backend/.venv/bin/python \
        --no-deps --force-reinstall /dist/av-*.whl \
    && /app/application/backend/.venv/bin/python -c \
        "import av; print('PyAV', av.__version__, av.library_versions)"

WORKDIR /app/application/backend
USER appuser
CMD ["./run.sh"]

# ===========================================================================
# Stage 6b: XPU (Intel) runtime target
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
        libze-intel-gpu1 \
        libze1 \
        libze-dev \
        intel-opencl-icd \
        intel-metrics-discovery \
        intel-gsc \
        intel-ocloc \
        clinfo \
        intel-media-va-driver-non-free \
        libmfx-gen1 \
        libvpl2 \
        libvpl-tools \
        libva-glx2 \
        va-driver-all \
        vainfo \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/sh", "-c"]

FROM runtime-xpu AS physical-ai-studio-xpu

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.source="https://github.com/open-edge-platform/physical-ai-studio" \
      org.opencontainers.image.description="Physical AI Studio is an end-to-end framework for teaching robots to perform tasks through imitation learning from human demonstrations" \
      org.opencontainers.image.revision="${GIT_SHA}"

ARG APP_UID=1000
ARG APP_GID=1000

# Virtual environment from XPU builder; app files from app-source.
COPY --link --from=builder-xpu --chown=${APP_UID}:${APP_GID} \
    /app/application/backend/.venv /app/application/backend/.venv
COPY --link --from=app-source --chown=${APP_UID}:${APP_GID} /app /app
COPY --link --from=robot-assets --chown=${APP_UID}:${APP_GID} \
    /app/application/backend/src/static/robot-assets /app/application/backend/src/static/robot-assets
RUN uv pip install --python /app/application/backend/.venv/bin/python --no-deps /app/application/backend

# Replace PyAV with the wheel built against the custom LGPL FFmpeg.
RUN --mount=type=bind,from=ffmpeg-builder,source=/dist,target=/dist,ro \
    uv pip install --python /app/application/backend/.venv/bin/python \
        --no-deps --force-reinstall /dist/av-*.whl \
    && /app/application/backend/.venv/bin/python -c \
        "import av; print('PyAV', av.__version__, av.library_versions)"

WORKDIR /app/application/backend
USER appuser
CMD ["./run.sh"]

# ===========================================================================
# Stage 6c: CUDA runtime target
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
# EULA compliance cleanup — remove non-redistributable CUDA system components.
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
    # Keep system static archives (e.g. libc_nonshared.a) required by
    # torch.compile/Triton runtime linking; remove only CUDA static libs.
    find /usr/local/cuda* \
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

FROM runtime-cuda AS physical-ai-studio-cuda

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.source="https://github.com/open-edge-platform/physical-ai-studio" \
      org.opencontainers.image.description="Physical AI Studio is an end-to-end framework for teaching robots to perform tasks through imitation learning from human demonstrations" \
      org.opencontainers.image.revision="${GIT_SHA}"

ARG APP_UID=1000
ARG APP_GID=1000

# Virtual environment from CUDA builder; app files from app-source.
COPY --link --from=builder-cuda --chown=${APP_UID}:${APP_GID} \
    /app/application/backend/.venv /app/application/backend/.venv
COPY --link --from=app-source --chown=${APP_UID}:${APP_GID} /app /app
COPY --link --from=robot-assets --chown=${APP_UID}:${APP_GID} \
    /app/application/backend/src/static/robot-assets /app/application/backend/src/static/robot-assets
RUN uv pip install --python /app/application/backend/.venv/bin/python --no-deps /app/application/backend

# Replace PyAV with the wheel built against the custom LGPL FFmpeg.
RUN --mount=type=bind,from=ffmpeg-builder,source=/dist,target=/dist,ro \
    uv pip install --python /app/application/backend/.venv/bin/python \
        --no-deps --force-reinstall /dist/av-*.whl \
    && /app/application/backend/.venv/bin/python -c \
        "import av; print('PyAV', av.__version__, av.library_versions)"

WORKDIR /app/application/backend
USER appuser
CMD ["./run.sh"]
