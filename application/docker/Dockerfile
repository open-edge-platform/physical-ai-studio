# syntax=docker/dockerfile:1@sha256:b6afd42430b15f2d2a4c5a02b919e98a525b785b1aaff16747d2f623364e39b6

# Build arguments
ARG PYTHON_VERSION=3.12

# ===========================================================================
# Stage 1: Clone Geti UI packages and URDF robot models
# ===========================================================================
FROM node:24-alpine3.22@sha256:76db75ca7e7da9148ae42c92d9be12d12a8d7b03e171f18339355d8078d644a0 AS ui-packages

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
FROM node:24-alpine3.22@sha256:76db75ca7e7da9148ae42c92d9be12d12a8d7b03e171f18339355d8078d644a0 AS web-ui

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
# Stage 3: App source assembly
#
# Collects all non-.venv application files into a single layer so that
# the final runtime stages can COPY them in one instruction. Built in
# parallel with the Python builder stages (no dependency between them).
# ===========================================================================
FROM scratch AS app-source

# Root workspace manifests (needed by uv run for workspace resolution)
COPY --link pyproject.toml uv.lock LICENSE README.md /app/

# Library source + manifest (needed by uv run for hatchling version)
COPY --link --from=libs ./src /app/library/src
COPY --link --from=libs ./LICENSE /app/library/LICENSE
COPY --link --from=libs ./README.md /app/library/README.md
COPY --link --from=libs ./pyproject.toml /app/library/pyproject.toml

# Backend source, entrypoint, and manifests
COPY --link application/backend/src /app/application/backend/src
COPY --link application/backend/pyproject.toml application/backend/uv.lock \
    /app/application/backend/
COPY --link application/backend/run.sh /app/application/backend/run.sh

# Built UI assets
COPY --link --from=web-ui /home/app/web_ui/dist /app/application/ui/

# ===========================================================================
# Stage 4: Python builder base
#
# Contains build-time-only packages (g++, build-essential, etc.)
# and dependency manifests — but NOT backend source code or UI assets.
# This separation ensures that source-only changes do not invalidate
# the expensive uv sync layers in the device-specific builders below.
# ===========================================================================
FROM python:${PYTHON_VERSION}-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4 AS builder-base

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

WORKDIR /app/application/backend

# Install hardware-independent dependencies (FastAPI, SQLAlchemy, etc.)
# first so this layer is shared across all device-specific builders.
# --no-install-local: skip building all local packages (the backend
# project AND the physicalai-train path dependency).  This avoids
# invoking hatchling, which would require library source files.
# The local packages are installed at runtime via app-source + uv run.
RUN --mount=type=cache,id=uv-base,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-local

# ===========================================================================
# Stage 4a: CPU builder
# ===========================================================================
FROM builder-base AS builder-cpu

RUN --mount=type=cache,id=uv-cpu,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-local --extra cpu

# ===========================================================================
# Stage 4b: XPU builder
# ===========================================================================
FROM builder-base AS builder-xpu

RUN --mount=type=cache,id=uv-xpu,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-local --extra xpu

# ===========================================================================
# Stage 4c: CUDA builder
# ===========================================================================
FROM builder-base AS builder-cuda

RUN --mount=type=cache,id=uv-cuda,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-local --extra cuda

# ===========================================================================
# Stage 4c: pyav + ffmpeg with hardware acceleration for Intel devices
# ===========================================================================
FROM python:${PYTHON_VERSION}-slim@sha256:7026274c107626d7e940e0e5d6730481a4600ae95d5ca7eb532dd4180313fea9 AS ffmpeg-xpu-builder

ARG DEBIAN_FRONTEND=noninteractive

# Pin refs/versions if you want reproducible builds.
ARG LIBVPL_REF=778a66d6c6537f08eabb91955dbbf1bce3812894
ARG FFMPEG_REF=00d1f41b2eb070803a927c6132c240e14142d0b9
ARG PYAV_VERSION=17.0.0
ARG OPENH264_REF=cf568c83f71a18778f9a16e344effaf40c11b752
ARG SVTAV1_REF=e7fcc226994c87b76b66389545eb73a844f13002

# Install build and runtime dependencies.
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
    patchelf \
    yasm \
    nasm \
    libdrm-dev \
    libva-dev \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packaging tools for building wheels.
RUN python -m pip install --upgrade pip setuptools wheel Cython auditwheel

# Build/install oneVPL dispatcher + headers.
ENV VPL_INSTALL_DIR=/opt/vpl
WORKDIR /tmp

RUN git clone --depth 1 https://github.com/intel/libvpl.git && \
    cd libvpl && \
    git checkout ${LIBVPL_REF} && \
    cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${VPL_INSTALL_DIR}" && \
    cmake --build build -j"$(nproc)" && \
    cmake --install build

# Make custom libvpl and FFmpeg visible to later build steps.
ENV FFMPEG_INSTALL_DIR=/opt/ffmpeg-vpl
ENV PKG_CONFIG_PATH="/opt/vpl/lib/pkgconfig:/opt/vpl/lib64/pkgconfig:/opt/ffmpeg-vpl/lib/pkgconfig:/usr/local/lib/pkgconfig"
ENV LD_LIBRARY_PATH="/opt/vpl/lib:/opt/vpl/lib64:/opt/ffmpeg-vpl/lib:/usr/local/lib"
ENV PATH="/opt/ffmpeg-vpl/bin:${PATH}"

# Optional software fallback encoders.
# Remove these two blocks if you do not want libopenh264 / libsvtav1.
WORKDIR /tmp

RUN git clone --depth 1 https://github.com/cisco/openh264.git && \
    cd openh264 && \
    git checkout ${OPENH264_REF} && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

RUN git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
    cd SVT-AV1 && \
    git checkout ${SVTAV1_REF} && \
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j"$(nproc)" && \
    cmake --install build && \
    ldconfig

# Build FFmpeg with libvpl enabled.
WORKDIR /tmp

RUN git clone --depth 1  https://github.com/FFmpeg/FFmpeg.git ffmpeg && \
    cd ffmpeg && \
    git checkout ${FFMPEG_REF} && \
    ./configure \
      --prefix="${FFMPEG_INSTALL_DIR}" \
      --extra-cflags="-I${VPL_INSTALL_DIR}/include" \
      --extra-ldflags="-L${VPL_INSTALL_DIR}/lib -L${VPL_INSTALL_DIR}/lib64" \
      --extra-libs="-lpthread -lm" \
      --enable-libvpl \
      --enable-vaapi \
      --enable-shared \
      --enable-libsvtav1 \
      --enable-libopenh264 && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

# Verify FFmpeg exposes QSV encoders.
RUN ffmpeg -version && \
    ffmpeg -hide_banner -encoders | grep qsv && \
    ffmpeg -hide_banner -h encoder=h264_qsv > /dev/null

# Build PyAV wheel from source against the FFmpeg we just built.
WORKDIR /build
RUN python -m pip wheel --no-binary av "av==${PYAV_VERSION}" -w /dist
# ===========================================================================
# Stage 5: Runtime base — shared across all hardware targets
#
# Provides the non-root user, common runtime libraries, environment
# variables, and directory structure. Healthcheck is defined in
# docker-compose.yaml (not here) so it can reference runtime PORT.
# ===========================================================================
FROM python:${PYTHON_VERSION}-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4 AS runtime-base

# uv is needed at runtime for `uv run` in run.sh
COPY --from=docker.io/astral/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac /uv /uvx /bin/

# Non-root user — override at build time to match host UID/GID for
# bind-mounted volumes (e.g. calibration data, HuggingFace cache).
ARG APP_UID=1000
ARG APP_GID=1000

RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home appuser
RUN groupadd -g 992 render_host || true && usermod -aG video,992 appuser

# Common runtime system dependencies (OpenCV, USB access, video decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libusb-1.0-0 \
        vainfo \
        intel-media-va-driver \
        libva2 \
        libva-drm2 \
        libmfx-gen1.2 \
    && rm -rf /var/lib/apt/lists/*

# Copy pyav wheel and runtime dependencies
COPY --link --from=ffmpeg-xpu-builder /dist /dist
COPY --link --from=ffmpeg-xpu-builder /opt/vpl /opt/vpl
COPY --link --from=ffmpeg-xpu-builder /opt/ffmpeg-vpl /opt/ffmpeg-vpl
COPY --link --from=ffmpeg-xpu-builder /usr/local/lib/libSvtAv1* /usr/local/lib/
COPY --link --from=ffmpeg-xpu-builder /usr/local/lib/libopenh264* /usr/local/lib/

# Let runtime use these paths
ENV LD_LIBRARY_PATH="/opt/vpl/lib:/opt/vpl/lib64:/opt/ffmpeg-vpl/lib:/usr/local/lib"
ENV PATH="/opt/ffmpeg-vpl/bin:${PATH}"

WORKDIR /app

# Create directories for mounted volumes and uv cache.
# /home/appuser/.cache must be pre-created so that bind mounts for
# subdirectories (e.g. huggingface calibration) don't cause Docker
# to create .cache as root, which blocks uv from writing its cache.
RUN mkdir -p /app/data /app/storage /app/tmp /home/appuser/.cache/huggingface \
    && chown -R "${APP_UID}:${APP_GID}" /app /home/appuser/.cache

# Environment configuration
ENV PATH="/app/application/backend/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    UV_NO_SYNC=1 \
    STATIC_FILES_DIR="/app/application/ui/" \
    DATA_DIR="/app/data" \
    STORAGE_DIR="/app/storage" \
    PYTHONPATH="/app/application/backend"

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

# Replace pyav
RUN uv pip uninstall av && uv pip install /dist/av-*.whl && \
    python - <<'PY'
import av
print("PyAV version:", av.__version__)
print("Library versions:", av.library_versions)
PY
RUN sed -i 's/VALID_VIDEO_CODECS = {"h264", "hevc", "libsvtav1", "auto"} | set(HW_ENCODERS)/VALID_VIDEO_CODECS = {"h264", "hevc", "libsvtav1", "auto"} | set(HW_ENCODERS) | {"av1_qsv"}/' /app/application/backend/.venv/lib/python3.12/site-packages/lerobot/datasets/video_utils.py

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

# Replace pyav
RUN uv pip uninstall av && uv pip install /dist/av-*.whl && \
    python - <<'PY'
import av
print("PyAV version:", av.__version__)
print("Library versions:", av.library_versions)
PY
RUN sed -i 's/VALID_VIDEO_CODECS = {"h264", "hevc", "libsvtav1", "auto"} | set(HW_ENCODERS)/VALID_VIDEO_CODECS = {"h264", "hevc", "libsvtav1", "auto"} | set(HW_ENCODERS) | {"av1_qsv"}/' /app/application/backend/.venv/lib/python3.12/site-packages/lerobot/datasets/video_utils.py

WORKDIR /app/application/backend
COPY application/docker/pyav-bench.py /app/application/backend

USER appuser
#CMD ["sleep","1000000"]
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

# Replace pyav
RUN uv pip uninstall av && uv pip install /dist/av-*.whl && \
    python - <<'PY'
import av
print("PyAV version:", av.__version__)
print("Library versions:", av.library_versions)
PY
RUN sed -i 's/VALID_VIDEO_CODECS = {"h264", "hevc", "libsvtav1", "auto"} | set(HW_ENCODERS)/VALID_VIDEO_CODECS = {"h264", "hevc", "libsvtav1", "auto"} | set(HW_ENCODERS) | {"av1_qsv"}/' /app/application/backend/.venv/lib/python3.12/site-packages/lerobot/datasets/video_utils.py

WORKDIR /app/application/backend
USER appuser
CMD ["./run.sh"]
