# syntax=docker/dockerfile:1@sha256:b6afd42430b15f2d2a4c5a02b919e98a525b785b1aaff16747d2f623364e39b6

ARG PYTHON_VERSION=3.13

# Build the trainer and its local physicalai-train dependency without copying
# Studio backend or UI sources into either builder or runtime images.
FROM python:${PYTHON_VERSION}-slim@sha256:eb43ff125d8d58d7449dcba7d336c23bcac412f526d861db493b9994d8010280 AS builder-base

COPY --from=docker.io/astral/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac /uv /uvx /bin/
# UV_PROJECT_ENVIRONMENT must match the runtime COPY destination (/opt/venv).
# uv embeds an absolute shebang in console-script entry points; if the venv
# is built at a different path the shebang breaks and the container cannot
# start.
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        git \
        libgl1 \
        libglib2.0-0 \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/application/trainer
COPY --link --from=libs / /app/library
COPY --link application/trainer /app/application/trainer

FROM builder-base AS builder-cuda
RUN --mount=type=cache,id=uv-trainer-cuda,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra cuda

# Remove test-only packages that executorch ships as unconditional runtime
# dependencies (the cuda extra pulls in physicalai-train[executorch]).
RUN uv pip uninstall --python /opt/venv/bin/python \
    pytest pluggy iniconfig \
    pytest-xdist execnet \
    pytest-json-report pytest-metadata \
    pytest-rerunfailures \
    expecttest hypothesis sortedcontainers kgb parameterized

FROM builder-base AS builder-xpu
RUN --mount=type=cache,id=uv-trainer-xpu,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra xpu

FROM python:${PYTHON_VERSION}-slim@sha256:eb43ff125d8d58d7449dcba7d336c23bcac412f526d861db493b9994d8010280 AS runtime-base

ARG TRAINER_UID=10001
ARG TRAINER_GID=10001
RUN groupadd --gid "${TRAINER_GID}" trainer \
    && useradd --uid "${TRAINER_UID}" --gid "${TRAINER_GID}" --create-home --shell /usr/sbin/nologin trainer \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall -y pip setuptools \
    && mkdir -p /var/lib/physicalai-trainer \
    && chown trainer:trainer /var/lib/physicalai-trainer

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    STORAGE_DIR=/var/lib/physicalai-trainer \
    HF_HOME=/var/lib/physicalai-trainer/huggingface \
    TORCH_HOME=/var/lib/physicalai-trainer/torch \
    XDG_CACHE_HOME=/var/lib/physicalai-trainer/cache

EXPOSE 8001
USER trainer
WORKDIR /var/lib/physicalai-trainer
ENTRYPOINT ["physicalai-trainer"]

FROM runtime-base AS runtime-xpu

# Intel GPU userspace is required in addition to the XPU PyTorch extra. The
# host remains responsible for the kernel driver and /dev/dri passthrough.
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update \
    && apt-get install -y --no-install-recommends gnupg curl ca-certificates \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL \
        "https://keyserver.ubuntu.com/pks/lookup?op=get&options=mr&exact=on&search=0x0C0E6AF955CE463C03FC51574D098D70AFBE5E1F" \
        | gpg --dearmor -o /etc/apt/keyrings/kobuk-intel.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/kobuk-intel.gpg] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" \
        > /etc/apt/sources.list.d/kobuk-intel.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clinfo \
        intel-gsc \
        intel-media-va-driver-non-free \
        intel-metrics-discovery \
        intel-ocloc \
        intel-opencl-icd \
        libze-dev \
        libze-intel-gpu1 \
        libze1 \
    && apt-get purge -y gnupg curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
SHELL ["/bin/sh", "-c"]
USER trainer

FROM runtime-base AS runtime-cuda

# NVIDIA CUDA 12.8 runtime libraries (matches cu128 PyTorch wheels).
# Requires NVIDIA Container Toolkit on the host for GPU passthrough.
#
# Uses the debian12 repo on Debian 13 (trixie) because the debian13 repo
# only has CUDA 13.1 packages (not 12.8).

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update \
    && apt-get install -y --no-install-recommends gnupg curl ca-certificates \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub \
        | gpg --dearmor -o /etc/apt/keyrings/nvidia-cuda.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64 /" \
        > /etc/apt/sources.list.d/nvidia-cuda.list \
    && printf '[hash_algorithms.sha1]\nsecond_preimage_resistance = "always"\n' > /etc/apt/nvidia-cuda-sequoia-policy.toml \
    && SEQUOIA_CRYPTO_POLICY=/etc/apt/nvidia-cuda-sequoia-policy.toml apt-get update \
    && SEQUOIA_CRYPTO_POLICY=/etc/apt/nvidia-cuda-sequoia-policy.toml apt-get install -y --no-install-recommends \
        cuda-cudart-12-8=12.8.90-1 \
        libcublas-12-8=12.8.4.1-1 \
        libcudnn9-cuda-12=9.19.0.56-1 \
    && apt-get purge -y gnupg curl \
    && apt-get autoremove -y \
    && rm -f /etc/apt/nvidia-cuda-sequoia-policy.toml \
    && rm -rf /var/lib/apt/lists/*
SHELL ["/bin/sh", "-c"]

# ---------------------------------------------------------------------------
# EULA compliance cleanup — remove non-redistributable CUDA system components.
# Keeps only the runtime shared libraries needed for inference/training.
# Based on: https://docs.nvidia.com/cuda/eula/index.html
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt-get remove -y --allow-remove-essential \
        cuda-compiler-12-8 cuda-cudart-dev-12-8 cuda-nvcc-12-8 \
        cuda-gdb-12-8 cuda-nsight-12-8 cuda-nsight-compute-12-8 \
        cuda-nsight-systems-12-8 || true; \
    find /usr/local/cuda* /usr/lib/x86_64-linux-gnu/ \
        \( -path '*/include' -o -path '*/headers' \) -type d \
        -exec rm -rf {} + 2>/dev/null || true; \
    find /usr/local/cuda* \
        -name '*.a' -delete 2>/dev/null || true; \
    rm -rf /usr/local/cuda*/doc /usr/local/cuda*/samples \
           /usr/local/cuda*/extras/demo_suite; \
    rm -rf /usr/local/cuda*/extras/CUPTI; \
    find /usr/lib/x86_64-linux-gnu/ -name 'libcusolverMg*' -delete 2>/dev/null || true; \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
USER trainer

FROM runtime-cuda AS physicalai-trainer-cuda
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
ARG APPLICATION_VERSION=unknown
ARG TRAINER_API_PROTOCOL_VERSION=1
LABEL org.opencontainers.image.source="https://github.com/open-edge-platform/physical-ai-studio" \
      org.opencontainers.image.description="Physical AI Studio remote CUDA training service" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${APPLICATION_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.title="physicalai-trainer-cuda" \
      org.opencontainers.image.vendor="Open Edge Platform" \
      org.open-edge-platform.physicalai.trainer.api-protocol="${TRAINER_API_PROTOCOL_VERSION}" \
      org.open-edge-platform.physicalai.trainer.device="cuda"
ENV TRAINER_BUILD_REVISION="${GIT_SHA}" \
    TRAINER_BUILD_DATE="${BUILD_DATE}" \
    TRAINER_APPLICATION_VERSION="${APPLICATION_VERSION}" \
    TRAINER_API_PROTOCOL_VERSION="${TRAINER_API_PROTOCOL_VERSION}" \
    TRAINER_DEVICE_TYPE=cuda
COPY --from=builder-cuda --chown=trainer:trainer /opt/venv /opt/venv

FROM runtime-xpu AS physicalai-trainer-xpu
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
ARG APPLICATION_VERSION=unknown
ARG TRAINER_API_PROTOCOL_VERSION=1
LABEL org.opencontainers.image.source="https://github.com/open-edge-platform/physical-ai-studio" \
      org.opencontainers.image.description="Physical AI Studio remote Intel XPU training service" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${APPLICATION_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.title="physicalai-trainer-xpu" \
      org.opencontainers.image.vendor="Open Edge Platform" \
      org.open-edge-platform.physicalai.trainer.api-protocol="${TRAINER_API_PROTOCOL_VERSION}" \
      org.open-edge-platform.physicalai.trainer.device="xpu"
ENV TRAINER_BUILD_REVISION="${GIT_SHA}" \
    TRAINER_BUILD_DATE="${BUILD_DATE}" \
    TRAINER_APPLICATION_VERSION="${APPLICATION_VERSION}" \
    TRAINER_API_PROTOCOL_VERSION="${TRAINER_API_PROTOCOL_VERSION}" \
    TRAINER_DEVICE_TYPE=xpu
COPY --from=builder-xpu --chown=trainer:trainer /opt/venv /opt/venv
