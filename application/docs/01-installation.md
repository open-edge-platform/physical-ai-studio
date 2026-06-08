# Installation

This guide helps you install Physical AI Studio and confirm that the UI is ready.

## Choose how to run

Docker is the recommended path for most users:

- Docker starts both backend and UI together.
- You open one URL and can begin setting up projects right away.

Use our native setup if you are planning to contribute to the project and want to make changes to either the backend or UI.

## Install with Docker (recommended)

Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) 24+ with Docker Compose v2.
For hardware-specific setup, see the [Docker README](../docker/README.md).

Then from `application/docker/`:

```bash
./setup-devices.sh --xpu # or use --cuda, or --cpu
docker compose up -d
```

When startup is done, open `http://localhost:7860`.

## First UI check

| **Project list**               |
|--------------------------------|
| ![Project list][projects-list] |

[projects-list]: ./assets/03-projects-list.png

After opening the app, you should see the projects page.
You can now create your first project. See [our getting started guide](./03-getting-started.md).

## If something does not open
[//]: # (Screenshot suggestion: Docker Desktop or terminal status indicating containers running, plus browser URL bar at localhost:7860.)

- Confirm Docker is running.
- Confirm no other app is already using the same port.
- Restart the stack from `application/docker/`.

## Alternatively install natively

You may choose to run the backend and UI directly on your system using your local python and node setup. This is mainly intended as a development setup, and requires installing prerequisites on your system.

### Prerequisites

Before starting the backend, install the system libraries.

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  libusb-1.0-0 \
  libusb-1.0-0-dev \
  libclang-dev \
  pkg-config \
  build-essential \
  g++ \
  git
```

If you are using another Linux distribution, install equivalent packages before running
the native setup steps below.

### Backend

Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
cd backend
uv sync --extra xpu # or `--extra cpu` or `--extra cuda`
./run.sh
```

The backend runs at http://localhost:7860

If you plan to train Hugging Face Hub-backed policies (for example, SmolVLA, Pi0,
and others), configure `HF_TOKEN` in `backend/.env`. See
[Hugging Face Integration](../backend/docs/huggingface_integration.md).

### Frontend

Install [Node.js v24](https://nodejs.org/en/download) (we recommend using nvm), then run:

```bash
cd ui
npm install
npm run start
```

UI runs at http://localhost:3000

## Next

- Continue with [Getting Started](./03-getting-started.md).
- If you are upgrading an existing setup, use [Update Existing Installation](./02-update-existing-installation.md).
