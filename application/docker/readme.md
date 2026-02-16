# Deploying Geti Action with Docker

Geti Action is a framework for training and deploying Vision-Language-Action (VLA)
models for robotic imitation learning. This guide covers running Geti Action as a
Docker container with support for CPU, Intel XPU, and NVIDIA CUDA hardware.

## Prerequisites

- **Docker Engine** 24+ with **Docker Compose** v2
- A supported hardware backend:
  - **CPU** — any x86_64 system (default)
  - **Intel XPU** — Intel discrete/integrated GPU with Level Zero drivers on the host
  - **NVIDIA CUDA** — NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- Robot hardware (optional): USB serial ports (`/dev/ttyACM*`), USB cameras (`/dev/video*`), or Intel RealSense cameras

## Quick Start

All commands should be run from the `application/docker/` directory.

```bash
# 1. Create your local environment file
cp .env.example .env

# 2. (Optional) Auto-detect host device GIDs for non-Debian systems
./setup-devices.sh

# 3. Start Geti Action
docker compose up
```

Geti Action will be available at **http://localhost:7860**.

To run with a different hardware backend:

```bash
# Intel XPU
AI_DEVICE=xpu docker compose up

# NVIDIA CUDA
AI_DEVICE=cuda docker compose up
```

To run in the background, add the `-d` flag:

```bash
docker compose up -d
```

## Configuration

All configuration is done through the `.env` file. Copy `.env.example` to get started.

### Environment Variables

| Variable      | Default                       | Description                                          |
|---------------|-------------------------------|------------------------------------------------------|
| `AI_DEVICE`   | `cpu`                         | Hardware backend: `cpu`, `xpu`, or `cuda`            |
| `PORT`        | `7860`                        | Host port to expose the web UI and API               |
| `REGISTRY`    | `ghcr.io/open-edge-platform/` | Container image registry                             |
| `IMAGE_TAG`   | `latest`                      | Image version tag                                    |
| `APP_UID`     | `1000`                        | UID for the in-container user (match your host user) |
| `APP_GID`     | `1000`                        | GID for the in-container user (match your host user) |
| `VIDEO_GID`   | `video` (group name)          | Host GID for the `video` group (cameras)             |
| `DIALOUT_GID` | `dialout` (group name)        | Host GID for the `dialout` group (serial ports)      |
| `PLUGDEV_GID` | `plugdev` (group name)        | Host GID for the `plugdev` group (USB devices)       |
| `HTTP_PROXY`  | *(empty)*                     | HTTP proxy for builds and runtime                    |
| `HTTPS_PROXY` | *(empty)*                     | HTTPS proxy for builds and runtime                   |
| `NO_PROXY`    | *(empty)*                     | Proxy exclusion list                                 |

## Hardware Targets

The Docker image is built in multiple variants, one per hardware backend.
The correct variant is selected automatically based on `AI_DEVICE`.

| Build target       | `AI_DEVICE` | Additional system packages                 |
|--------------------|-------------|--------------------------------------------|
| `geti-action-cpu`  | `cpu`       | None                                       |
| `geti-action-xpu`  | `xpu`       | Intel GPU runtime (Level Zero, OpenCL, VA) |
| `geti-action-cuda` | `cuda`      | CUDA 12.8 runtime (cudart, cuBLAS, cuDNN)  |

## Hardware Access

The container needs access to host devices (serial ports, cameras, USB) to
communicate with robots and capture data. There are two modes:

### Option 1: Privileged Mode (default)

The default `docker-compose.yaml` runs the container in privileged mode. This
grants full access to all host devices with zero configuration:

```yaml
privileged: true
```

This is the easiest option and is recommended for development and prototyping.

### Option 2: Non-Privileged Mode (more secure)

For production or shared environments, you can restrict the container to only
the specific devices it needs. Edit `docker-compose.yaml`:

1. Comment out `privileged: true`
2. Uncomment the `devices` section and adjust the device paths to match your
   hardware:

```yaml
privileged: false
devices:
   # Servo USB connections
   - /dev/ttyACM0:/dev/ttyACM0
   - /dev/ttyACM1:/dev/ttyACM1
   # USB cameras
   - /dev/video0:/dev/video0
   - /dev/video2:/dev/video2
   # Intel RealSense (uncomment if needed)
   # - /dev/bus/usb:/dev/bus/usb
```

### Device Group Setup (non-Debian hosts)

Docker resolves group names like `video` and `dialout` against the
*container's* `/etc/group`, not the host's. On Debian/Ubuntu, the GIDs
typically match. On other distros (Arch Linux, Fedora, etc.) they may differ,
which means the container process won't have permission to access devices.

Run the setup script to auto-detect your host's GIDs and write them to `.env`:

```bash
# Detect GIDs and write to .env
./setup-devices.sh

# Check GIDs without modifying .env
./setup-devices.sh --check
```

The script detects `VIDEO_GID`, `DIALOUT_GID`, and `PLUGDEV_GID`, checks
whether your user is a member of each group, and warns you if you need to
add yourself:

```bash
sudo usermod -aG video $USER
sudo usermod -aG dialout $USER
# Log out and back in for group changes to take effect
```

## Volumes and Data Persistence

The compose file defines two named volumes and one bind mount:

| Volume         | Container path                                         | Purpose                                       |
|----------------|--------------------------------------------------------|-----------------------------------------------|
| `geti-data`    | `/app/data`                                            | Application database and runtime data         |
| `geti-storage` | `/app/storage`                                         | Trained models, datasets, and other artifacts |
| *(bind mount)* | `~/.cache/huggingface/lerobot/calibration` (read-only) | Shared robot calibration data from the host   |

To inspect volume contents:

```bash
# List files in a volume
docker run --rm -v geti-data:/data alpine ls -la /data

# Back up a volume
docker run --rm -v geti-storage:/storage -v $(pwd):/backup alpine \
  tar czf /backup/geti-storage-backup.tar.gz -C /storage .
```

To reset all data:

```bash
docker compose down -v
```

> **Warning:** `docker compose down -v` removes the named volumes and all data
> they contain (database, models, datasets). This cannot be undone.

## Building from Source

To build the Docker image locally instead of pulling a pre-built image:

```bash
# Build for CPU (default)
docker compose build

# Build for Intel XPU
AI_DEVICE=xpu docker compose build

# Build for NVIDIA CUDA
AI_DEVICE=cuda docker compose build
```

The Dockerfile uses a multi-stage build. The build context is the repository
root, and the image is built with the target `geti-action-${AI_DEVICE}`.
Proxy environment variables from `.env` are passed as build arguments
automatically.

## Troubleshooting

### Container fails to access serial ports or cameras

Ensure your host user is in the required device groups:

```bash
groups  # Check your current groups
sudo usermod -aG dialout,video,plugdev $USER
# Log out and back in
```

On non-Debian systems, also run `./setup-devices.sh` to set numeric GIDs
in `.env`.

### `ERROR: could not select device driver "" with capabilities: [[gpu]]`

The NVIDIA Container Toolkit is not installed or not configured. Follow the
[installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and restart the Docker daemon.

### Out of shared memory (`RuntimeError: DataLoader worker ... is killed`)

The default `shm_size` is set to 2 GB. If you still encounter shared memory
errors with large datasets, increase it in `docker-compose.yaml`:

```yaml
shm_size: 4g  # or higher
```

### Proxy issues during build or runtime

Set the proxy variables in `.env`:

```bash
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
NO_PROXY=localhost,127.0.0.1
```

These are passed to both the Docker build process and the running container.
