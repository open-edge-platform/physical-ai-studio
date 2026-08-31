# Physical AI Studio Backend

FastAPI server for demonstration data management and VLA model training orchestration.

## Overview

The backend provides RESTful APIs and services for:

- **Camera Management** - Configure and stream from multiple camera sources (RealSense, USB, GenICam)
- **Dataset Management** - Store and organize demonstration recordings
- **Training Orchestration** - Launch and monitor policy training jobs
- **Model Management** - Track trained models and export configurations
- **WebRTC Streaming** - Real-time video streaming for data collection

## Architecture

```
backend/src/
├── api/          # FastAPI route handlers
├── core/         # Business logic and domain models
├── db/           # Database models and migrations (SQLAlchemy + Alembic)
├── repositories/ # Data access layer
├── schemas/      # Pydantic request/response schemas
├── services/     # Business logic services
├── utils/        # Shared utilities
├── webrtc/       # WebRTC signaling and streaming
└── workers/      # Background task workers
```

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Install Dependencies

```bash
cd application/backend
uv sync
```

This installs all backend dependencies including FastAPI, SQLAlchemy, aiortc, and the physicalai library.

### (Optional) Enable hardware acceleration for video encoding
Using hardware acceleration for video encoding can improve the speed of recording significantly.
Please check out [this document](docs/video_hardware_acceleration_intel.md) for more information.

## Usage

### Start Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Run server (backend with in-process training; local by default)
uv run physicalai-studio serve

# Equivalent thin wrapper
./run.sh
```

Server starts at `http://localhost:7860` by default.

To change the host/port:

```bash
# Option A: CLI flags
uv run physicalai-studio serve --host 127.0.0.1 --port 8000

# Option B: environment variables
HOST=127.0.0.1
PORT=8000
```

### Remote Training

The `serve` process supports local and remote training at the same time. Configure
remote trainer URLs in the Studio UI, then choose the execution target when you
submit a training job. `run.sh` is a thin wrapper around the CLI.

To run training on a separate, GPU-enabled machine, deploy and configure a
Physical AI Trainer service from
[`docs/remote-trainer.md`](docs/remote-trainer.md), then register its URL
as a remote trainer in the Studio UI. The backend sends dataset snapshots to
the service, monitors the training job, and imports the resulting model.

Alternatively, the backend can provision a job-scoped trainer over SSH on a
server you can reach directly. This feature is off by default and has no
authentication model of its own — see
[`docs/explanation/ssh-remote-trainer.md`](docs/explanation/ssh-remote-trainer.md) before enabling it.

### Database Migrations

```bash
# Create new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback migration
uv run alembic downgrade -1
```

### CLI Commands

```bash
# Initialize database
uv run physicalai-studio db init

# Run migrations
uv run physicalai-studio db migrate
```

## API Documentation

Once the server is running:

- **Interactive API Docs** - http://localhost:7860/docs (Swagger UI)
- **Alternative Docs** - http://localhost:7860/redoc (ReDoc)
- **OpenAPI Schema** - http://localhost:7860/api/openapi.json

## Configuration

Configuration via environment variables (see `src/settings.py`):

| Variable      | Description                                                                                                  | Default                                                                                                 |
|---------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `STORAGE_DIR` | Root directory for persistent artifacts (`datasets/`, `models/`, `snapshots/`, `robots/`, `cache/`, `logs/`) | Linux: `${XDG_DATA_HOME:-~/.local/share}/physicalai`; macOS: `~/Library/Application Support/physicalai` |

Create `.env` file in backend directory for local overrides.

## Development

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run mypy src/

# Type check (Pyrefly)
uv run pyrefly check -c pyproject.toml
```

### Project Structure

- **API Layer** (`api/`) - HTTP endpoints, request validation
- **Service Layer** (`services/`) - Business logic, orchestration
- **Repository Layer** (`repositories/`) - Database queries
- **Core** (`core/`) - Domain models and pure business logic
- **Schemas** (`schemas/`) - Input/output data validation

### Adding New Endpoints

1. Define Pydantic schemas in `schemas/`
2. Create repository methods in `repositories/`
3. Implement service logic in `services/`
4. Add route handlers in `api/`
5. Register routes in `main.py`

## Troubleshooting

### Data/Storage Migration Behavior

On startup (`./run.sh`), the backend runs migration checks before Alembic:

- Storage migration: old `~/.cache/physicalai` -> `STORAGE_DIR`
- Database migration: old `data/physicalai.db`, Docker legacy `/app/data/physicalai.db`, or a legacy `$DATA_DIR/physicalai.db` -> `$STORAGE_DIR/data/physicalai.db`

In interactive terminals, users are prompted for confirmation when a migration is needed.

### Camera Not Detected

- **RealSense**: Install [librealsense](https://github.com/IntelRealSense/librealsense)
- **GenICam**: Install vendor-specific SDKs
- **USB**: Check permissions (`sudo usermod -a -G video $USER`)

## See Also

- **[Application Overview](../README.md)** - Full application architecture
- **[UI](../ui/README.md)** - React frontend
- **[Library](../../library/README.md)** - Python SDK for training
