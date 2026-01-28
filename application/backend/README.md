# Geti Action Backend

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
- RealSense SDK (optional, for Intel RealSense cameras)

### Install Dependencies

```bash
cd application/backend
uv sync
```

This installs:
- FastAPI and Uvicorn for the web server
- SQLAlchemy and Alembic for database management
- aiortc for WebRTC streaming
- getiaction library for training integration
- Camera SDKs (OpenCV, RealSense, GenICam)

## Usage

### Start Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Run server
./run.sh
```

Server starts at `http://localhost:8000`

### Seed Database (Development)

```bash
# Seed with sample data and pre-trained model
SEED_DB=true ./run.sh
```

**Note**: Ensure model artifacts are uploaded before seeding.

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
uv run src/cli.py init-db

# Seed database with sample data
uv run src/cli.py seed --with-model=True
```

## API Documentation

Once the server is running:

- **Interactive API Docs** - http://localhost:8000/docs (Swagger UI)
- **Alternative Docs** - http://localhost:8000/redoc (ReDoc)
- **OpenAPI Schema** - http://localhost:8000/openapi.json

## Configuration

Configuration via environment variables (see `src/settings.py`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | SQLite database path | `sqlite+aiosqlite:///./geti_action.db` |
| `CORS_ORIGINS` | Allowed CORS origins | `["http://localhost:3000"]` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `SEED_DB` | Seed database on startup | `false` |

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

### Database Locked Error

SQLite doesn't handle high concurrency well. For production, use PostgreSQL:

```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost/geti_action"
```

### Camera Not Detected

- **RealSense**: Install [librealsense](https://github.com/IntelRealSense/librealsense)
- **GenICam**: Install vendor-specific SDKs
- **USB**: Check permissions (`sudo usermod -a -G video $USER`)

### WebRTC Connection Issues

- Ensure firewall allows UDP traffic
- Check browser console for ICE candidate errors
- Verify STUN/TURN server configuration

## See Also

- **[Application Overview](../README.md)** - Full application architecture
- **[UI](../ui/README.md)** - React frontend
- **[Library](../../library/README.md)** - Python SDK for training
