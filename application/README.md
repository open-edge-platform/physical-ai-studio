<p align="center">
  <img src="../docs/assets/banner_application.png" alt="Geti Action Application" width="100%">
</p>

# Geti Action Application

Studio application for collecting demonstration data and managing VLA model training.

## Overview

The application provides a graphical interface to:

- **Collect** demonstration data from robotic systems
- **Manage** datasets and training configurations
- **Train** policies using the Geti Action library
- **Deploy** trained models to production

## Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **[Backend](./backend/)** | FastAPI server for data management and training orchestration | [Backend README](./backend/) |
| **[UI](./ui/)** | React web application | [UI README](./ui/README.md) |

## Quick Start

### Backend

```bash
cd backend
uv sync
source .venv/bin/activate
./run.sh
```

### Frontend

```bash
cd ui
npm install
npm run start
```

## See Also

- **[Library](../library/)** - Python SDK for programmatic usage
- **[Main Repository](../README.md)** - Project overview
