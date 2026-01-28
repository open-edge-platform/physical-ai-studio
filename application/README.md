<p align="center">
  <img src="../docs/assets/banner_application.png" alt="Geti Action Application" width="100%">
</p>

# Geti Action Application

Studio application for collecting demonstration data and managing VLA model training.

The application provides a graphical interface to:

- **Collect** demonstration data from robotic systems
- **Manage** datasets and training configurations
- **Train** policies using the Geti Action library
- **Deploy** trained models to production

<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="../docs/assets/application.gif" alt="Application demo" width="100%">
</p>
<!-- markdownlint-enable MD033 -->

## Components

| Component                 | Description                                                   | Documentation                         |
| ------------------------- | ------------------------------------------------------------- | ------------------------------------- |
| **[Backend](./backend/)** | FastAPI server for data management and training orchestration | [Backend README](./backend/README.md) |
| **[UI](./ui/)**           | React web application                                         | [UI README](./ui/README.md)           |

## Quick Start

Full setup instructions in component READMEs. Quick reference:

### Backend

```bash
cd backend
uv sync
source .venv/bin/activate
./run.sh
```

Backend runs at http://localhost:8000

### Frontend

```bash
cd ui
npm install
npm run start
```

UI runs at http://localhost:3000

## See Also

- **[Library](../library/)** - Python SDK for programmatic usage
- **[Main Repository](../README.md)** - Project overview
