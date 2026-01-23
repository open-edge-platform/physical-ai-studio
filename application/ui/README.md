# Geti Action UI

React web application for the Geti Action studio.

## Overview

Web interface for:

- **Camera Management** - Configure and preview camera sources
- **Data Collection** - Record and manage demonstration datasets
- **Training** - Launch and monitor policy training jobs
- **Model Management** - Track trained models and deployments

## Setup

### Prerequisites

- Node.js 18+
- npm or pnpm
- Backend server running (see [Backend README](../backend/README.md))

### Install Dependencies

```bash
npm install
```

## Development

### Start Dev Server

```bash
npm run start
```

UI runs at http://localhost:3000

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Code Quality

```bash
# Format code
npm run format

# Lint code
npm run lint

# Type check
npm run type-check
```

## Testing

```bash
# Unit tests
npm run test:unit

# Component tests
npm run test:component
```

## Project Structure

```
ui/src/
├── api/          # API client and hooks
├── components/   # Reusable UI components
├── features/     # Feature-specific modules
├── routes/       # Page components
└── assets/       # Static assets
```

## Configuration

Environment variables (create `.env.local`):

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_WS_URL` | WebSocket URL | `ws://localhost:8000/ws` |

## See Also

- **[Application Overview](../README.md)** - Application components
- **[Backend](../backend/README.md)** - FastAPI backend service
- **[Library](../../library/README.md)** - Python SDK
