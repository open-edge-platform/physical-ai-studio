# Physical AI Studio UI

React web application for the Physical AI Studio.

## Overview

Web interface for:

- **Camera Management** - Configure and preview camera sources
- **Data Collection** - Record and manage demonstration datasets
- **Training** - Launch and monitor policy training jobs
- **Model Management** - Track trained models and deployments
- **Inference** - Perform inference on robots using the trained models.

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

### Environment Variables

Create `.env.local` for custom configuration:

| Variable              | Description          | Default                 |
| --------------------- | -------------------- | ----------------------- |
| `PUBLIC_API_BASE_URL` | Backend API base URL | `http://localhost:3000` |

### Development Proxy

The dev server proxies `/api` requests to the backend:

```typescript
// rsbuild.config.ts
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:7860',  // Backend server
      changeOrigin: true,
      ws: true,  // WebSocket support
    },
  },
}
```

This allows the UI to make API calls to `/api/*` which are automatically forwarded to the backend at `http://localhost:7860`.

## See Also

- **[Application Overview](../README.md)** - Application components
- **[Backend](../backend/README.md)** - FastAPI backend service
- **[Library](../../library/README.md)** - Python SDK
