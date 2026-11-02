# Update Existing Installation

Use this guide after pulling new changes, so the UI and backend stay in sync.

First, make sure you are on the latest `main` branch:

```bash
git fetch origin
git rebase origin/main
```

## Update Docker setup

From `application/docker/`:

```bash
docker compose build
docker compose up -d --force-recreate
```

Then open `http://localhost:7860` and confirm your project loads.

## Update native setup

### Update the backend
To update the backend, first go to `./application/backend` and update its dependencies,

```bash
uv sync --extra xpu # or --extra cpu, --extra cuda
```

Then restart the backend by running `./run.sh` as mentioned in [Installation](./01-installation.md).

### Update the UI

Go to `./application/ui` and update its dependencies,

```bash
npm install
```

and start the UI by running `npm run start`.

## Next

- If this is your first run on a new machine, go to [Installation](./01-installation.md).
- For the full workflow, continue with [Getting Started](./03-getting-started.md).
