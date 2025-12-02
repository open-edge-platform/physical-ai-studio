# Developer Guide: Working with the geti-action Monorepo

This repository is a monorepo containing the `geti-action` project. The `library` contains the Vision-Language-Action (VLA) framework for training and inference, while the `application` is the studio application with backend and UI components. Each project has its own independent virtual environment and tooling configuration.

## Project Structure

```
geti-action/
├── library/                    # VLA framework (ML engineers)
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version
│   └── .pre-commit-config.yaml
├── application/
│   ├── backend/                # FastAPI server (Backend engineers)
│   │   ├── pyproject.toml
│   │   ├── uv.lock
│   │   ├── .python-version
│   │   └── .pre-commit-config.yaml
│   └── ui/                     # React/TypeScript frontend
├── .pre-commit-config.yaml     # Root pre-commit hooks (universal)
└── pyproject.toml              # Root configuration (minimal)
```

Each project directory contains its own:

- `pyproject.toml`: Dependencies and tool configurations (`[tool.ruff]`, `[tool.mypy]`, etc.)
- `uv.lock`: Lock file for reproducible environments
- `.python-version`: Python version specification
- `.pre-commit-config.yaml`: Project-specific pre-commit hooks (used by prek)

## Development Workflows

### Library Development (ML Engineers)

1. **Navigate to the library directory:**

   ```bash
   cd library
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**

   ```bash
   source .venv/bin/activate
   ```

4. **Run tests:**

   ```bash
   uv run pytest tests/unit
   ```

5. **Manage dependencies:**

   ```bash
   # Add a new dependency
   uv add <package-name>

   # Update lock file
   uv lock
   ```

---

### Backend Development (Backend Engineers)

1. **Navigate to the backend directory:**

   ```bash
   cd application/backend
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**

   ```bash
   source .venv/bin/activate
   ```

4. **Run the server:**

   ```bash
   ./run.sh
   ```

5. **Manage dependencies:**

   ```bash
   # Add a new dependency
   uv add <package-name>

   # Update lock file
   uv lock
   ```

> **Note**: The backend includes `getiaction` (the library) as an editable dependency, so changes to the library are immediately available in the backend environment.

---

## Code Quality with prek

This project uses [prek](https://prek.j178.dev/) for pre-commit hooks. Prek is a fast, Rust-based reimplementation of pre-commit with built-in monorepo support.

### Workspace Structure

```
geti-action/
├── .pre-commit-config.yaml     # Root: universal hooks
├── library/
│   └── .pre-commit-config.yaml # Library-specific hooks
└── application/
    └── backend/
        └── .pre-commit-config.yaml # Backend-specific hooks
```

### Installing prek

```bash
# Using cargo
cargo install prek

# Or using the install script
curl -fsSL https://prek.j178.dev/install.sh | sh
```

### Running Hooks

```bash
# Run all hooks on all files
prek run --all-files

# Run only library hooks
prek run --all-files library/

# Run only backend hooks
prek run --all-files application/backend/

# Run specific hook across all projects
prek run ruff

# Run on staged files only (default behavior)
prek run
```

### Installing Git Hooks

```bash
# Install hooks to run automatically on git commit
prek install
```

### Hook Configuration

Each project's hooks use the `[tool.*]` configurations from their respective `pyproject.toml`:

- **Library** (`library/pyproject.toml`): `[tool.ruff]`, `[tool.mypy]`, `[tool.bandit]`
- **Backend** (`application/backend/pyproject.toml`): `[tool.ruff]`, `[tool.mypy]`

---

## CI/CD

The GitHub Actions workflows are configured to:

1. **Library** (`library.yml`):

   - Run prek hooks on library files
   - Run unit tests

2. **Backend** (`backend.yml`):

   - Run ruff and mypy
   - Verify lock file integrity

3. **UI** (`ui.yml`):
   - Build and test the frontend
