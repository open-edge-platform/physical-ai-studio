# Developer Guide: Working with the geti-action Monorepo

This repository is a monorepo containing the `geti-action` project, which is structured as a `uv` workspace. It includes a core `library` and an `application` that consumes it. This guide outlines the recommended workflows for developers.

## Project Structure

-   **/application**: Contains the main application, split into a Python `backend` and a `ui`.
-   **/library**: Contains the core, reusable Python library.
-   **pyproject.toml**: Defines the `uv` workspace, making `library` and `application/backend` its members.
-   **uv.lock**: The root lock file for the entire workspace, used for CI and integrated development.

## Development Workflows

This project supports two primary development workflows, designed to accommodate different teams and use cases.

---

### 1. Standalone Package Development (Recommended for Library/Backend Teams)

If you are working exclusively on the `library` or the `application/backend`, you can treat it as a standalone project. This workflow is ideal for focused development without needing to manage the entire workspace.

**Instructions:**

1.  **Navigate to your project directory:**
    ```bash
    # For the library team
    cd library

    # Or for the backend team
    cd application/backend
    ```

2.  **Install dependencies:** Use the local `uv.lock` to create a virtual environment with all necessary packages.
    ```bash
    uv sync
    ```

3.  **Activate the virtual environment (optional but recommended):**
    ```bash
    source .venv/bin/activate
    ```

4.  **Manage dependencies:** When adding or updating packages, `uv` will automatically update your local `pyproject.toml` and `uv.lock`.
    ```bash
    # Add a new dependency
    uv pip install <some-package>

    # Sync your environment after changes
    uv sync
    ```
    > **Note**: You do not need to worry about the root `uv.lock` file. The CI system will automatically update it when you push your changes.

---

### 2. Integrated Workspace Development (For Full-Stack and CI)

If you need to work on both the `library` and the `application` simultaneously, you should work from the project root. This gives you a single, unified environment.

**Instructions:**

1.  **Stay at the project root.**

2.  **Install all dependencies for the entire workspace:**
    ```bash
    uv sync
    ```

3.  **Activate the workspace virtual environment (optional):**
    ```bash
    source .venv/bin/activate
    ```

4.  **Manage dependencies:** When adding a dependency, you must specify which package it belongs to using the `-p` or `--package` flag.
    ```bash
    # Add a dependency to the 'library' package
    uv pip install -p library <some-package>

    # Add a dependency to the 'application/backend' package
    uv pip install -p application/backend <another-package>
    ```

5.  **Update the root lock file:** After modifying dependencies, regenerate the main `uv.lock`.
    ```bash
    uv lock
    ```

6.  **Sync your environment** with the new lock file.
    ```bash
    uv sync
    ```
