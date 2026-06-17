## Project Overview

Full-stack application:
- **Backend**: Python FastAPI (`application/backend/`)
- **Frontend**: React/TypeScript (`application/ui/`)
- **Library**: Vision-language-action policies (`library/`, `physicalai-train` package, `physicalai` import)

## Python

- **Always use `uv`** — never `pip` directly. Commands: `uv pip install`, `uv sync`, `uv run pytest`.
- Type hints on all functions. `pathlib.Path` over string paths.
- `ruff` for linting and formatting — address all warnings.
- Google-style docstrings. `logging` over `print()`.
- Prefer dataclasses or Pydantic models.

## TypeScript/React

- Functional components, named exports, TypeScript strict mode.
- React Query for data fetching. Error boundaries on route-level components.

## Writing Style

Applies to comments, docstrings, commit messages, and PR descriptions.

- State the point first. Active voice. No hedging ("may", "might", "could potentially").
- Cut filler: no "It is important to note that", "Furthermore", "Moreover".
- Comments explain *why*, not *what*.
- Commit messages and PR titles: conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

❌ "The methodology demonstrates significant improvements in terms of performance metrics."
✅ "The method improves performance."

## Testing

- Python: `pytest` via `uv run pytest`. Tests in `tests/unit/` and `tests/integration/`. Mock external dependencies.
- TypeScript: Vitest for unit tests, Playwright for E2E.

## Security

- For all `library/` code, follow `.github/instructions/lib.security.instructions.md` — read it before making changes.
- Validate all inputs. Store secrets in environment variables, never in source files.

## AI/ML

- Version-control training configs. Log metrics and artifacts.
- Lazy-load heavy dependencies. Mind inference latency and memory usage.
