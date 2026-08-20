# Coding Standards

These standards apply to contributors and agents across the repository.

## Python

- Use `uv` for Python dependency and test commands; do not use `pip` directly.
- Add type hints to functions.
- Prefer `pathlib.Path` over string path manipulation.
- Use `ruff` for linting and formatting, and address all warnings.
- Use Google-style docstrings for public Python APIs.
- Use `logging` in library/application code; avoid `print()` except in CLI output paths.
- Prefer dataclasses or Pydantic models for structured data.

## TypeScript and React

- Use functional components and named exports.
- Keep TypeScript strict-mode compatible.
- Use React Query for server-state data fetching.
- Add route-level error boundaries for routes that perform data loading or orchestration.

## Writing Style

This applies to comments, docstrings, commit messages, and PR descriptions.

- State the point first.
- Use active voice.
- Avoid hedging (`may`, `might`, `could potentially`) unless uncertainty is real and relevant.
- Cut filler such as "It is important to note that", "Furthermore", and "Moreover".
- Comments explain why, not what.
- Use Conventional Commits for commit messages and PR titles (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

Prefer:

```text
The method improves performance.
```

Avoid:

```text
The methodology demonstrates significant improvements in terms of performance metrics.
```

## Testing

- Run Python tests with `uv run pytest` from the relevant project directory.
- Keep Python tests under `tests/unit/` and `tests/integration/`.
- Mock external services and hardware unless a test is explicitly marked as integration or download-dependent.
- Use Vitest for frontend unit tests and Playwright for end-to-end tests.

## AI and ML

- Version-control training configs.
- Log metrics and artifacts for training and evaluation runs.
- Lazy-load heavy dependencies.
- Account for inference latency and memory use when changing model/export paths.
