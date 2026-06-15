## Project Overview

Full-stack application with:

- **Backend**: Python FastAPI (`application/backend/`)
- **Frontend**: React/TypeScript (`application/ui/`)
- **Library**: Vision-language-action policies (`library/`)

## Coding Standards

### Python Environment Management

- **Always use `uv`** for package management and virtual environments
- Use `uv` generated virtual environments (`.venv`)
- Install with `uv pip install` or `uv sync`
- Create environments with `uv venv`
- Never use `pip` directly

### Python Code

- Follow PEP 8
- Use type hints for all functions
- Prefer `pathlib.Path` over string paths
- Use `ruff` for linting and formatting
- Address all ruff warnings
- Use Google style docstrings
- Use `logging` instead of `print()`
- Prefer dataclasses or Pydantic models
- Use context managers for resource management

### TypeScript/React Code

- Use functional components with hooks
- Prefer named exports over default exports
- Use TypeScript strict mode with explicit types
- Follow component structure in `application/ui/src/`
- Use proper prop types and interfaces
- Implement error boundaries
- Use React Query for data fetching

### General Principles

DRY, single responsibility, informative error messages, and tests for new functionality. Store secrets in environment variables. Mind performance of ML operations.

## Writing Style

Apply to code comments, documentation, commit messages, and PR descriptions:

- **Be concise**: cut unnecessary words, avoid repetition, prefer short sentences.
- **Be direct**: state the point first, use active voice, drop hedging ("may", "might", "could potentially").
- **Use simple language**: plain words over complex ones, explain domain-specific terms.
- **Sound natural**: write as if explaining to a colleague. Avoid formulaic transitions ("Furthermore", "Moreover") and filler ("It is important to note that", "It is worth noting"). Vary sentence length.
- **Clarity over impressive vocabulary.**

Example — ❌ "The methodology demonstrates significant improvements in terms of performance metrics." ✅ "The method improves performance."

## Documentation Standards

**Code Comments**

- Write self-documenting code
- Add comments only for the "why", not the "what"
- Update comments when code changes

**README Files**

- Each major component needs a README.md
- Include: purpose, installation, usage examples, configuration, troubleshooting

**API Documentation**

- Document endpoints with OpenAPI/Swagger
- Include request/response examples
- Document error responses and status codes

**Inline Documentation**

- Use JSDoc for TypeScript/JavaScript
- Use docstrings for Python
- Document parameters, return values, exceptions
- Include usage examples for complex functions

## Testing Guidelines

**Python Tests**

- Use `pytest` with `uv` (e.g., `uv run pytest`)
- Place in `tests/unit/` and `tests/integration/`
- Name files: `test_<module_name>.py`
- Name functions: `test_<feature>_<scenario>`
- Use fixtures from `conftest.py`
- Aim for >80% coverage
- Mock external dependencies

**TypeScript Tests**

- Use Vitest for unit tests
- Use Playwright for E2E tests
- Name files: `<component>.test.ts(x)`
- Use descriptive `describe` and `it` blocks
- Mock API calls and dependencies

## File Organization

**Backend**

```
application/backend/src/
├── api/          # API routes and endpoints
├── control/      # Robot control logic
├── core/         # Core business logic
├── db/           # Database models and migrations
├── internal_datasets/ # Built-in datasets
├── middleware/   # Request/response middleware
├── models/       # Domain models
├── repositories/ # Data access layer
├── robots/       # Robot integrations
├── schemas/      # Pydantic schemas
├── services/     # Business logic services
├── utils/        # Utility functions
└── workers/      # Background workers
```

**Frontend**

```
application/ui/src/
├── api/          # API client and hooks
├── components/   # Reusable UI components
├── features/     # Feature-specific code
├── routes/       # Page components
└── assets/       # Static assets
```

**Library** (`physicalai-train` distribution, `physicalai` import package)

```
library/src/physicalai/
├── benchmark/    # Benchmarking
├── cli/          # Studio CLI subcommands
├── config/       # Configuration management
├── data/         # Data loading and processing
├── devices/      # Device handling
├── eval/         # Evaluation
├── export/       # Artifact export
├── gyms/         # Environments
├── inference/    # Inference engine
├── policies/     # Policy implementations
├── train/        # Training logic
└── transforms/   # Data transforms
```

## Git Commit Messages

Use conventional commits:

- `feat:` - new features
- `fix:` - bug fixes
- `docs:` - documentation changes
- `refactor:` - code refactoring
- `test:` - adding tests
- `chore:` - maintenance tasks

Write clear, concise messages. Reference issue numbers.

## Pull Request Guidelines

- Use the PR template (`.github/pull_request_template.md`)
- Fill out: Description, Type of Change, Related Issues, Changes Made, Examples, Breaking Changes
- Provide usage examples and before/after comparisons
- Follow conventional commit format for PR title
- **Tip**: Draft PRs in `tmp_PR_TEMPLATE_<branch-name>.md` for preview

## Performance Considerations

- Lazy load heavy dependencies
- Consider memory usage, inference latency, and throughput for ML models

## Security Best Practices

- For `library/` code, follow `.github/instructions/lib.security.instructions.md`
- Validate inputs; use parameterized queries

## AI/ML Specific Guidelines

- Version-control training configurations; log training metrics and artifacts
- Document model architectures, hyperparameters, limitations, and assumptions


## When Suggesting Code Changes

- Explain reasoning
- Consider backward compatibility
- Highlight breaking changes
- Suggest related test updates
- Note configuration changes
- Consider impact on existing functionality
