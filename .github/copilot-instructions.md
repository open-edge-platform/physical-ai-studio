# GitHub Copilot Instructions for PhysicalAI

> **Note**: This file is synchronized with `.cursorrules`. Both Cursor and VS Code (GitHub Copilot) use these same standards.

## Table of Contents

- [GitHub Copilot Instructions for PhysicalAI](#github-copilot-instructions-for-physical-ai-studio)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Coding Standards](#coding-standards)
    - [Python Environment Management](#python-environment-management)
    - [Python Code](#python-code)
    - [TypeScript/React Code](#typescriptreact-code)
    - [General Principles](#general-principles)
  - [Writing Style](#writing-style)
  - [Documentation Standards](#documentation-standards)
  - [Testing Guidelines](#testing-guidelines)
  - [File Organization](#file-organization)
  - [Git Commit Messages](#git-commit-messages)
  - [Pull Request Guidelines](#pull-request-guidelines)
  - [Performance Considerations](#performance-considerations)
  - [Security Best Practices](#security-best-practices)
  - [AI/ML Specific Guidelines](#aiml-specific-guidelines)
  - [Questions to Consider Before Coding](#questions-to-consider-before-coding)
  - [When Suggesting Code Changes](#when-suggesting-code-changes)

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
- Ensure `.venv` is in `.gitignore`

### Python Code

- Follow PEP 8
- Use type hints for all functions
- Prefer `pathlib.Path` over string paths
- Use `ruff` for linting and formatting
- Address all ruff warnings

**Docstrings** - Google style format:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised

    Examples:
        >>> result = function_name("test", 42)
        >>> print(result)
        True

        Multi-line example without prompt:

        from module import function_name

        result = function_name("test", 42)
        if result:
            print("Success")
    """
```

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

- **DRY**: Extract common logic
- **Single Responsibility**: One clear purpose per function/class
- **Error Handling**: Handle errors with informative messages
- **Testing**: Write tests for new functionality
- **Security**: Use environment variables for secrets
- **Performance**: Consider implications, especially for ML operations

## Writing Style

Apply to code comments, documentation, commit messages, and PR descriptions:

**Be Concise**

- Remove unnecessary words
- Avoid repeating ideas
- Use 10 words instead of 20
- Prefer short sentences

**Be Direct**

- State the point immediately
- Use active voice
- Remove hedging language ("may", "might", "could potentially")

**Use Simple Language**

- Choose simple words over complex ones
- Avoid jargon unless necessary
- Break complex sentences into shorter ones

**Sound Natural**

- Write as if explaining to a colleague
- Avoid formulaic transitions ("Furthermore", "Moreover", "Additionally")
- Don't use numbered lists when a paragraph works
- Avoid: "It is important to note that", "It should be mentioned", "It is worth noting"
- Vary sentence length naturally

**Academic But Accessible**

- Use technical terms when needed, explain domain-specific ones
- Prefer clarity over impressive vocabulary

**Examples:**

❌ "It is important to note that the workshop aims to establish a comprehensive platform that serves to bring together researchers from diverse backgrounds in order to facilitate meaningful collaboration."

✅ "The workshop brings together researchers from diverse backgrounds to facilitate collaboration."

❌ "The methodology demonstrates significant improvements in terms of performance metrics."

✅ "The method improves performance."

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
├── core/         # Core business logic
├── db/           # Database models and migrations
├── repositories/ # Data access layer
├── schemas/      # Pydantic schemas
├── services/     # Business logic services
└── utils/        # Utility functions
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

**Library**

```
library/src/physicalai/
├── configs/      # Configuration management
├── data/         # Data loading and processing
├── inference/    # Inference engine
├── policy/       # Policy implementations
└── trainer/      # Training logic
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
- Use async/await for I/O
- Implement caching
- Optimize database queries (indexes, avoid N+1)
- Profile before optimizing
- Consider memory usage for ML models

## Security Best Practices

- Validate and sanitize inputs
- Use parameterized queries
- Implement authentication and authorization
- Store secrets in environment variables
- Keep dependencies updated
- Follow OWASP guidelines

## AI/ML Specific Guidelines

- Document model architectures and hyperparameters
- Version control training configurations
- Log training metrics and artifacts
- Handle model inference errors
- Consider inference latency and throughput
- Document model limitations and assumptions

## Questions to Consider Before Coding

1. Does this align with project architecture?
2. Can I reuse existing utilities/components?
3. How will this be tested?
4. What error cases need handling?
5. Are there performance implications?
6. Does this need documentation?
7. Security considerations?

## When Suggesting Code Changes

- Explain reasoning
- Consider backward compatibility
- Highlight breaking changes
- Suggest related test updates
- Note configuration changes
- Consider impact on existing functionality
