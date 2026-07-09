# Physical AI Studio Agent Guide

Physical AI Studio is the training-side repo for the Physical AI workflow: collect data, train policies, benchmark, and export artifacts that the Runtime package loads for robot deployment.

## Repository Layout

- `library/`: `physicalai-train`, the Python library for data, policies, training, benchmarking, export, and Studio-owned `physicalai` CLI subcommands.
- `application/backend/`: FastAPI backend for the GUI and orchestration workflows.
- `application/ui/`: React frontend that consumes generated OpenAPI types.
- `skills/library/` and `skills/application/`: agent skills (canonical). Adapter symlinks under `.claude/skills/` and `.agents/skills/` are committed so clones work out of the box. See `skills/README.md`.

## Setup

- Library: run `uv sync` from `library/`.
- Backend: run `uv sync --extra xpu`, `uv sync --extra cuda`, or `uv sync --extra cpu` from `application/backend/`.
- UI: run `npm install` from `application/ui/`.
- Docker GUI: use `application/docker/setup-devices.sh --xpu|--cuda|--cpu` and `docker compose up -d` from `application/docker/`.

## Build, Test, Lint

- Run repo hooks with `prek run --all-files` from the repo root.
- Limit hooks to the library with `prek run --all-files library/`.
- Limit hooks to the backend with `prek run --all-files application/backend/`.
- Regenerate UI API types with `npm run build:api:download && npm run build:api` from `application/ui/` while the backend is serving OpenAPI.

## Cross-Repo Rules

- Runtime owns the `physicalai` executable and `pai` alias. Studio contributes subcommands through `physicalai.cli.subcommands`.
- Studio-owned CLI subcommands include `fit`, `validate`, `test`, `predict`, `benchmark`, and `export`.
- Studio owns the export side of the export/load contract. Runtime consumes exported artifacts with `InferenceModel(...)`.
- Keep customer-facing instructions stable and avoid exposing internal scaffolding unless the user is contributing to the repo.

## Contribution Notes

- Use Conventional Commits for PR titles and commits.
- Sign commits when committing changes.
- Follow `docs/development/coding-standards.md` for repo-wide coding standards.
- For `library/` code, follow `library/docs/development/security.md`.
