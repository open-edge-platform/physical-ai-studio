# Alembic Migration Guide

This document covers how database migrations work in this project, how conflicts arise, and how to prevent and resolve them.

Tl/dr: Run the script below to check if you have migration conflicts.

```bash
cd application/backend
bash scripts/check_alembic_heads.sh
```

## Overview

We use [Alembic](https://alembic.sqlalchemy.org/) to manage database schema changes. Migrations live in `src/alembic/versions/` and form a **linear chain** — each migration has a `revision` (its own ID) and a `down_revision` (the ID of the migration it follows).

```
migration_A (down_revision=None)  <-- initial
    |
migration_B (down_revision=A)
    |
migration_C (down_revision=B)     <-- HEAD
```

## How Migration Conflicts Happen

When two developers independently create migrations from the same HEAD, their migrations share the same `down_revision`. After both PRs merge, the chain forks:

```
migration_C (down_revision=B)     <-- HEAD (main)
    |
    +-- migration_D (down_revision=C)   <-- PR #1
    |
    +-- migration_E (down_revision=C)   <-- PR #2
```

After both merge, Alembic sees **two heads** (D and E) and refuses to run `upgrade head`:

```
ERROR: Multiple head revisions are present; please specify a specific target.
```

This is called the **"multiple heads"** problem.

## Automated Protection with pre-commit

Prek is used to run a pre-commit hook that checks whenever you commit changes to migration files (`src/alembic/versions/*.py`). This catches conflicts before you push.

The hook is defined in `.pre-commit-config.yaml` under the `check-alembic-heads` ID.
We also run this check in the backend Github Action workflow.

### Running the Check Locally

You can run the migration head check at any time:

```bash
cd application/backend
bash scripts/check_alembic_heads.sh
```

Or trigger it via pre-commit:

```bash
pre-commit run check-alembic-heads
```

## Creating a New Migration

```bash
# From application/backend/
uv run alembic -c src/alembic.ini revision --autogenerate -m "describe your change"
```

Before creating a migration:

1. **Pull the latest `main`** and rebase your branch
2. **Check for existing heads**: `uv run alembic -c src/alembic.ini heads`
3. Create your migration only when there is a single head

## Resolving a Conflict

If CI fails with "Multiple migration heads detected", follow these steps:

### Step 1: Pull latest main and rebase

```bash
git fetch origin main
git rebase origin/main
```

### Step 2: Identify the current heads

```bash
cd application/backend
uv run alembic -c src/alembic.ini heads
```

This will show two (or more) head revisions. One belongs to `main`, the other is yours.

### Step 3: Find the correct HEAD from main

```bash
uv run alembic -c src/alembic.ini history
```

Look at the history to identify which revision is the new HEAD on `main` (the one that was merged while your PR was open).

### Step 4: Update your migration's down_revision

Open your migration file in `src/alembic/versions/` and change `down_revision` to point to the new HEAD from main:

```python
# Before (conflict — same down_revision as another merged migration)
down_revision: str | None = "abc123"

# After (fixed — now chains after the newly merged migration)
down_revision: str | None = "def456"  # the new HEAD from main
```

### Step 5: Verify

```bash
uv run alembic -c src/alembic.ini heads
# Should show exactly 1 head

uv run alembic -c src/alembic.ini history
# Should show a clean linear chain
```

### Step 6: Commit and push

```bash
git add src/alembic/versions/your_migration.py
git commit -m "fix: update migration down_revision after rebase"
git push
```

## Best Practices

- **Rebase before creating migrations.** Always pull the latest `main` and rebase your branch before running `alembic revision`. This ensures your migration chains from the actual current HEAD.

- **Coordinate with your team.** If you know another PR with a migration is about to merge, wait for it to land first, then rebase and create yours.

- **Keep migrations small.** Smaller, focused migrations are easier to rebase and less likely to conflict.

- **Check heads after rebase.** After any rebase, run `uv run alembic -c src/alembic.ini heads` to verify you still have a single head.

- **Never manually edit revision IDs.** Only change `down_revision` when resolving a conflict. Never modify the `revision` field — it is auto-generated and referenced by other migrations.
