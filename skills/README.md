# Physical AI Studio agent skills

Studio owns these canonical skills. **`skills/using/` is published to the OEP catalog; `skills/contributing/` stays in this repository.** A using skill must work in a customer project with the installed packages, without a Studio source checkout. A contributing skill changes Studio or Runtime source. An external robot plugin belongs to `using/`; changes to Studio's curated manifest or UI belong to contributors.

| Audience     | Skill                                                                        | Workflow                                |
| ------------ | ---------------------------------------------------------------------------- | --------------------------------------- |
| Using        | [`pai-train-policy`](using/pai-train-policy/SKILL.md)                        | Train, validate, test, predict          |
| Using        | [`pai-work-with-datasets`](using/pai-work-with-datasets/SKILL.md)            | Use LeRobot datasets and datamodules    |
| Using        | [`pai-export-policy`](using/pai-export-policy/SKILL.md)                      | Export and check deployment parity      |
| Using        | [`pai-benchmark-policy`](using/pai-benchmark-policy/SKILL.md)                | Evaluate a policy in a gym              |
| Using        | [`pai-create-robot-plugin`](using/pai-create-robot-plugin/SKILL.md)          | Develop a third-party robot plugin      |
| Contributing | [`pai-add-policy`](contributing/pai-add-policy/SKILL.md)                     | Extend the library with a policy family |
| Contributing | [`pai-add-benchmark`](contributing/pai-add-benchmark/SKILL.md)               | Extend first-party benchmarks           |
| Contributing | [`pai-add-robot-form-field`](contributing/pai-add-robot-form-field/SKILL.md) | Change Studio UI and plugin SDK         |
| Contributing | [`prepare-release`](contributing/prepare-release/SKILL.md)                 | Prepare app and library releases        |

## Discovery and authoring

Canonical content is in `skills/<audience>/<name>/`. Both `.agents/skills/<name>` and `.claude/skills/<name>` are committed adapter symlinks. They expose **both audiences to agents working in this repository**, but the org catalog indexes only `using/`. Never publish an adapter as a second source. Windows requires symlink support (Developer Mode or `core.symlinks true`); the sync script falls back to local junctions where available. After a rename, remove any stale junction reported by validation before rerunning sync.

- Choose a concise, distinct action name (directory and frontmatter must match; `^[a-z0-9]+(-[a-z0-9]+)*$`, at most 64 characters). Keep names unique within the installation scope. Contributor skills are repo-local and can use short names such as `prepare-release`; coordinate any public using-skill rename with the OEP catalog.
- Describe _what_ and _when_, with distinct user triggers. Give a specific negative trigger where neighboring workflows overlap; keep `SKILL.md` focused and disclose detailed unique knowledge in bundled `references/`. Link to existing documentation at its canonical URL so an installed copy does not depend on the Studio checkout.
- Write steps with checkable completion criteria; test with fake devices and ask before network downloads, hardware I/O, or mutating a running system.
- Run at least three audience-appropriate prompts from [`using/EVALUATION.md`](using/EVALUATION.md) or [`contributing/EVALUATION.md`](contributing/EVALUATION.md). Catalog publication and quality gates are managed in `open-edge-platform/skills`; future RFC family tags describe a **different axis** from these audiences.

```bash
# from the Studio repo root after adding, moving, or renaming a skill
python3 .github/scripts/skills/agent_skills.py sync
python3 .github/scripts/skills/agent_skills.py validate
```

CI validates the committed adapters; it does not regenerate them. Register a using skill under its canonical `skills/using` path in the catalog config after its source changes land. Remove stale installed names and add the new name when migrating an existing install.
