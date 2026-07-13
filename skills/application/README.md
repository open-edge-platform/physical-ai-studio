# Studio (application) agent skills

Skills for the GUI and orchestration stack under `application/`:

- `application/backend/` — FastAPI, job orchestration, OpenAPI
- `application/ui/` — React, generated API types
- `application/docker/` — compose, device setup

Name skills with the `studio-` prefix (see [`../README.md`](../README.md)) and add an [`EVALUATION.md`](EVALUATION.md) with at least three scenarios per skill.

## Add a studio skill

```bash
NAME=studio-my-workflow
mkdir -p "skills/application/$NAME"
$EDITOR "skills/application/$NAME/SKILL.md"
python3 .github/scripts/skills/agent_skills.py sync
```

Global authoring rules: [`../README.md`](../README.md).
