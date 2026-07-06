# Library agent skills

Skills for `library/` (`physicalai-train`): policies, datasets, training, benchmarking, and export. Library skills must cover both direct Python API usage and the `physicalai` CLI whenever both surfaces exist.

Run commands from `library/` unless noted otherwise (`uv sync`, `uv run pytest ...`, `physicalai ...` with configs under `library/configs/`).

## Skills

| Skill                              | Covers                                                                                                           |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `library-adding-a-policy`          | Create/modify a policy family (config/model/policy split), register it, keep it train/export compatible.         |
| `library-training-a-policy`        | `physicalai.train.Trainer`, `physicalai fit/validate/test/predict`, YAML configs, debugging runs.                |
| `library-benchmarking-a-policy`    | `Benchmark(...).evaluate(...)`, `physicalai benchmark`, gym rollouts, `results.json`/`.csv`, adding a Benchmark. |
| `library-working-with-datasets`    | `LeRobotDataModule`, direct dataloader inspection, `repo_id`, format conversion, observation Features.           |
| `library-exporting-and-validating` | `policy.export(...)`, `physicalai export`, ONNX/OpenVINO/Torch/ExecuTorch, parity, the export/load contract.     |

New library skills must pass at least three scenarios in [`EVALUATION.md`](EVALUATION.md).

## Add a library skill

```bash
NAME=library-my-workflow
mkdir -p "skills/library/$NAME"
$EDITOR "skills/library/$NAME/SKILL.md"
python3 .github/scripts/skills/agent_skills.py sync
```

Global authoring rules: [`../README.md`](../README.md).
