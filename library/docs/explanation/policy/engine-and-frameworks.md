# Native Framework Engines and Policy Plugins

Robotics policies come from different frameworks, each with its own configuration, training loop, checkpoint format and inference processing. Reimplementing those recipes in one training loop can change their behavior. Engines let a framework or model repository keep its own training, while Physical AI provides one place to discover engines, launch native training in isolated environments, and load the results through Runtime.

A new model is enabled in the earliest of these stages that meets its quality and deployment needs:

1. **Framework plugin**: expose the policies a training framework already supports, such as LeRobot.
2. **Repository plugin**: support one model from its own repository, early.
3. **Enablement**: add a capability, such as accelerator training or an optimized export, preferably upstream.
4. **First-party policy**: implement the policy in Studio when an external integration is too fragile.

## Architecture

```text
Python API / pai CLI / Studio
            │
            ▼
Engine registry (`physicalai.engines`)
  ├── Trainer: first-party Lightning training, in-process
  ├── LeRobotEngine: LeRobot's own config, CLI and trainer, in a locked worker environment
  └── other framework or repository engines, added as plugins
            │
locked worker environments (`physicalai.isolate`)
            ▼
native checkpoint → Runtime package → InferenceModel ← plugin RuntimeAdapter
```

Engines and plugins do not import Studio. Studio owns presentation, projects, training targets and jobs; Runtime loads packages through the plugins' registered inference adapters.

## Engine API

`Engine` packages models for Runtime and reports capabilities; `TrainingEngine` adds native training. An engine that only packages or exports models subclasses `Engine`, and `pai train` rejects it.

```python
class Engine(ABC):
    framework: ClassVar[str]
    env: ClassVar[WorkerEnv | None] = None                       # the engine's own environment
    policy_envs: ClassVar[Mapping[str, WorkerEnv]] = {}           # environments for specific policies
    enablements: ClassVar[tuple[Enablement, ...]] = ()

    def export(self, source, output_path, backend=None) -> Path: ...   # abstract
    def policies(self) -> list[str]: ...                               # abstract
    def capabilities(self, policy: str) -> Capabilities: ...


class TrainingEngine(Engine):
    def cli(self, argv: Sequence[str], *, interpreter: Path | None = None) -> int: ...   # abstract
    def run(self, settings, *, dataset, output_dir, device, hooks=None, interpreter=None) -> Run: ...
    def config_schema(self, policy: str) -> dict: ...                                   # abstract
```

`Engine.get(name)` returns an engine class, `Engine.create(name, **options)` an instance, and `Engine.load(package)` a Runtime model. Both are typed as the base `Engine` for plugin engines; import a plugin's class to use its own methods with type checking. `Run` describes a native run; `TrainHooks` receives progress and can request cancellation.

### First party: Trainer

`physicalai.train.Trainer` is the first-party engine. It inherits Lightning's `fit`, `validate`, `test` and `predict` unchanged, and can also be selected by name:

```python
from physicalai.engine import Engine

trainer = Engine.create("physicalai", max_epochs=10, precision="bf16-mixed", accelerator="gpu")
trainer.fit(policy, datamodule=data)
Trainer.export(policy, "exports/act", backend="openvino")  # the backend is required
```

### LeRobot

`LeRobotEngine` keeps LeRobot's own configuration and trainer. It serializes a `TrainPipelineConfig` and runs LeRobot in its locked worker environment; the host and worker LeRobot versions must match.

```python
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from physicalai.engine import Engine
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

engine = LeRobotEngine()
config = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id="lerobot/pusht"),
    policy=ACTConfig(device="cuda", push_to_hub=False),
    steps=20_000,
)
run = engine.train(config)
package = engine.export(run, "exports/act", backend="native")
model = Engine.load(package)
```

Other engines register through the `physicalai.engines` entry-point group and define their own constructor and methods. A plugin's base install registers its Runtime adapter; `physicalai-train` is needed only for training.

## CLI

```bash
pai train lerobot --policy.type=act --dataset.repo_id=lerobot/pusht --steps=20000
pai train lerobot --help
pai train --interpreter /opt/lerobot/bin/python lerobot --config_path=./train_config.json
```

Everything after the framework name goes to the framework's own CLI unchanged, with its streams and exit status. `--interpreter` comes before the framework name and runs a user-managed environment as is. Device selection stays native: LeRobot's `--policy.device` picks the matching worker environment, and the engine detects the hardware when it is not set.

## Worker environments

Conflicting framework and Runtime dependencies are never installed into one interpreter. `physicalai.isolate` prepares, caches and launches locked environments; the engine adds accelerator selection, version checks and the training protocol.

| `WorkerEnv` field  | Purpose                                                                                                                                            |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project`          | `BundledProject` (a project and lock shipped in the plugin) or `GitProject` (a framework's own project at a commit, optionally with patch series). |
| `name`             | The engine's own environment (`default`), a policy family's, or an enablement's.                                                                   |
| `python`, `extras` | uv `--python` version and the uv extras for each accelerator profile (`cpu`, `cuda`, `xpu`), for example `{"cuda": ("cu132",)}`.                   |
| `worker`           | Worker script; `run_worker(..., worker=...)` selects another, such as an export worker.                                                            |

Environments are cached by source and sync profile under `~/.cache/physicalai/isolate/<key>` (`PHYSICALAI_ISOLATE_CACHE` overrides it), a cache Studio shares with Runtime. The engine records its framework, environment name and accelerator on each environment it prepares, so `engine list` and `engine prune` see only those. A bundled project's lock must be current (`uv sync --locked`); an upstream Git project's lock is installed as is (`uv sync --frozen`). The worker reports its framework version, which must match the lock and the plugin's declared range; the host then acknowledges, and only then does training start.

The LeRobot plugin ships one worker project, `env/`, built for CPU, CUDA 13.2 (`cu132`) and XPU, matching Studio's host build. Policy families with their own LeRobot extras (`smolvla`, the `pi` family, `molmoact2`) are mutually exclusive extras of that project: each trains in its own environment, every other policy uses the project without one, and all share one lock.

```bash
pai engine prepare lerobot --accelerator cuda   # pre-provision an environment
pai engine prepare lerobot --policy smolvla     # a policy family's environment
pai engine list                                 # show the engine's cached environments
pai engine prune                                # remove those no installed engine declares
```

## Runtime packages

The LeRobot engine packages the native `pretrained_model/` folder with a Runtime manifest. The plugin's Runtime adapter runs LeRobot's own preprocessor, `select_action()`, postprocessor and reset, one step at a time; executable code stays in the installed plugin, not the package.

The plugin supports only policy types whose training, packaging and Runtime inference have been checked: ACT, Diffusion, SmolVLA and Pi0.5. Training, the CLI, export and the Runtime adapter reject other types before any LeRobot code runs; several LeRobot policies load remote code without a pinned revision. Policies that need a different inference loop than `select_action()` are out of scope until checked. Before LeRobot loads a checkpoint's saved processors, each step must be a registered LeRobot step or a class under `lerobot.`, and each state file a `.safetensors` file in the checkpoint.

## Enablements

An engine declares capabilities it has validated for each policy. An enablement that needs different code or dependencies brings its own environment:

```python
Enablement(
    name="openvino",
    provides={"acme_policy": {"export": ("openvino",)}},
    env=WorkerEnv(..., name="openvino", project=GitProject(upstream, commit, patches=series)),
)
```

A capability is offered only after a test passes: for training, the native run uses the expected device and recipe; for export, the package loads through the plugin's Runtime adapter and matches the native model's actions.

Changes are contributed upstream first, and a plugin then pins the upstream release. Until then, an enablement can apply a patch series to a pinned upstream commit:

| Series             | Patches live                                   | Pinned by                                           |
| ------------------ | ---------------------------------------------- | --------------------------------------------------- |
| `PublishedPatches` | In another repository at an immutable revision | Per-file SHA-256 digests and the resulting Git tree |
| `BundledPatches`   | In the plugin, reviewed in this repository     | Digests computed from the files; optional tree pin  |

Runtime applies patches with `git am` to a clean cached checkout only, never to a user's checkout or a model package. Moving the same patches between repositories keeps the environment's identity. Applying another team's patches requires SDL and license review.
