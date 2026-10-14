# Physical AI LeRobot Policy Plugin

This plugin connects LeRobot's native trainer and policy format to Physical AI. It provides a `LeRobotEngine` under the `physicalai.engines` entry-point group and a Runtime adapter under `physicalai.inference.adapters`. The adapter owns LeRobot preprocessing, stateful action selection, postprocessing, and episode reset; robot installations do not need `physicalai-train`.

## Install

For the library development environment, install the CPU variant and LeRobot engine extra from `library/`:

```bash
uv sync --extra cpu --extra lerobot
```

Choose `--extra cu128` or `--extra xpu` for the matching host PyTorch build. The engine itself runs training in a separate, locked worker environment, with its own CPU, CUDA (`cu128`), and XPU torch variants.

## Native CLI

`pai train lerobot` passes native arguments to LeRobot's `lerobot-train` command. The worker's standard output, standard error, and exit code are passed through unchanged:

```bash
pai train lerobot --policy.type=act --dataset.repo_id=lerobot/pusht --steps=20000
pai train --accelerator xpu lerobot --policy.type=act --dataset.repo_id=lerobot/pusht
pai train --interpreter /opt/lerobot/bin/python lerobot --help
```

Prepare the managed environment before a job (or while building an image), inspect its lock and size, and remove stale environments with:

```bash
pai engine prepare lerobot --accelerator xpu
pai engine list
pai engine prune
```

The default interpreter is managed by `physicalai-train` and provisioned with the pinned `uv` package. `--interpreter` skips provisioning and uses the supplied Python. Python config objects passed to `LeRobotEngine.train()` must match the worker's exact LeRobot version; native CLI flags and config files only need to satisfy the plugin's supported version range.

## Python API

```python
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from physicalai.engine import Engine
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

config = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id="lerobot/pusht"),
    policy=ACTConfig(push_to_hub=False),
    steps=20_000,
)

engine = LeRobotEngine(accelerator="xpu")
run = engine.train(config)
package = engine.export(run, "exports/act")
model = Engine.load(package)
```

`Engine.get("lerobot")` returns the class, and `Engine.create("lerobot", accelerator="xpu")` creates an engine instance. `LeRobotEngine.run()` accepts native settings and overrides `dataset.repo_id`, `dataset.root`, `output_dir`, and `policy.device` with its standard inputs.

LeRobot's worker project and lock ship as plugin package data. The core runner synchronizes that project into a cache keyed by framework, accelerator, and lock hash, checks the worker's version handshake, forwards JSON-lines progress, and propagates cancellation without redirecting native CLI output. The managed progress/cancellation runner currently requires POSIX inherited file descriptors; native CLI passthrough remains platform-native.

## Runtime Packages

Exporting a native LeRobot checkpoint creates a Runtime package containing `manifest.json` and the unchanged `pretrained_model/` files under a `.lerobot` artifact directory. The plugin Runtime adapter loads LeRobot's saved pre/postprocessor pipelines, calls `select_action()`, and forwards resets to the policy and processors. It does not delegate LeRobot preprocessing or postprocessing to Studio or Runtime components.

The Engine omits Groot from its policy list, and the Runtime adapter rejects Groot artifacts before loading a policy class because LeRobot enables `trust_remote_code=True` for Groot. That guard stays until the remote-code path is reviewed and pinned. Optimized export enablements and downstream patch materialization are not included; upstream support comes first, and patch application remains SDL-gated.

## Development Checks

From the repository root, run the focused contract tests with the library dependencies installed:

```bash
uv run --project library pytest library/tests/unit/engine
uv run --project plugins/policies/lerobot --extra train --extra tests pytest plugins/policies/lerobot/tests
```

Optional public-dataset training checks are gated because they download data and start the isolated worker:

```bash
PHYSICALAI_RUN_LEROBOT_INTEGRATION=1 \
HF_LEROBOT_HOME=/path/to/lerobot-cache \
uv run --project library --extra cpu --extra tests --extra lerobot \
    pytest library/tests/integration/test_engine_lerobot_hf_training.py -s
```
