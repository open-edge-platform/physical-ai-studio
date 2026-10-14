# Policy Plugins: Engine and Native Frameworks

> Status: Proposal, revised 2026-10-14. The proof of concept in the `feat/engine-poc` worktree validated the core idea: run LeRobot's own trainer and package its checkpoint for Runtime (see [POC Validation](#poc-validation)). This revision replaces the POC's routing `Engine`, its `Framework` protocol, and its `uv run --with` environments with one engine class per framework, a native CLI passthrough, Studio-owned display data, and locked worker environments. The revision is not implemented yet. It also proposes how models are enabled step by step, including XPU training and optimized export (OpenVINO, ONNX, ExecuTorch) through enablements (see [Model Enablement](#model-enablement)); whether another team's patch series may be applied at install time still needs SDL review.

Paths are relative to the Studio repository root unless marked Runtime, which refers to [`openvinotoolkit/physicalai`](https://github.com/openvinotoolkit/physicalai). LeRobot behavior and dependency claims below are checked against the upstream [`v0.6.1` tag](https://github.com/huggingface/lerobot/tree/v0.6.1), the latest stable release published on [PyPI](https://pypi.org/project/lerobot/0.6.1/) as of 2026-10-12. Studio's library and plugin declare `lerobot>=0.6.1,<0.7`; `library/uv.lock` pins 0.6.1 for reproducible CI.

## Summary

Physical AI should train, package, and deploy policies from other frameworks (LeRobot first, then others such as OpenPI or Xiaomi's) through its Python API, its CLI, and the Studio GUI, without porting them first. Someone who knows LeRobot should be able to use LeRobot exactly as they already do.

- **The framework trains itself.** Each framework runs its own training entry point (for LeRobot, `lerobot-train`) instead of placing its model inside a Lightning loop.
- **Files are the contract.** The input is a LeRobot-format dataset, which Studio already records. The output is a Runtime policy package: `manifest.json` plus the framework's native model folder.
- **One `Engine` class at the top, one engine per framework.** Each engine has the signatures that suit its framework. `physicalai.train.Trainer` is the first-party engine and stays exactly the Lightning `Trainer`; `LeRobotEngine.train()` takes LeRobot's own `TrainPipelineConfig`. The base class declares only what every framework shares.
- **The CLI passes native arguments through.** `pai train lerobot …` behaves like `lerobot-train …`: same flags, same output, same exit code.
- **Plugins expose facts; Studio owns display.** Engines report which policies they can train, their native config schema, and how to run a job. Studio decides what to show and how. Plugins never import Studio code.
- **Each framework trains in its own locked environment.** A plugin ships a uv project and lock for its worker; `physicalai-train` provisions it into a cache keyed by the lock.
- **Models are enabled step by step.** A framework plugin enables all of a framework's policies, and a repo plugin enables one model from its repository on day 0 or 1. Enablements add XPU training or optimized export: upstream first, otherwise our own overlay code or another team's patch series by reference. A first-party port is the last step. The plugin's own Runtime adapter runs every package it produces, including the framework's pre- and post-processing.
- **It lives in the library.** The `Engine` base class, the registry, and the environment mechanism ship in `physicalai-train`. Plugins are ordinary pip packages discovered through entry points.

## Why Not Wrap the Training Loop

The current `LeRobotPolicy` puts LeRobot's model inside a Lightning module. That makes the model match LeRobot, but the result of training also depends on the training script: the learning-rate schedule, gradient clipping, the frames each sample contains, normalization statistics, precision, and checkpointing. Comparing `src/lerobot/scripts/lerobot_train.py` at `v0.6.1` with Studio's path (`library/src/physicalai/policies/lerobot/policy.py`, `library/src/physicalai/train/utils.py`, and `application/backend/src/training/job.py`) shows the differences below. The POC validates an ACT native/Engine CPU parity slice and one real isolated Pi05 training/package step; it does not reproduce every row for every policy.

| `lerobot-train` (`v0.6.1` defaults for the policy families compared here)                                                        | Studio today                                                                             | Effect                                                                                                                                                              |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Policy presets supply warmup/cosine schedules for Diffusion, SmolVLA, Pi0/Pi05, and GR00T N1.7; ACT has no scheduler preset      | Not built; constant LR                                                                   | Different optimization, especially for VLA fine-tuning                                                                                                              |
| Gradient clipping from the optimizer preset (10.0 for ACT, Diffusion, and SmolVLA; 1.0 for Pi0/Pi05 and GR00T N1.7)              | None in production                                                                       | Updates are never clipped                                                                                                                                           |
| Action and observation windows from the config's `*_delta_indices`                                                               | `_get_delta_indices()` approximates them as `range(n_action_steps)`                      | Diffusion gets 32 actions from t instead of 64 from t-1 and fails `assert horizon == self.config.horizon`; ACT and others break when `n_action_steps != chunk_size` |
| `EpisodeAwareSampler` with `drop_n_last_frames` (7 for Diffusion)                                                                | Plain shuffle                                                                            | More samples padded past the end of the episode                                                                                                                     |
| Camera mean and std replaced with ImageNet statistics (`dataset.use_imagenet_stats=True`)                                        | Dataset statistics                                                                       | Different image normalization for ACT and Diffusion                                                                                                                 |
| Accelerator precision follows policy `dtype` (fp32 by default here); GR00T N1.7 uses internal bf16 autocast with fp32 parameters | GUI default `bf16-mixed`                                                                 | Different numerics                                                                                                                                                  |
| Fine-tuning refits normalizer statistics and output features to the new dataset; `rename_map` maps camera keys                   | Pretrained processors and config are kept as they are                                    | Wrong normalization when fine-tuning; XR0 works around the same issue with `_rebuild_preprocessors()`                                                               |
| Held-out loss through `policy.forward()` (off by default; trains on all episodes)                                                | `validation_step()` only accepts gym environments, and the GUI holds out 10% of episodes | The first validation calls `env.reset()` on an `Observation` and fails                                                                                              |
| `policy.update()` after each optimizer step                                                                                      | Never called                                                                             | Only TDMPC's EMA target network depends on it                                                                                                                       |
| 100k steps; keeps the last checkpoint                                                                                            | 5 epochs; keeps the best `val/loss` checkpoint                                           | Different budget and checkpoint choice                                                                                                                              |
| `--peft`, per-sample loss weighting, language-annotated datasets                                                                 | Not exposed                                                                              | Optional features are missing                                                                                                                                       |

These are LeRobot trainer defaults, not a support matrix. Validation status is recorded in [POC Validation](#poc-validation).

Inference has similar gaps when a wrapped model runs through Runtime. `TorchAdapter.predict()` calls the policy's eval `forward()`, which returns one action from LeRobot's internal queue. `InferenceModel.reset()` never reaches the policy, so queues and observation history leak across episodes. LeRobot's real-time chunking arguments (`inference_delay` and `prev_chunk_left_over`) cannot be passed through.

`library/tests/integration/test_lerobot_wrapper_equivalence.py` proves that the wrapper reproduces LeRobot's loss, gradients, and optimizer step. It cannot catch the differences above. The native side replays the batches Studio produced, neither side uses a scheduler, gradient clipping is passed to the test trainer by hand, and the runs last a few steps (in fp32 for the non-VLA policies). The dataset is also built through `get_delta_timestamps_from_policy()`, with its default `fps=10` against a 50 fps dataset, instead of the production path.

Lightning is not the limitation. It can express all of this, and XR0 and RLDX-1 already configure step-based schedulers and gradient clipping. But each framework's recipe would have to be reimplemented and proven equivalent, and each framework release could change it again. Running the framework's own trainer makes training equivalent by construction.

## Design

### Layers

```text
GUI      Studio backend + UI    owns display: which policies, labels, forms
            │
CLI      pai train <fw> …       passes the framework's own arguments through
            │
            ▼
API      physicalai.engine      Engine ◄─ Trainer (physicalai) · LeRobotEngine · OpenPIEngine · …
            │
            ▼
Runtime  InferenceModel         ◄─ adapters (physicalai.inference.adapters)
```

Dependencies point down only. An engine never imports Studio code. The CLI and Studio call engines only through the shared methods of the base class; Python users also get each engine's own methods.

Each piece builds on an extension point that already exists:

| Piece     | Contract                                                                       | Builds on                                                                   |
| --------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Input     | LeRobot-format dataset on disk                                                 | Studio dataset snapshots, which training already receives as `dataset_root` |
| Execution | `Engine.run()` and `Engine.cli()`, in the engine's worker environment          | Studio job control; Runtime's CLI host                                      |
| Output    | `manifest.json` plus the native model folder                                   | Runtime's `policy_package` manifest and Studio's model import               |
| Loading   | A Runtime adapter per framework                                                | Runtime's `physicalai.inference.adapters` entry-point group                 |
| Discovery | `physicalai.engines` entry point pointing at an `Engine` subclass              | Runtime CLI subcommands and robot plugins use the same mechanism            |
| Display   | Studio's policy catalog and form overlays, applied to `Engine.config_schema()` | The policy catalog from #1294 and the schema-driven robot forms             |

### The `Engine` Base Class

```python
class Engine(ABC):
    """Top of the hierarchy. Declares only what every framework shares."""

    framework: ClassVar[str]          # "physicalai", "lerobot", …
    env: ClassVar[WorkerEnv | None]   # None for engines that run in-process (first party)
    enablements: ClassVar[tuple[Enablement, ...]] = ()  # see Model Enablement

    # Not declared here: __init__ and the training methods (fit, train, validate, …).
    # Each engine defines the signatures that suit its framework.

    @classmethod
    @abstractmethod
    def cli(cls, argv: Sequence[str], *, interpreter: str | Path | None = None,
            accelerator: str = "auto") -> int:
        """Run the framework's own CLI unchanged and return its exit code."""

    @classmethod
    @abstractmethod
    def run(cls, settings: Mapping[str, Any], *, dataset: Path, output_dir: Path, device: str,
            hooks: TrainHooks | None = None, interpreter: str | Path | None = None) -> Run:
        """Train headless from native settings on standard inputs, with progress and cancellation."""

    @classmethod
    @abstractmethod
    def export(cls, source: Run | str | Path, output_path: str | Path, backend: str | None = None) -> Path:
        """Write a Runtime package. Engines may add optional keyword arguments."""

    @classmethod
    @abstractmethod
    def policies(cls) -> list[str]:
        """Policy types this engine can train, taken from the framework itself."""

    @classmethod
    @abstractmethod
    def config_schema(cls, policy: str) -> dict[str, Any]:
        """JSON Schema of the native training config for one policy type."""

    @classmethod
    def capabilities(cls, policy: str) -> Capabilities:
        """Validated training accelerators and export formats, from the stock environment and enablements."""

    # Registry over the physicalai.engines entry points
    @classmethod
    def create(cls, name: str, /, **options: Any) -> Engine: ...
    @classmethod
    def get(cls, name: str) -> type[Engine]: ...
    @classmethod
    def names(cls) -> list[str]: ...

    @staticmethod
    def load(package: str | Path, **kwargs: Any) -> InferenceModel:
        """Runtime's InferenceModel; identical for every framework."""


class TrainHooks(Protocol):
    def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None: ...
    def should_stop(self) -> bool: ...


@dataclass(frozen=True)
class Run:
    framework: str
    output_dir: Path | None
    checkpoint: Path | None  # native model folder, for example checkpoints/last/pretrained_model
```

The rules that keep the contract clean:

- Nothing on the base class is about display. A quick test: if a script, a CI job, or another orchestrator could use it, it belongs on `Engine`; if it mentions labels, forms, projects, or training targets, it belongs in Studio.
- Progress and cancellation hooks are not GUI logic; scripts, notebooks, and CI use them too.
- Engine modules import only light dependencies at module level; framework imports happen inside methods or in the worker.
- `physicalai.engine` cannot live under `physicalai.train` or `physicalai.policies`, because their `__init__` modules import Lightning, torch, and LeRobot. The registry loads engine classes lazily through entry points.

### First Party: `Trainer`

The existing `physicalai.train.Trainer` becomes the first-party engine. It subclasses both `Engine` and the Lightning `Trainer`, so its constructor and its `fit`, `validate`, `test`, and `predict` are Lightning's, unchanged.

```python
class Trainer(Engine, lightning.Trainer):
    framework = "physicalai"
    env = None  # runs in-process

    # __init__, fit, validate, test, predict: Lightning's, unchanged

    @classmethod
    def cli(cls, argv, **_) -> int: ...  # same as `pai fit …`

    @classmethod
    def export(cls, source: Policy | Run | str | Path, output_path, backend="openvino", *,
               policy_class=None, input_sample=None, post_export_hooks=None, **export_kwargs) -> Path:
        policy = source if isinstance(source, Policy) else load_policy(policy_class, source)
        policy.export(output_path, backend, input_sample=input_sample,
                      post_export_hooks=post_export_hooks, **export_kwargs)
        return Path(output_path)

    @classmethod
    def policies(cls) -> list[str]:
        return list_physicalai_policies()
```

- Parity with today's API comes for free because nothing is copied. `isinstance(trainer, lightning.Trainer)` and `Tuner(trainer)` keep working; this was checked with Lightning and the Studio `Trainer`.
- `export()` keeps every option of `policy.export()`. The base class declares only the generic minimum that the CLI and Studio pass; adding optional keyword arguments in a subclass type-checks with `mypy --strict` and basedpyright.
- `PhysicalAIEngine = Trainer` can be offered as an alias for symmetry in documentation.
- Studio keeps its current first-party form and `run_training_job()` at first. Moving them onto `run()` and `config_schema()` is a follow-up.

### LeRobot: `LeRobotEngine`

```python
# physicalai_lerobot_policy_plugin/engine.py
class LeRobotEngine(Engine):
    framework = "lerobot"
    env = WorkerEnv(
        project=files(__package__) / "env",  # pyproject.toml + uv.lock
        worker=files(__package__) / "_worker.py",
        distribution="physicalai-lerobot-policy-plugin",
        package="lerobot",
    )

    def __init__(self, *, interpreter: str | Path | None = None, accelerator: str = "auto") -> None: ...

    def train(self, config: TrainPipelineConfig, hooks: TrainHooks | None = None) -> Run:
        """LeRobot's trainer on a native config object, sent to the worker as train_config.json."""

    @classmethod
    def cli(cls, argv, *, interpreter=None, accelerator="auto") -> int:
        python = interpreter or prepare(cls.env, accelerator)
        return subprocess.call([scripts_dir(python) / "lerobot-train", *argv])  # terminal attached

    @classmethod
    def run(cls, settings, *, dataset, output_dir, device, hooks=None, interpreter=None) -> Run:
        flags = to_flags(settings) + ["--dataset.repo_id=local", f"--dataset.root={dataset}",
                                      f"--output_dir={output_dir}", f"--policy.device={device}"]
        return run_worker(cls.env, flags, hooks, device=device, interpreter=interpreter)

    @classmethod
    def export(cls, source, output_path, backend="native") -> Path:
        return package_lerobot_checkpoint(source, output_path)  # manifest.json + pretrained_model/

    @classmethod
    def policies(cls) -> list[str]: ...  # supported subset of PreTrainedConfig.get_known_choices()

    @classmethod
    def config_schema(cls, policy) -> dict: ...  # schemas/<policy>.json, generated when the wheel is built
```

- The standard inputs of `run()` (`dataset`, `output_dir`, `device`) override the same native fields in `settings`.
- `export()` accepts a `Run` or a local native checkpoint folder. Hub repository IDs must be downloaded first.
- Schemas come from LeRobot's own config classes. Pydantic's `TypeAdapter(...).json_schema()` works on the LeRobot 0.6.1 dataclasses: 37 fields for `ACTConfig`, 56 for `PI05Config`, and 118 across 15 nested types for `TrainPipelineConfig`. The generated fields have no descriptions, and `TrainPipelineConfig.policy` is typed as the abstract base, so schemas are generated per policy type. Generation needs LeRobot importable, which the backend cannot assume, so it runs when the plugin wheel is built.

### Third-Party Engines

Third-party engines follow the same shape and add whatever methods their framework needs. The OpenPI names below are placeholders:

```python
class OpenPIEngine(Engine):
    framework = "openpi"
    env = WorkerEnv(...)

    def __init__(self, *, interpreter=None, accelerator="auto"): ...
    def train(self, config: openpi.TrainConfig) -> Run: ...
    def compute_norm_stats(self, config) -> Path: ...  # framework-specific methods are fine
    # plus cli, run, export, policies, config_schema
```

### Selecting an Engine

```python
engine = LeRobotEngine(accelerator="xpu")            # fixed framework: best IDE support
engine = Engine.create("lerobot", accelerator="xpu")  # framework from a string: same object
EngineCls = Engine.get("lerobot")                     # the class, for cli/run/export/policies
```

`Engine.create()` declares overloads for the engines core depends on: first party, and LeRobot through the `[lerobot]` extra. Checked with `mypy --strict`, basedpyright, and at runtime:

| Call                                         | Type checkers see | Engine-specific methods checked | Constructor options checked |
| -------------------------------------------- | ----------------- | ------------------------------- | --------------------------- |
| `LeRobotEngine(...)`                         | `LeRobotEngine`   | Yes                             | Yes                         |
| `Engine.create("lerobot", ...)`              | `LeRobotEngine`   | Yes                             | No; at runtime only         |
| `Engine.create("openpi", ...)` (third party) | `Engine`          | No; they exist at runtime       | No; at runtime only         |
| `Engine.create(name)` with `name: str`       | `Engine`          | No                              | No                          |

A wrong option such as `Engine.create("lerobot", max_epochs=3)` falls through to the generic `str` overload, so engines must declare explicit keyword arguments that reject it at runtime instead of accepting `**kwargs`.

Calling `Engine("lerobot", ...)` as a constructor is not supported. Python calls `__init__` again with the same arguments after `__new__`, so every engine, including `Trainer`, would need an extra leading parameter for the name; the metaclass alternative loses all static typing in both type checkers.

### Plugin Packaging

```toml
[project]
name = "physicalai-lerobot-policy-plugin"
dependencies = ["physicalai~=0.3.0", "lerobot[dataset]>=0.6.1,<0.7"]  # loader side, for robots; also the supported LeRobot range

[project.optional-dependencies]
train = ["physicalai-train>=0.4"]  # engine side; LeRobot's training dependencies live in env/

[project.entry-points."physicalai.engines"]  # read by physicalai-train
lerobot = "physicalai_lerobot_policy_plugin.engine:LeRobotEngine"

[project.entry-points."physicalai.inference.adapters"]  # read by Runtime (existing group)
lerobot = "physicalai_lerobot_policy_plugin.inference:register"
```

```text
physicalai_lerobot_policy_plugin/
├── __init__.py     does not import engine.py; robots install the plugin without physicalai-train
├── engine.py       LeRobotEngine: API and CLI; no Studio imports
├── _worker.py      runs inside the worker environment; standard library and LeRobot only
├── env/            stock worker environment: pyproject.toml + uv.lock (cpu, cu128, and xpu extras)
├── enablements/    one directory per enablement: pyproject.toml + uv.lock + glue.py
├── schemas/        config_schema() output, generated when the wheel is built
├── inference.py    registers the Runtime adapter lazily; never imports physicalai-train
└── runtime.py      Runtime adapter
```

Training plugins need no Studio entry point. Robot plugins ship a `studio_catalog.py` because port selection, calibration, and identification cannot be derived from a driver; training configs can be derived from the framework's own config classes.

## CLI

```bash
pai train lerobot --policy.type=act --dataset.repo_id=lerobot/pusht --steps=20000  # same as lerobot-train …
pai train lerobot --help                                                           # lerobot-train's own help
pai train --interpreter /opt/lerobot/bin/python lerobot --config_path=…/train_config.json --resume=true
pai train --accelerator xpu lerobot --policy.type=act …

pai engine prepare lerobot --accelerator xpu  # provision ahead of time (Docker builds, the GUI's preparing step)
pai engine list                               # prepared environments: engine, accelerator, lock, size, path
pai engine prune                              # delete environments whose lock is no longer current

pai fit --config act.yaml                     # first party, unchanged, like validate, test, predict, export, benchmark
```

```python
# pai train [--interpreter PY] [--accelerator NAME] <framework> <native args…>
def run(parser, cfg) -> int:
    return Engine.get(cfg.framework).cli(cfg.args, interpreter=cfg.interpreter, accelerator=cfg.accelerator)
```

- **The framework is a positional argument.** Runtime's CLI host treats the first token without a leading dash as the subcommand (`_select_subcommand()` in Runtime `src/physicalai/cli/main.py`), so `pai --framework lerobot …` looks for a subcommand named `lerobot`. `pai train --framework lerobot --policy.type=act` fails with `unrecognized arguments: --policy.type=act` in the CLI's jsonargparse parser. With the positional form, everything after the framework name is forwarded unchanged, including a native `--help` or a native flag that happens to be called `--framework`.
- **Studio's options come first.** `--interpreter` and `--accelerator` precede the framework name and never reach the framework.
- **The terminal is attached.** `cli()` runs the framework's own command with inherited standard streams and returns its exit code; the progress protocol is not used in this mode. The POC did not meet this bar:

  | Command          | `lerobot-train`     | `pai train lerobot` (POC)                                           |
  | ---------------- | ------------------- | ------------------------------------------------------------------- |
  | `--help`         | 386 lines on stdout | 0 lines on stdout; the worker redirected LeRobot's stdout to stderr |
  | `--bogus_flag=1` | Usage error, exit 2 | `RuntimeError` traceback, exit 1                                    |

- `pai train physicalai …` is an alias for `pai fit …`.

## Worker Environments

### Why Separate Environments

LeRobot's `[pi]` extra requires `transformers>=5.4.0,<5.6.0`, while [Runtime requires `transformers>=5.15.1,<5.16.0`](https://github.com/openvinotoolkit/physicalai/blob/main/pyproject.toml). One environment cannot hold both. Each engine with an `env` therefore trains in its own process and environment, which never installs `physicalai` or `physicalai-train`. The pattern is common: MLflow's `--env-manager` restores a model's own environment instead of using the caller's, and Ray's `runtime_env` gives each job its own `pip` or `uv` environment, or a given `py_executable`.

### Declared by the Plugin

The plugin ships a small uv project as package data: `env/pyproject.toml` and `env/uv.lock`. It uses the same accelerator pattern as `library/pyproject.toml`: `cpu`, `cu128`, and `xpu` extras, declared as conflicting, each mapped to its own PyTorch index.

```toml
# physicalai_lerobot_policy_plugin/env/pyproject.toml
[project]
name = "lerobot-worker-env"
version = "0"
requires-python = ">=3.12,<3.15"
dependencies = ["lerobot[training,pi]>=0.6.1,<0.7", "diffusers>=0.38.0,<0.40.0"]

[project.optional-dependencies]
cpu = ["torch", "torchvision"]
cu128 = ["torch", "torchvision"]
xpu = ["torch", "torchvision"]

[tool.uv]
package = false
conflicts = [[{ extra = "cpu" }, { extra = "cu128" }, { extra = "xpu" }]]

# Indexes and [tool.uv.sources] for torch and torchvision as in library/pyproject.toml:
# pytorch-cpu, pytorch-cu128, and pytorch-xpu, each selected by its extra.
```

The lock has hashes and pins every transitive dependency, including torch.

### Provisioned by Core

```python
def prepare(env: WorkerEnv, accelerator: str, *, offline: bool = False) -> Path:
    root = engine_cache() / f"{env.name}-{accelerator}-{env.lock_hash[:12]}"  # immutable per lock
    copy_once(env.project, root / "project")  # atomic; outside any host project
    subprocess.run([uv_bin(), "sync", "--frozen", "--extra", accelerator,
                    "--project", root / "project", *(["--offline"] if offline else [])],
                   env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(root / "venv")}, check=True)
    return root / "venv" / "bin" / "python"  # near-instant when the environment is already synced
```

- **Accelerator.** `accelerator="auto"` picks `xpu`, `cu128`, or `cpu` from the host's torch; an explicit value overrides it.
- **Pinned uv.** `uv_bin()` comes from the `uv` package on PyPI (`find_uv_bin()`), constrained to the release line the Docker images use (0.10.x), instead of whatever `uv` is on `PATH`.
- **A visible step.** `prepare()` is a separate step: `pai engine prepare`, or a "preparing environment" phase in the GUI. `run()` and `cli()` call it implicitly and log the first-run download.
- **Images.** The Docker and remote-trainer images run `pai engine prepare` at build time, so jobs never download dependencies. The images set `UV_PYTHON_DOWNLOADS=0`, so the interpreter an environment requires must be present in the image.
- **Non-Python dependencies.** These stay host prerequisites. LeRobot's video decoding needs FFmpeg; the Studio image builds an LGPL FFmpeg and pins `av` to it, and worker locks must stay compatible with that build.
- **Concurrency.** Copying the project is atomic, and uv locks an environment while it syncs, so concurrent jobs can prepare the same environment.

### Running the Worker

One runner in core serves every engine:

- It starts `<python> <worker> <native args>` for `run()`; `cli()` bypasses it.
- The worker's first message reports the protocol version and the installed framework versions, which the host checks (see [One Version Source](#one-version-source)).
- Progress, results, errors, and cancellation travel as JSON lines on a dedicated pipe, so the framework's stdout and stderr stay plain logs and are forwarded to the job log. The POC used stdout for the protocol and redirected LeRobot's stdout to stderr.
- Cancellation is a control message on stdin, followed by SIGTERM after a grace period, as in the POC.
- Each job runs in its own worker process, so the POC's process-wide `sys.argv` scoping and `update_policy` instrumentation never affect the caller.

### Bring Your Own Interpreter

`interpreter=/path/to/python` (CLI: `--interpreter`) skips provisioning. That environment must already contain the framework and its training dependencies. The worker's first message reports the framework version, and the host fails fast if it is outside the supported range. This matters because the worker instruments LeRobot internals (`update_policy` and `TrainPipelineConfig.validate`).

### One Version Source

- **Supported range:** the plugin's own `pyproject.toml`, read at run time from installed metadata with `importlib.metadata.requires()`, the same way #1294 reads the extras of `physicalai-train`. For the LeRobot plugin, this is `lerobot>=0.6.1,<0.7`.
- **Exact version:** the worker lock.
- **Consistency:** a plugin test asserts that the locked version is inside the supported range, so the two cannot drift. The plugin never repeats the version in Python.
- **Config objects:** when `LeRobotEngine.train()` receives a `TrainPipelineConfig`, the host's LeRobot serializes it and the worker's LeRobot parses it, so the two versions must match. Otherwise the caller passes flags or a config file.

```python
@dataclass(frozen=True)
class WorkerEnv:
    project: Traversable  # pyproject.toml + uv.lock
    worker: Traversable   # worker script
    distribution: str     # "physicalai-lerobot-policy-plugin"
    package: str          # "lerobot"

    def supported(self) -> SpecifierSet:
        for raw in metadata.requires(self.distribution) or []:
            requirement = Requirement(raw)
            if requirement.name == self.package and requirement.marker is None:
                return requirement.specifier
        raise LookupError(self.package)

    def pinned(self) -> Version:
        return lock_version(self.project / "uv.lock", self.package)


def test_worker_lock_within_supported_range():
    assert LeRobotEngine.env.pinned() in LeRobotEngine.env.supported()
```

### What Changes From the POC

| POC                                                                                                  | Problem                                                                                                                                                                                                          | Replaced by                                     |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `uv run --isolated --no-project --with "lerobot[training,pi]==0.6.1"`                                | Only LeRobot is pinned; torch, transformers, and every other dependency are resolved when a job starts, without hashes, so environments differ across machines and over time                                     | A lock with hashes, synced with `--frozen`      |
| The same command                                                                                     | No accelerator selection: the worker ignores the library's `cpu`, `cu128`, and `xpu` extras and gets PyPI's CUDA build of torch. In uv 0.10.4 only `uv pip` has `--torch-backend`; `uv run` and `uv sync` do not | Accelerator extras in the worker project        |
| Run uv from a temporary directory and restore the caller's directory through an environment variable | Workaround: `uv run --no-project` still reads the host project's `[tool.uv.index]`                                                                                                                               | `uv sync --project <cached copy>`               |
| Environment created implicitly by the first `train()`                                                | A multi-gigabyte download with no separate step, no progress for the GUI, no pre-warming, and network access for every new resolution                                                                            | `prepare()` and `pai engine prepare`            |
| Requirement strings as Python constants; user interpreters unchecked                                 | The host and worker LeRobot versions can differ, and a user interpreter with an untested version fails in confusing ways                                                                                         | One version source and the version handshake    |
| `environment="auto"` or `"manual"` plus `interpreter`, validated against each other in two places    | Two parameters that must agree                                                                                                                                                                                   | One optional `interpreter`; unset means managed |
| Process, protocol, and cancellation code in the LeRobot plugin (about 540 lines in `framework.py`)   | Every new plugin would repeat it                                                                                                                                                                                 | One worker runner in core                       |

Verified with uv 0.10.4: from inside a host project whose `[tool.uv.index]` points at an unreachable index, `uv run --isolated --no-project --with …` failed to resolve, while `uv sync --frozen --project <worker project>` with `UV_PROJECT_ENVIRONMENT=<cache>/venv` installed correctly. A second, offline sync of the same environment took about 60 ms. These checks used small packages; LeRobot, torch, and XPU have not been provisioned this way yet.

This reverses the POC statement that the Engine owns no lock or cache. Core owns the mechanism (a `uv sync` into a cache keyed by the lock); plugins own the declaration (the lock).

## Model Enablement

Physical AI enables a model in steps and stops at the first step that is good enough:

| Step                | What                                                                     | How                                                                                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Framework plugin | Every policy of a framework, for example LeRobot                         | An engine with a stock worker environment; native export                                                                                                                                 |
| 2. Repo plugin      | One model from its own repository on day 0 or 1, for example Isaac-GR00T | An engine whose stock environment pins the repository by commit; native export                                                                                                           |
| 3. Enablement       | XPU training or optimized export, done by us or by another team          | Upstream release first; otherwise our own overlay code or another team's patch series by reference. Each enablement has its own locked environment and is validated before it is offered |
| 4. First-party port | When an enablement is too brittle or too slow                            | The model is implemented in Studio, as for Pi05 and XR0                                                                                                                                  |

Upstream comes first. Enablement changes should be contributed upstream and maintained there. A downstream enablement is a bridge until an upstream release includes the change; it then reduces to a version pin.

### Repo Plugins

A repo plugin is an engine for a single model's repository. Its stock environment pins the repository by commit, `cli()` passes arguments through to the repository's own training script, and `run()` maps the standard inputs onto that script's flags. Packages are native, and the plugin's Runtime adapter runs them with the repository's own inference code.

- Many research repositories are not installable packages. Their source tree is placed on `sys.path` in the worker environment, and their requirements are declared in the plugin's environment project.
- System dependencies that only conda provides, such as MuJoCo or EGL, stay host prerequisites.

### Enablements

An enablement is a named, pinned bundle that a plugin declares. It adds a capability (training on an accelerator, or export to a format) for specific policies, and it brings its own locked worker environment:

```python
class Gr00tEngine(Engine):  # a repo plugin for NVIDIA/Isaac-GR00T
    framework = "gr00t"
    env = WorkerEnv(project=files(__package__) / "env", ...)  # upstream as released

    enablements = (
        Enablement(
            name="intel_ov",
            provides={"n1.7": {"export": ("openvino",)}},  # what it adds, per policy
            upstream=GitSource("https://github.com/NVIDIA/Isaac-GR00T", rev="3df8b38…"),
            patches=PatchSeries(  # referenced, never copied into our repository
                repo="https://github.com/open-edge-platform/robotics-ai-suite",
                rev="1a7deb8…",
                files=["src/pipelines/gr00t-n1d7-ov/patches/0001-….patch", "…"],  # exact list, in order
                sha256={"0001-….patch": "…", "…": "…"},
            ),
            project=files(__package__) / "enablements/intel_ov",  # pyproject.toml + uv.lock + glue.py
            accepts_stock_checkpoints=True,  # proven by tests; see Compatibility Across Environments
        ),
    )
```

### How Changes Reach the Environment

| Order | Mechanism                                            | What we ship                | Third-party code on the user's machine                                  | Who maintains the change               |
| ----- | ---------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------- | -------------------------------------- |
| 1     | Upstream release                                     | A version pin               | Unmodified                                                              | Upstream                               |
| 2     | Overlay code: wrappers, export recipes, scoped hooks | Our own code                | Unmodified; patched in memory only while exporting                      | Us, or the enabling team               |
| 3     | Referenced patch series                              | URL, commit, and checksums  | Modified on the user's machine from the owning team's published patches | The owning team; we bump the reference |
| None  | A fork, or patch files copied into our repository    | Modified third-party source | Modified and owned by us                                                | Us; not used                           |

Overlay code follows what Studio already does for XR0: `library/src/physicalai/policies/xr0/export_openvino.py` swaps `forward` on individual transformers modules only while exporting, without forking transformers. Preferred techniques, most robust first: public switches (for example `attn_implementation`), wrapper modules that define the exported graph, subclasses, per-instance swaps inside a context manager, and class-level swaps inside a context manager. Swaps run only against the framework version they were tested with.

Referenced patch series let us adopt another team's work as published, without forking and without copying it. Whether that is acceptable is an SDL question; see [Remaining Decisions](#remaining-decisions). If it is not, enablements are limited to upstream releases and overlay code.

### Referenced Patch Series

Preparing an enablement materializes the patched source once per upstream revision and patch set:

```python
def prepare_enablement(enablement: Enablement, accelerator: str) -> Path:
    root = engine_cache() / f"{enablement.key}-{accelerator}"  # upstream revision + patch hashes + lock hash
    if not (root / "src").exists():
        with staging_dir(root) as tmp:  # built in a temporary directory, renamed at the end
            fetch(enablement.upstream, tmp / "src")  # pinned commit, never a branch
            for patch in fetch_verified(enablement.patches):  # every SHA-256 must match
                git("apply", "--check", patch, cwd=tmp / "src")
                git("apply", patch, cwd=tmp / "src")
            copy(enablement.project, tmp / "project")
    uv_sync(root / "project", extra=accelerator, venv=root / "venv")  # --frozen
    return root / "venv" / "bin" / "python"
```

```toml
# enablements/intel_ov/pyproject.toml; CI builds the patched tree the same way, then runs `uv lock`
[project]
name = "gr00t-intel-ov-env"
version = "0"
requires-python = ">=3.10,<3.13"
dependencies = ["gr00t", "openvino", "nncf"]

[tool.uv.sources]
gr00t = { path = "../src", editable = false } # the patched tree; its own dependency changes land in the lock
```

- **Supply chain.** Commits are pinned, every patch file is verified by SHA-256, and only allowlisted repositories are fetched.
- **Images.** Docker and remote-trainer images prepare enablements at build time, like other worker environments.
- **Provenance.** Each package records the framework, the enablement, the upstream revision, and the patch-series revision in `manifest.json`, which already allows extra keys.
- **Exact patch lists.** An enablement lists the patch files it applies. A series that mixes demo changes into shared files (for example a fixed camera mode in a training script) may need a subset, and the subset must still apply and pass the tests.

### Exporting Through an Enablement

The glue runs inside the enablement's environment, like the worker, and calls the owning team's converter unchanged:

```python
# enablements/intel_ov/glue.py (run by path; standard library and the framework only)
def export(checkpoint: Path, staging: Path, *, precision: str = "fp16") -> None:
    run([sys.executable, SRC / "scripts/deployment/export_ov_n1d7_single_ov.py",
         "--model-path", checkpoint, "--output-dir", staging / "ir",
         "--precision", precision, "--llm-lang-tokens", "64"])
    save_reference(checkpoint, staging / "reference.npz")  # PyTorch outputs on fixed inputs
```

The host side is the same for every enablement:

```python
@classmethod
def export(cls, source, output_path, backend="native", **options) -> Path:
    if backend == "native":
        return package_native(source, output_path)
    enablement = cls.enablement_for(policy_of(source), "export", backend)  # validated combinations only
    staging = run_worker(enablement, ["export", str(source)], **options)
    package = write_package(staging, output_path, provenance=enablement.provenance())
    check_parity(InferenceModel(package), staging / "reference.npz")  # through the adapter robots use
    return package
```

`capabilities()` lists a format for a policy only when an enablement provides it and its parity check passes. CI re-runs the export and the parity check whenever a lock or a patch-series reference changes.

### Choosing an Enablement

Users ask for a capability, and the engine picks the validated enablement that provides it. An illustrative matrix:

```text
lerobot / pi05    train   cuda, xpu → stock (LeRobot supports XPU natively)
                  export  openvino  → intel_pi05_ov (LeRobot 0b067df plus its patch series)
act / act         train   cuda → stock; xpu → intel_xpu (an IPEX patch series)
                  export  openvino  → intel_ov
gr00t / n1.7      export  openvino  → intel_ov
```

```bash
pai train act --accelerator xpu …               # trains in the intel_xpu enablement
pai engine prepare gr00t --enablement intel_ov  # explicit, for example when building an image
```

### Compatibility Across Environments

An enablement often pins a different upstream revision than the stock environment. The Pi05 OpenVINO work in Robotics AI Suite targets LeRobot `main` from January 2026, while the stock environment pins 0.6.1. A model trained in the stock environment can be exported or run through such an enablement only if the enablement declares `accepts_stock_checkpoints` and its tests prove that those checkpoints load and pass parity. Otherwise, training must also run in the enablement.

### Running Enabled Packages

Packages contain data only; the code that runs them comes from installed plugins registered under `physicalai.inference.adapters`. The plugin's Runtime adapter owns inference end to end, including the framework's own pre- and post-processing:

```python
class Gr00tOpenVINOAdapter(RuntimeAdapter):  # in the plugin, registered like the native adapter
    def load(self, package: Path) -> None:
        self.pre, self.post = framework_processors(package)  # the framework's own processing code
        self.net = ov.Core().compile_model(package / "ir" / "gr00t.xml", self.device)

    def predict(self, observation):
        return self.post(self.net(self.pre(observation)))
```

- **No porting.** Runtime runs no preprocessors or postprocessors of its own for these packages, so nothing has to be ported to Runtime components. The LeRobot native package already works this way: its manifest lists no Runtime processors, and the adapter calls LeRobot's own processors and `select_action()`.
- **Robot dependencies.** The robot needs the plugin and the framework's inference dependencies in addition to Runtime and OpenVINO. When those conflict with Runtime's own pins, the inference-isolation decision applies.
- **Patched processing.** If an enablement's patches change pre- or post-processing code, the robot needs that enablement's build of the framework. The package's provenance records it, and the parity check runs through the same adapter and build the robot will use.

### Evidence: Robotics AI Suite

[Robotics AI Suite](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines) publishes downstream enablements for several models. Each pipeline pins the upstream repository as a git submodule and applies a numbered `git am` patch series:

| Pipeline              | Upstream, pinned commit                                      | Added files                                            | Edits to upstream code                                                                         |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `gr00t-n1d7-ov`       | Isaac-GR00T `3df8b38` (2026-05-07)                           | About 10,800 lines of export and inference scripts     | About 11 library lines, plus their own inference server                                        |
| `pi05-rtc-ov`         | LeRobot `0b067df` (2026-01-20, unreleased `main`, not 0.6.1) | Conversion scripts, ALOHA examples, and tests          | `modeling_pi05.py` +177/−71 (OpenVINO inside `PI05Policy`), `pyproject.toml` +80, CI workflows |
| `diffusion-policy-ov` | Diffusion Policy                                             | Four conversion scripts                                | Five policy files, +34 to +146 lines each                                                      |
| `act-sample`          | `tonyzhaozh/act` `742c753` (2024-01-28)                      | `ov_convert.py`                                        | `imitate_episodes.py` +262/−57 (IPEX for XPU, a real robot, fixed cameras)                     |
| `rdt-ov`, `idp3-ov`   | RDT and iDP3                                                 | Converters and OpenVINO model wrappers                 | Small edits, +3 to +23 lines                                                                   |
| `vla-pi0.5-openvino`  | No patches; its scripts import LeRobot's stock `pi05` policy | ONNX conversion, NNCF int8 compression, and validation | None                                                                                           |

- **Pins lag.** Each series has its own upstream pin, often not a release, so each enablement needs its own environment.
- **Concerns are mixed.** A series combines conversion scripts (what an enablement's glue calls), OpenVINO inside the framework's own policy class, XPU and IPEX device edits, security fixes (`torch.load`, Coverity, Bandit), and demo, ROS 2, simulation, Docker, and CI changes.
- **Both styles exist.** The same organization also enables Pi0.5 without patches: `vla-pi0.5-openvino` wraps unmodified LeRobot, which is the overlay-code style.
- **Their main GR00T fix is already an overlay.** `patch_diffusers_attention_for_openvino()` swaps `diffusers.Attention.forward` while their script runs and restores it afterwards; it is not a source edit.

### Limits

- Every enablement is another environment of several gigabytes, and its upstream pin lags behind. We cannot bump its upstream on our own: the owning team rebases and we bump the reference.
- The owning team's scripts are command-line tools without a stability promise, which is why CI re-runs export and parity on every reference bump.
- The series are Apache-2.0, and some pipelines ship `NOTICES` files. Their terms apply if patch content is ever redistributed.

## Usage

### LeRobot Users

In Python, with LeRobot's own configuration objects:

```python
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig

from physicalai.engine import Engine
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

cfg = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id="lerobot/pusht"),
    policy=ACTConfig(push_to_hub=False),
    steps=20_000,
)

engine = LeRobotEngine()  # managed environment; accelerator detected
run = engine.train(cfg)  # LeRobot's own trainer, in the worker
package = engine.export(run, "exports/act")  # manifest.json plus LeRobot's pretrained_model/
model = Engine.load(package)  # Runtime InferenceModel
```

On the command line, with `lerobot-train`'s flags unchanged (see [CLI](#cli)). Models trained with plain `lerobot-train` need no retraining:

```python
LeRobotEngine.export("outputs/train/act_pusht/checkpoints/last/pretrained_model", "exports/act")
```

### First-Party Policies

Nothing changes for first-party users; `Trainer` is the first-party engine:

```python
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT
from physicalai.train import Trainer

trainer = Trainer(max_epochs=10)
trainer.fit(ACT(), datamodule=LeRobotDataModule(repo_id="lerobot/pusht"))
Trainer.export(trainer.lightning_module, "exports/act", backend="openvino")
```

### Choosing the Framework at Run Time

```python
engine = Engine.create(name, **options)  # name from a config file, the CLI, or the GUI
```

### Robots

Robots need Runtime, the loader, and the framework's inference dependencies, but not `physicalai-train` or Lightning:

```bash
pip install physicalai physicalai-lerobot-policy-plugin  # Runtime plus the LeRobot loader
pai run --config runtime.yaml  # InferenceModel(".../exports/act")
```

## Keeping LeRobot Exactly LeRobot

- **Same entry point.** `cli()` runs the environment's own `lerobot-train` script, which is `lerobot.scripts.lerobot_train:main`; it registers third-party plugins and calls the decorated `train()` function. LeRobot's parser handles `sys.argv`, validation, `lerobot_policy_*` packages, `--policy.path` fine-tuning, `--config_path`, `--resume`, and native defaults.
- **Flags, output, and exit codes pass through unchanged.** Studio's options precede the framework name and never enter LeRobot's argv.
- **`sys.argv` is scoped in the worker.** For `run()` and `train()`, `TrainPipelineConfig.validate()` reads `sys.argv` through `parser.get_path_arg("policy")`, `parser.get_cli_overrides("policy")`, and `parser.parse_arg("config_path")`. The worker sets `sys.argv` to `lerobot-train` plus the native arguments.
- **GUI forms use LeRobot's names.** Schemas come from LeRobot's own config types, and Studio's overlays refer to native field names such as `policy.chunk_size`, `steps`, and `batch_size`.
- **No lock-in.** The package keeps LeRobot's `pretrained_model/` folder, so `ACTPolicy.from_pretrained()` and LeRobot's own tools keep working on it.

## Studio Integration

### Plugins Expose Facts; Studio Owns Display

| Concern             | Engine (neutral, no Studio imports)                            | Studio                                                                     |
| ------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------- |
| What can be trained | `policies()`                                                   | Which policies are shown; display name, description, license, minimum VRAM |
| Config shape        | `config_schema(policy)`                                        | Labels, basic and advanced fields, order, fields Studio fills in           |
| Running             | `run(settings, dataset=, output_dir=, device=, hooks=)`        | Dataset snapshots, training targets, queue, persistence, secrets           |
| Packaging           | `export(source, output_path, backend)`, `capabilities(policy)` | Which formats are offered, model registry                                  |

### Studio Side

[#1294](https://github.com/open-edge-platform/physical-ai-studio/pull/1294) added `application/backend/src/training/policy_catalog.py`. It already follows this split: `_STUDIO_POLICIES` holds Studio-owned display data (`AvailablePolicy`: ID, name, description, minimum VRAM) and acts as an allowlist; availability comes from the environment that will run the job, including a `GET /policies` endpoint on each remote trainer. Today it covers first-party policies only. The proposal generalizes it across engines:

```python
_STUDIO_POLICIES = {  # Studio-owned display data, keyed by "<framework>/<policy>"
    "physicalai/act": AvailablePolicy(id="physicalai/act", name="ACT", description="…", min_vram=2 * GiB),
    "lerobot/act": AvailablePolicy(id="lerobot/act", name="ACT (LeRobot)", description="…", min_vram=2 * GiB),
}
_FORMS = {  # plain data, using the framework's own field names
    "lerobot/act": {"basic": ["steps", "batch_size"], "labels": {"steps": "Training steps"}},
}


def available_policies() -> list[AvailablePolicy]:  # the same allowlist rule #1294 uses, across engines
    return [_STUDIO_POLICIES[key] for name in Engine.names() for policy in Engine.get(name).policies()
            if (key := f"{name}/{policy}") in _STUDIO_POLICIES]


def form_schema(framework: str, policy: str) -> dict:  # rendered by the robot-form renderer
    return apply_overlay(Engine.get(framework).config_schema(policy), _FORMS.get(f"{framework}/{policy}"))


# Job runner, in the local backend or on a remote trainer
run = Engine.get(job.framework).run(job.settings, dataset=snapshot.path, output_dir=out, device=device, hooks=hooks)
```

The renderer is the one robot forms already use (`application/ui/src/features/robots/robot-form/robot-schema/schema-form.tsx`), with UI hints under `x-physicalai-ui` such as `advanced_configuration`. Existing bare policy IDs map to `physicalai/<id>`.

### Forms and Config Files

| When            | What                   | Source                                                                               |
| --------------- | ---------------------- | ------------------------------------------------------------------------------------ |
| Before training | Form                   | `config_schema()` with Studio's overlay: labels, defaults, basic and advanced fields |
| Before training | Raw editor (optional)  | The framework's native config, with the fields Studio fills in locked                |
| After training  | Saved settings         | Stored JSON, shown read-only with the same schema; retraining pre-fills the form     |
| After training  | Resolved native config | The framework's own file, for example LeRobot's `train_config.json`, shown as text   |

### Open Items

- **Inference isolation.** GUI live inference and benchmarks run `InferenceModel` in the backend process, so a framework's Runtime adapter and its dependencies must be installable there. Pi05's Transformers range conflicts with Runtime's.
- **Metrics.** Charts read Lightning's `version_0/metrics.csv` (`application/backend/src/services/model_metrics_service.py`); LeRobot writes none. Studio should write a normalized metrics file from `on_step` for every engine, with standard keys (`loss`, `val_loss`, `lr`, `epoch`).
- **Logs.** Worker stdout and stderr must reach the job log.
- **Model registry.** `_EXPORT_BACKENDS` in `application/backend/src/services/model_import_service.py` and `InferenceBackend` in `application/backend/src/schemas/hardware.py` are fixed lists, and model import checks artifacts with `is_file()` while native packages are folders. Model records should store `<framework>/<policy>` and the framework version.
- **Per-target availability.** Remote trainers report `Engine.names()` and `policies()` from their own environment through the endpoint #1294 added.
- **Environment preparation.** A GUI phase for `prepare()`, pre-warming when a plugin is installed, and a story for offline or air-gapped installs.
- **Dataset format version.** Studio records LeRobot v3.0 datasets; an engine pinned to an older LeRobot reads v2.x. Engines declare the formats they accept and convert when needed.
- **Settings versioning.** Stored settings use one framework version's field names. Store the version with them, then migrate or pin.
- **Retraining and resuming** differ per framework (Lightning's `ckpt_path`; LeRobot's `--resume` or `--policy.path`) and should be exposed as capabilities.
- **Trust.** Settings from the UI become native config. Expose only allowlisted fields, keep the raw editor for trusted users, and gate `trust_remote_code` and fields that take arbitrary paths or URLs.

## Package Layout

```text
outputs/train/act_pusht/                    # LeRobot's own output layout
├── checkpoints/last/pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── train_config.json                   # fully resolved native TrainPipelineConfig
│   └── processor files
└── exports/lerobot/
    ├── manifest.json                       # "model": {"artifacts": {"lerobot": "act.lerobot"}}
    └── act.lerobot/pretrained_model/       # native checkpoint folder
```

## Runtime Loader

Each plugin ships a regular `RuntimeAdapter` registered in the existing `physicalai.inference.adapters` group, and that adapter owns inference end to end; [Running Enabled Packages](#running-enabled-packages) covers packages produced by enablements. For LeRobot the adapter loads the folder with `PreTrainedPolicy.from_pretrained()` and `make_pre_post_processors()`, and keeps LeRobot's inference behavior (action queue, observation history, and temporal ensembling) by calling LeRobot's own `select_action()` and `reset()`. Returning one action per call is faithful but runs the model at every control step; returning whole chunks needs observation history in the adapter and is a later optimization.

## Potential Runtime Changes

The POC packages LeRobot artifacts and a reset callback using Runtime's existing adapter and manifest extension points. These broader changes would be useful beyond LeRobot but are not required:

1. `InferenceModel.reset()` also calls a new `RuntimeAdapter.reset()`, which is a no-op by default. Without it, stateful policies keep queues and observation history across episodes. A manifest callback that uses `on_load()` and `on_reset()` can stand in meanwhile.
2. `InferenceModel._get_model_path()` resolves `model.artifacts[backend]` to a path inside the export directory, reusing the containment check in `resolve_artifact()`, so an artifact can be a folder. Today the loader only reads the key and finds files by extension.

`Manifest` already allows extra top-level keys, so provenance (framework name and version, policy ID, and source checkpoint) can be recorded without a schema change. A typed field can follow.

## Placement

```text
physicalai (Runtime)        CLI host (pai), InferenceModel, adapter registry, manifest format
   ▲
physicalai-train            physicalai.engine: Engine base class, registry, WorkerEnv, prepare(), worker runner;
   │                        Trainer (the first-party engine); pai train and pai engine
   ▲                  ▲
plugin[train]      Studio backend and GUI (Engine API; owns display data)

plugin (base) ──► physicalai (Runtime) only: the loader, for robots
```

- The LeRobot inference loader belongs on the deployment side. Runtime's `packages/physicalai-lerobot-plugin` bridges LeRobot robots and teleoperators into Physical AI; the Studio-side `physicalai-lerobot-policy-plugin` registers the LeRobot model loader for inference packages.
- This matches the rule in the modular-packages proposal: deployment-artifact loaders belong with inference, and training code belongs in `physicalai-train`.
- Existing APIs stay. `physicalai.train.Trainer` and `pai fit|validate|test|predict|export|benchmark` keep working unchanged; `Trainer` additionally becomes an `Engine`. The `LeRobotPolicy` Lightning wrapper remains available independently.

## Trade-Offs

What cannot be identical to LeRobot:

- The command is `pai train lerobot ...` instead of `lerobot-train ...`. Flags, defaults, output, and exit codes are the same, and plain `lerobot-train` keeps working.
- **Version compatibility.** The plugin declares a LeRobot range and its worker lock pins one version; upgrading LeRobot means bumping the lock, which Renovate's lock maintenance already does for this repository. Pi05 training works in its own environment, while host-side Runtime inference remains blocked by the Transformers mismatch. GR00T remains metadata-only pending remote-code review.
- Robots run policies through Runtime's `InferenceModel`, not LeRobot's control loop. The native folder still works with LeRobot directly.
- Intel XPU support depends on LeRobot's own device handling (`lerobot.utils.device_utils` detects XPU), so fixes go upstream instead of into Studio. The worker environment installs XPU builds of torch through its `xpu` extra.

What Physical AI gives up compared with the wrapper:

- Studio-only training features (SnapFlow, the PEFT mixins, Studio's image augmentations, and best-`val/loss` checkpoint selection) do not apply to framework-trained models. The framework's own equivalents do, for example LeRobot's `--peft`.
- The GUI training form is generated per framework and policy instead of being one fixed form.
- Each prepared environment takes several gigabytes of disk per engine, accelerator, and lock; `pai engine prune` removes outdated ones. The first preparation downloads them.
- Optimized formats for framework models come only through enablements. Their packages are run by the plugin's Runtime adapter with the framework's own pre- and post-processing, so robots need the framework's inference dependencies in addition to Runtime.
- Frameworks that cannot read LeRobot datasets need a conversion step inside their engine.

## Naming

Settled for this proposal:

- **Engine and framework.** `Engine` names the classes users instantiate; "framework" is the identifier (`Engine.get("lerobot")`, `Run.framework`, entry-point names). Alternatives were rejected: `Framework` or `LeRobotFramework` names the library rather than an object you run; `LeRobotTrainer` collides with the Lightning, Hugging Face, and Studio `Trainer` classes, and the class also exports; `Backend` already means export format, inference backend, the FastAPI server, and `TrainingBackend`; `Runner` and `Adapter` collide with Runtime's `InferenceRunner`, Studio's `TrainerRunner`, and `RuntimeAdapter`.
- **Module.** `physicalai.engine`; `from physicalai import Engine` is not possible because `physicalai` is a namespace package shared with Runtime.
- **Entry-point group.** `physicalai.engines`, pointing at engine classes. It replaces the POC's `physicalai.engine.frameworks` group and its `register(registry)` functions.
- **First-party engine.** `physicalai.train.Trainer`; `PhysicalAIEngine` may alias it.
- **Framework plugin, repo plugin, and enablement.** A framework plugin covers a whole framework; a repo plugin is an engine for a single model's repository. An enablement is a named, pinned bundle that a plugin declares to add a capability, such as training on an accelerator or export to a format, for specific policies.
- **Selecting by name.** `Engine.create()`, `Engine.get()`, and `Engine.names()`. A `get_engine = Engine.create` alias would mirror `get_policy()` if wanted.
- **CLI.** `pai train <framework> …` and `pai engine prepare|list|prune`; `fit`, `validate`, `test`, `predict`, `export`, and `benchmark` stay the Lightning-level commands.

## POC Validation

The POC in the `feat/engine-poc` worktree implemented the earlier design: a routing `Engine.train(config, framework=…)`, a `Framework` protocol with `accepts()` type dispatch, `register(registry)` entry points, `environment="auto"|"manual"` with `uv run --with`, and a stdout progress protocol. This revision replaces those parts; the results below remain the evidence for the native-trainer approach.

- **Passed:** one-step ACT and Diffusion CUDA training, ALOHA ACT training, seeded native/Engine ACT parity in the same interpreter, and pinned pretrained ACT/Diffusion Runtime action parity on CUDA (maximum action difference 0.0 in these runs).
- **Passed:** one real Pi05 native training update and package export in an isolated uv environment on an RTX A6000, using `lerobot/pi05_base@22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0` and episode 0 of `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. The worker resolved LeRobot 0.6.1, Transformers 5.5.4, and PyTorch 2.11.0+cu130, used 12.85 GiB of GPU memory, and installed neither `physicalai` nor `physicalai-train`.
- **Passed:** all library unit tests (1,476 passed, 11 skipped for unavailable optional integrations), `prek run --all-files library/`, and the plugin's Ruff, mypy, and Bandit checks.
- **Gated:** Pi05 native/Runtime inference parity is skipped because the host Runtime's Transformers pin cannot coexist with LeRobot `[pi]`. GR00T N1.7 is config/metadata-only: LeRobot's loader sets [`trust_remote_code=True`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/groot/modeling_groot.py), so no remote model code runs until it is reviewed and pinned. FLUX 3 is future external-plugin work; LeRobot 0.6.1 has no built-in `flux3` policy.
- **Full suite:** not completely green in that environment, for reasons outside the Engine tests. The optional ExecuTorch export test fails because ExecuTorch is not installed, and a first-party ONNX inference test fails while TorchCodec decodes a cached LeRobot video (`Invalid data found when processing input`).
- **Not shipped:** `physicalai-lerobot-policy-plugin` is not published; local development resolves it from the library's uv path source.
- **CLI parity gaps** found in the POC are listed under [CLI](#cli).

LeRobot 0.6.1 fixed its previous Diffusers range conflict (`>=0.38.0,<0.40.0`; LeRobot 0.6.0 required `>=0.27.2,<0.36.0`). Separate environments resolve conflicting pins between frameworks, but they cannot make incompatible requirements compatible within one environment.

## Remaining Decisions

- **GUI paths for LeRobot.** Whether the GUI also offers the Lightning `LeRobotPolicy` path, or only `LeRobotEngine`.
- **Uncurated policies.** Whether policies that an installed engine supports but Studio has not curated are hidden, as #1294's allowlist does today, or shown as experimental with the generated schema.
- **Inference isolation.** How GUI live inference, benchmarks, and robots run packages whose framework dependencies conflict with the backend's or Runtime's, including packages produced by enablements.
- **Validation events.** `TrainHooks` reports steps only; first party also reports `val/loss`.
- **Plugin trust UI.** Plugins and `lerobot_policy_*` packages execute code on import. The GUI could label unverified plugins, as robot plugins do with `TRUST_UNVERIFIED_PLUGIN`.
- **Referenced patch series and SDL.** Whether applying another team's published patch series at install time (pinned, verified by checksum, and recorded in each package) is acceptable. If not, enablements are limited to upstream releases and our own overlay code.
- **Patch subsets.** Whether an enablement may apply a subset of a series, for example to leave out demo changes in shared files.
- **Where enablements live.** In our plugins, as proposed, or later as separate packages.

## References

Studio:

- `library/src/physicalai/policies/lerobot/policy.py`: `LeRobotPolicy`
- `library/src/physicalai/train/trainer.py`: `Trainer`, the first-party engine
- `library/src/physicalai/train/utils.py`: `reformat_dataset_to_match_policy()` and `_get_delta_indices()`
- `library/src/physicalai/data/datamodules.py`: training and validation data loaders
- `library/src/physicalai/export/mixin_policy.py`: `ExportablePolicyMixin.export()`
- `library/src/physicalai/policies/xr0/export_openvino.py`: export-time swaps of transformers modules, the overlay-code precedent
- `library/pyproject.toml`: the `cpu`, `cu128`, and `xpu` extras and their PyTorch indexes
- `library/tests/integration/test_lerobot_wrapper_equivalence.py`: wrapper equivalence tests
- `application/backend/src/training/job.py`: `TrainingJobSpec` and `run_training_job()`
- `application/backend/src/training/policy_catalog.py`: the Studio policy catalog from #1294
- `application/backend/src/services/training_backends/base.py`: `TrainingBackend` and `TrainingContext`
- `application/backend/src/trainer/`: the remote trainer service and its isolated job runner
- `application/backend/src/services/model_import_service.py`: model import validation
- `application/backend/src/services/model_metrics_service.py`: training metrics for the GUI
- `application/backend/src/api/robot_catalog.py` and `application/ui/src/features/robots/robot-form/robot-schema/`: schema-driven robot forms
- `application/docker/Dockerfile`: pinned uv, per-accelerator `uv sync`, and the bundled FFmpeg
- `application/backend/src/plugins/`: `PluginManager` and the curated plugin manifest

Runtime:

- `src/physicalai/cli/main.py` and `src/physicalai/cli/_spec.py`: the CLI host, `_select_subcommand()`, and `SubcommandSpec`
- `src/physicalai/inference/model.py`: `InferenceModel`
- `src/physicalai/inference/adapters/registry.py`: the adapter registry
- `src/physicalai/inference/manifest.py`: `Manifest` and `PolicySource`
- `packages/physicalai-lerobot-plugin`: LeRobot robots and teleoperators

Robotics AI Suite:

- [`src/pipelines/`](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines): downstream enablements as a pinned upstream submodule plus a `git am` patch series, for example `gr00t-n1d7-ov`, `pi05-rtc-ov`, `diffusion-policy-ov`, and `act-sample`; `vla-pi0.5-openvino` is the patch-free wrapper style

LeRobot `v0.6.1`:

- [`pyproject.toml`](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml): Python requirement and optional dependency ranges
- [`src/lerobot/scripts/lerobot_train.py`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_train.py): `main()`, `train()`, and `update_policy()`
- [`src/lerobot/configs/parser.py`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/configs/parser.py) and [`src/lerobot/configs/train.py`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/configs/train.py): CLI parsing and `TrainPipelineConfig.validate()`
- [`src/lerobot/configs/policies.py`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/configs/policies.py): `PreTrainedConfig` and its registered policy types
- [`src/lerobot/datasets/factory.py`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/factory.py): `resolve_delta_timestamps()` and ImageNet statistics
- [`src/lerobot/common/train_utils.py`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/common/train_utils.py): the checkpoint layout
- [`src/lerobot/policies/pi05/configuration_pi05.py`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/pi05/configuration_pi05.py), [`src/lerobot/policies/groot/configuration_groot.py`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/groot/configuration_groot.py), and [`src/lerobot/policies/groot/modeling_groot.py`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/groot/modeling_groot.py): policy and trust/dependency claims

The former Diffusers range is in the [`v0.6.0` `pyproject.toml`](https://github.com/huggingface/lerobot/blob/v0.6.0/pyproject.toml).
