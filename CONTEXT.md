# Physical AI Studio

Physical AI Studio covers robot-learning data, policy training, evaluation, and production of deployment artifacts for Physical AI Runtime.

## Language

### Policy integration

**First-party policy implementation**:
A policy implementation whose model code and ongoing maintenance are owned by Studio.
_Avoid_: Native policy (ambiguous about which project owns it)

**Upstream policy implementation**:
A policy implementation maintained by its originating project rather than Studio.
_Avoid_: Native policy (ambiguous about which project owns it)

**Official API**:
The classes, entry points, and dataset formats an upstream project publishes for its own users. Integrations use it directly rather than recreating it in Studio.
_Avoid_: Native API, Studio wrapper classes for upstream types

**Official checkpoint**:
A trained model saved in its upstream project's own format and kept unchanged in Studio.
_Avoid_: Lightning checkpoint, `model.ckpt`, when the checkpoint came from an upstream workflow

**Dataset conversion**:
Producing a new dataset on disk in another project's dataset format from an existing dataset, which is preserved.
_Avoid_: Format conversion, which converts individual batches in memory

**Upstream-native route**:
A policy workflow whose training lifecycle remains owned by the originating project, including when that project itself uses Lightning.
_Avoid_: Native route without naming the owner

**Policy provider**:
An integration that offers specific policy workflows for an upstream project or Studio-owned implementation. A provider's presence does not imply that every model or capability it exposes is validated.
_Avoid_: Model architecture, support guarantee

**Policy plugin**:
A separately distributed policy integration that an external contributor can supply without modifying Studio core. It can expose a single validated capability rather than the complete policy lifecycle.
_Avoid_: Robot plugin when referring to policy integration

**Curated policy plugin**:
A policy plugin listed in Studio's GUI after review, with a named maintainer, a pinned upstream revision, and evidence for each capability it claims.
_Avoid_: Official plugin ("official" refers to an upstream project's own API)

**Policy capability**:
A specific policy workflow or deployment outcome, such as fine-tuning, inference, export to a particular format, or quantized inference. Capabilities are independent; training support does not imply deployment support.
_Avoid_: Model support without naming the supported workflow

**Validated capability**:
A policy capability backed by evidence for the stated integration, policy/checkpoint revision, software stack, hardware, and scenario. An upstream capability claim alone does not validate Studio's integration.
_Avoid_: Supported because available, supported because registered

**Python environment**:
A prepared Python interpreter and set of installed packages in which a policy workflow runs, however it is provisioned. Compatible policies can share one; it is not a running model, a training job, or a robot setup.
_Avoid_: Execution environment, runtime environment, env, sandbox

**Upstream enablement**:
Capability or hardware improvements contributed to the originating policy or framework project so they are available through that project's supported workflow.
_Avoid_: Studio-only optimization when the change also benefits upstream users

### Studio setup

**Environment**:
A Studio configuration of the robots and cameras used together for recording and deployment.
_Avoid_: Using it to mean a Python environment or a simulation environment
