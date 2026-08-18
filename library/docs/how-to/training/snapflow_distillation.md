# Train a policy with SnapFlow (1-step action generation)

## What SnapFlow is

Flow-matching VLAs such as Pi0.5 and SmolVLA generate an action chunk by
integrating a learned velocity field from noise back to an action over a
`K`-step Euler loop (typically `K = 10`). Every one of those steps is a full
forward pass through the action expert, and on Pi0.5 that loop accounts for
roughly 80% of end-to-end inference latency. Simply setting `K = 1` does not
work: the velocity field is calibrated for small local steps, not one global
jump.

SnapFlow ([arXiv:2604.05656](https://arxiv.org/abs/2604.05656)) compresses that
loop into a **single forward pass (1-NFE)** through self-distillation. Reported
results: on Pi0.5 / LIBERO it reaches 98.75% success against 97.75% for the
10-step teacher, with a 9.6x denoising speedup (274 ms -> 83 ms end to end); on
SmolVLA it cuts offline action MSE by 8.3% with a 3.56x end-to-end speedup.

Both [`Pi05`](../../explanation/policy/README.md) and `SmolVLA` support it.

## Why it needs two phases

SnapFlow is a _fine-tuning_ stage, not a from-scratch training mode. It builds
its distillation target from the model's own velocity predictions, so it needs a
model whose velocity field is already good:

| Phase | What trains                                            | Objective                       | Typical budget                 |
| ----- | ------------------------------------------------------ | ------------------------------- | ------------------------------ |
| 1     | Full model                                             | Standard flow matching          | 5-10 epochs (warm-started VLA) |
| 2     | Action expert + target-time embedding (~10% of params) | Mixed FM + shortcut consistency | 3-5 epochs                     |

Three properties make phase 2 cheap and safe:

- **The VLM backbone is frozen.** Only a thin head moves, so perception and
  language representations cannot drift while the action head is reshaped.
- **The target-time embedding is zero-initialised.** At the start of phase 2 the
  model is numerically identical to the phase-1 teacher — the transition cannot
  regress the model on contact.
- **The `alpha` mix keeps half the batch on standard flow matching.** That
  preserves the multi-step behaviour the shortcut target depends on, preventing
  catastrophic forgetting.

Distilling an undertrained phase-1 model just distills noise. Do not skip
phase 1.

---

## Option 1 — One run, phase switch via callback (recommended)

`SnapFlowPhaseCallback` flips the policy into SnapFlow mode at a configured
phase boundary and rebuilds the optimizer over the now-trainable parameters. No
checkpoint handoff, no second command.

A complete worked config ships at
[`configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml`](../../../configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml):

```bash
physicalai fit --config configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml
```

The parts that matter:

```yaml
model:
  class_path: physicalai.policies.Pi05
  init_args:
    pretrained_name_or_path: lerobot/pi05_base
    # Phase 1 is plain flow matching — the callback turns SnapFlow on later.
    train_expert_only: false
    scheduler_decay_steps: null # cosine horizon = real step budget
    scheduler_warmup_steps: 100

trainer:
  max_epochs: 14 # phase 1 (epochs 0-9) + phase 2 (epochs 10-13)
  precision: bf16-mixed
  callbacks:
    - class_path: physicalai.train.SnapFlowPhaseCallback
      init_args:
        start_epoch: 10 # phase-2 boundary
        alpha: 0.5
        lambda_: 0.1
        num_inference_steps: 1
    # monitor set -> best_model_path also feeds SnapFlowPhaseCallback's default
    # restore_best_teacher. See "Which checkpoint phase 2 distills from" below.
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: val/loss
        mode: min
        save_top_k: 3 # keep the 3 best of each phase (6 files total)
        filename: "epoch{epoch:03d}"
        auto_insert_metric_name: false
```

Note the YAML key is `lambda_`, with a trailing underscore (`lambda` is a Python
keyword). `every_n_epochs` is left at its default of `1` (matching
`check_val_every_n_epoch: 1`) so `monitor` sees every epoch — coarsening it
would let the true best epoch slip through.

At the boundary the callback:

1. Calls `policy.enable_snapflow(alpha, lambda_, num_inference_steps)`, which
   activates the mixed objective, sets `train_expert_only`, freezes the VLM via
   the policy's existing `set_requires_grad()` primitive, and refreshes the
   checkpoint hparams so checkpoints saved afterwards reload as SnapFlow
   policies.
2. Calls `trainer.strategy.setup_optimizers(trainer)` so the optimizer covers
   only the unfrozen parameters and starts with clean state.
3. Prints a phase banner, starts logging `snapflow` to the progress bar, and
   prefixes new checkpoint filenames — see the next section.

### What to expect at the phase boundary

The transition is deliberately noisy, because a run silently changing its
training objective is worse than a run that tells you.

**A banner is printed once**, routed through the progress bar so it does not
garble the bar's own output:

```text
==============================================================================
SnapFlow distillation ENABLED at step 30000 (epoch 10)
  alpha=0.50  lambda_=0.10  num_inference_steps=1
  trainable params: 311.9M / 3138.4M (9.9%) - VLM backbone is now frozen
  Optimizer and LR scheduler rebuilt; phase 2 restarts the warmup.
  Expect slower steps: the consistency branch runs 3 velocity passes per sample.
  checkpoints from here on: 'snapflow-epoch{epoch:03d}.ckpt'
==============================================================================
```

**The progress bar gains a `snapflow` entry** for the rest of the run. It is not
logged during phase 1, so the key's presence is itself the phase indicator:

```text
Epoch 10/13 ━━━━━━━━━╺━━━━━━ 812/1875 train/loss: 0.243  snapflow: 1.000
```

**Checkpoints written after the boundary are prefixed** with `snapflow-`, so a
staged run is unambiguous on disk:

```text
epoch008.ckpt            <- phase 1, top-3 best-val-loss
epoch009.ckpt            <- phase 1, top-3 best-val-loss
epoch010.ckpt            <- phase 1, top-3 best-val-loss (the true best)
snapflow-epoch012.ckpt   <- phase 2, top-3 best-val-loss (1-NFE), distilled
snapflow-epoch013.ckpt   <- phase 2, top-3 best-val-loss (1-NFE), distilled
```

Change the prefix with `checkpoint_prefix: "distilled-"`, or set it to `null`
to leave filenames alone. Every checkpoint also carries the phase in
machine-readable form:

```python
import torch

ckpt = torch.load("snapflow-epoch012.ckpt", weights_only=False)
print(ckpt["snapflow"])
# {'enabled': True, 'alpha': 0.5, 'lambda_': 0.1,
#  'num_inference_steps': 1, 'activated_at_step': 30000}
```

### Which checkpoint phase 2 distills from

SnapFlow's shortcut target is bootstrapped from the model's own velocity
predictions, so distilling an undertrained phase-1 model distills noise.

By default (`restore_best_teacher: true`), at the boundary the callback loads
the best-`val/loss` weights from a monitored `ModelCheckpoint`
(`monitor is not None` — an unmonitored one is ignored, since its
`best_model_path` just means "latest") **before** calling `enable_snapflow()`.
If none is configured or it hasn't saved yet, it warns and falls back to the
live model. Set `restore_best_teacher: false` to always use the live model;
set `best_teacher_monitor` if more than one monitored checkpoint exists.

`val/loss` is also not comparable across the boundary: `compute_val_loss` runs
the full denoising loop, so its step count drops from `num_inference_steps`
to SnapFlow's (typically 1-NFE) count the instant the objective changes.
Ranking phase-2 checkpoints against phase-1 scores would almost always favor
an un-distilled checkpoint. So, by default (`scope_best_to_phase: true`), the
callback also resets the monitored checkpoint's tracking state at the
boundary, so it tracks phase-2's own best from a clean slate — nothing on
disk is deleted by the reset. Set `scope_best_to_phase: false` to keep phase-1
scores in the running (not recommended).

The banner reports both actions:

```text
  Distillation teacher: restored best checkpoint '/mnt/data/experiments/snapflow-multi-object-augment/epoch010.ckpt'.
  Best-checkpoint tracking reset for monitor(s) ['val/loss']: val/loss is not comparable across num_inference_steps.
```

**The first phase-2 step is slow.** Two independent reasons:

- The consistency branch runs three velocity passes per sample (`v_1`, `v_half`,
  `v_pred`) where flow matching runs one, so phase-2 steps are ~2-3x slower for
  the whole phase. This is by construction.
- With `compile_model: true`, flipping the SnapFlow flag invalidates the
  `torch.compile` guards along with the `requires_grad` and eval-mode changes
  from freezing the VLM, so the first phase-2 step pays a full recompile of a
  new, larger graph. That can take minutes on a VLA. It is not a hang.

### Step-based boundary

If your run is budgeted in steps rather than epochs, use `start_step` instead.
Exactly one of the two must be set:

```yaml
- class_path: physicalai.train.SnapFlowPhaseCallback
  init_args:
    start_step: 30000
```

### Useful overrides

```bash
# Smoke-test the wiring without training anything
physicalai fit --config configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml \
    --trainer.fast_dev_run 1

# Smaller GPU
physicalai fit --config configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml \
    --data.train_batch_size 8 --trainer.accumulate_grad_batches 2

# Longer total budget
physicalai fit --config configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml \
    --trainer.max_epochs 20
```

### Caveat: the phase-2 LR horizon

Rebuilding the optimizer also rebuilds the LR scheduler, so phase 2 gets a fresh
warmup. The cosine decay horizon, however, is derived from
`Trainer.estimated_stepping_batches`, which reports the _total_ run budget
rather than the phase-2 remainder. Phase 2 therefore decays more slowly than a
standalone phase-2 run would. If you need an exact phase-2 decay horizon, use
Option 3 (Python API), which warm-starts phase 2 from the phase-1 weights only
and constructs a fresh `Trainer` — Option 2's `--fit.ckpt_path` is a full
resume and has the same combined-horizon behavior as Option 1.

---

## Option 2 — Two explicit runs

Use this when you are distilling from a checkpoint you already have (for
example a published checkpoint, or a phase-1 run from a separate machine or
job). Note that `--fit.ckpt_path` is a full Lightning resume, so this does
**not** give phase 2 an independent LR horizon — see the caveat below.

```bash
# Phase 1 — standard flow-matching training
physicalai fit --config configs/physicalai/pi05.yaml

# Phase 2 — SnapFlow distillation, VLM frozen, resumed from phase 1
physicalai fit \
    --config configs/physicalai/pi05_snapflow_distillation.yaml \
    --fit.ckpt_path ./experiments/lightning_logs/version_0/checkpoints/last.ckpt \
    --trainer.max_steps 60000
```

Substitute `pi05` with `smolvla` for the SmolVLA policy. Phase-2 templates:

- `configs/physicalai/pi05_snapflow_distillation.yaml`
- `configs/physicalai/smolvla_snapflow_distillation.yaml`

Both set `snapflow_enabled: true`, `train_expert_only: true`, and the paper
defaults (`snapflow_alpha: 0.5`, `snapflow_lambda: 0.1`,
`snapflow_num_inference_steps: 1`).

Two things to be aware of:

- The flag is **`--fit.ckpt_path`**, not `--ckpt_path`. Method-level arguments
  are namespaced under the subcommand (`--validate.ckpt_path`,
  `--test.ckpt_path`, `--predict.ckpt_path`).
- `--fit.ckpt_path` is a full Lightning **resume**: it restores the global step,
  optimizer state, and LR schedule from phase 1. Set `--trainer.max_steps` to
  the _combined_ phase-1 + phase-2 budget, not the phase-2 budget alone. This
  means Option 2 does **not** give phase 2 an independent LR horizon either —
  the LR schedule still spans both phases, exactly like Option 1's
  single-run caveat above. There is currently no supported way to warm-start
  phase 2 from a checkpoint's weights only (a fresh optimizer/LR schedule);
  doing so would require loading just the `state_dict` manually instead of
  passing `--fit.ckpt_path` (see Option 3 below).

---

## Option 3 — Python API

Prefer warm-starting phase 2 from the **best** phase-1 checkpoint (lowest
`val/loss`), not just whatever happens to be `last.ckpt`. Add a
`monitor="val/loss", mode="min"` `ModelCheckpoint` to the phase-1 run and use
its `best_model_path`:

```python
from lightning.pytorch.callbacks import ModelCheckpoint

best_ckpt_cb = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1)
Trainer(max_epochs=10, callbacks=[best_ckpt_cb]).fit(phase1_policy, datamodule=datamodule)
```

```python
from physicalai.policies import Pi05
from physicalai.train import Trainer

policy = Pi05.load_from_checkpoint(
    best_ckpt_cb.best_model_path,  # or any known-good checkpoint path
    map_location="cpu",
    snapflow_enabled=True,
    snapflow_alpha=0.5,
    snapflow_lambda=0.1,
    snapflow_num_inference_steps=1,
    train_expert_only=True,
    # compile_model is excluded from saved hparams, so re-pass it explicitly.
    compile_model=True,
)

Trainer(max_epochs=3, precision="bf16-mixed").fit(policy, datamodule=datamodule)
```

Or flip an already-constructed policy after `setup()` has run:

```python
policy.enable_snapflow(alpha=0.5, lambda_=0.1, num_inference_steps=1)
```

---

## Hyperparameter guidance

| Parameter                | Paper default            | Notes                                                                                                                           |
| ------------------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `alpha`                  | `0.5`                    | FM-loss weight. Keep at or above `0.5` to preserve multi-step ability.                                                          |
| `lambda_`                | `0.1`                    | Shortcut-loss scale, balances the two gradient magnitudes.                                                                      |
| `num_inference_steps`    | `1`                      | `1` gives the full SnapFlow speedup; raise it for intermediate modes.                                                           |
| `checkpoint_prefix`      | `"snapflow-"`            | Marks phase-2 checkpoints on disk. `null` disables the rewrite.                                                                 |
| `restore_best_teacher`   | `true`                   | Restore the monitored `ModelCheckpoint`'s best-val-loss weights before enabling SnapFlow, instead of distilling the live model. |
| `best_teacher_monitor`   | `None`                   | Disambiguates which monitored `ModelCheckpoint` to restore from when more than one is configured.                               |
| `scope_best_to_phase`    | `true`                   | Reset best-checkpoint tracking at the boundary, since `val/loss` is not comparable across `num_inference_steps`.                |
| Phase-1 budget           | 5-10 epochs              | Fewer for a warm-started VLA than for from-scratch training.                                                                    |
| Phase-2 budget           | ~3-5 epochs (~30k steps) | Short because the target-time embedding is zero-initialised.                                                                    |
| `scheduler_warmup_steps` | ~5% of total steps       | No fractional option exists; compute it from your dataset (below).                                                              |

Converting an epoch budget into steps, for warmup sizing:

```text
steps_per_epoch = floor(train_frames / (batch_size * devices)) / accumulate_grad_batches
total_steps     = max_epochs * steps_per_epoch
warmup          = 0.05 * total_steps
```

Set `scheduler_decay_steps: null` so the cosine horizon follows
`Trainer.estimated_stepping_batches` and the LR lands on `scheduler_decay_lr`
exactly at the end of the run.

Hold out a validation split (`data.init_args.val_split`) on small datasets.
Without it there is no way to distinguish convergence from memorisation.

---

## Verifying the transition

After the phase boundary, check that:

- The log line `SnapFlowPhaseCallback: activating SnapFlow distillation at
step N / epoch M` appeared, followed by the phase-2 trainable-parameter
  count. That count should be roughly 10% of total parameters.
- `train/loss` shifts level at the boundary — expected, the objective changed.
  It should settle rather than diverge.
- `val/loss` shifts level at the boundary too — expected, since its step count
  drops from `num_inference_steps` to SnapFlow's (typically 1-NFE) count the
  instant the objective changes (see "Which checkpoint phase 2 distills from").
  Within phase 2 it should trend down or hold steady; a steady climb means
  something is misconfigured.
- Every VLM parameter reports `requires_grad == False`:

  ```python
  frozen = [n for n, p in policy.named_parameters() if not p.requires_grad]
  ```

- A 1-step benchmark matches or beats the multi-step teacher. Prefer phase 2's
  best checkpoint (`ckpt_cb.best_model_path`), not just the last epoch saved:

  ```bash
  physicalai benchmark --config configs/benchmark/libero.yaml \
      --ckpt_path experiments/.../snapflow-epoch013.ckpt
  ```

## Exporting the distilled policy

The SnapFlow checkpoint exports like any other policy. The exported artifact
carries the 1-NFE sampling path, which is where the latency win materialises for
Runtime. As with benchmarking, prefer phase 2's best checkpoint:

```bash
physicalai export --ckpt_path experiments/.../snapflow-epoch013.ckpt --backend openvino
```

See [Export and deploy](../export/export_inference.md).
