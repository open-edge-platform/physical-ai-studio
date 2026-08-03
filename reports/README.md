# Training reports — Intel Arc XPU / SO-ARM101

Field notes from fine-tuning Pi0.5 on a physical SO-ARM101 arm using Intel Arc
GPUs. These document the failures that motivated the code changes in this PR.

Hardware: Intel i9-11900T · 30 GB RAM · Arc Pro **B70** (34.2 GB, `xpu:0`,
`0000:03:00.0`) + Arc Pro **B50** (17.1 GB, `xpu:1`, `0000:07:00.0`) ·
Ubuntu, kernel 6.17.0-41, `xe` driver, GuC firmware 70.65.0.

## Start here

| File | What it covers |
|---|---|
| `firmware-fix-report-2026-07-30.html` | **The main issue log.** §1–8 the GuC firmware upgrade, §9 why firmware was *not* the cause, §10 Run 12 and why the GT1 theory also fails |
| `training-report-2026-07-30.html` | Loss vs steps across all runs, episodes vs resulting val loss, dataset inventory, run history |
| `RUN12-FAILURE-862b848f.md` | Newest failure: froze at step 1750/5000, hung 42 h undetected |

Open the HTML files in a browser — both are self-contained (inline CSS/JS,
light and dark themes, no network calls). The firmware report's §10 is at
anchor `#run12`.

## Supporting notes

| File | What it covers |
|---|---|
| `SESSION-2026-07-28-nan-divergence.txt` | The NaN divergence at ~step 2775 that motivated `gradient_clip_val=1.0` |
| `RUN11-STATUS-490a12e3.md` | Run 11 launch config and checkpoint ETAs |
| `success-rate-log.txt` | Physical arm trial log — per-attempt outcomes |

## Two distinct failure modes

Worth separating, because they have different causes and different fixes:

**Failure A — dies *after* training, during save/export.** Host RAM
exhaustion. A completed 5000-step run held ~26 GB of trainer state on a 30 GB
host, then OpenVINO conversion asked for several GB more. OOM kill is SIGKILL,
so no `try/except` can rescue it. Runs `ec4c4cef` (2000 steps) and `9200784f`
(5000 steps) both finished all their training steps and produced **no model**.

→ Fixed in this PR: `del trainer, l_dm` + `_release_memory()` before export,
and OpenVINO made opt-in via `PHYSICALAI_EXPORT_BACKENDS`.

**Failure B — hangs *mid*-training.** A CPU core takes a GPU interrupt, spins
forever in the `xe` driver's GuC message-ring read (`g2h_read` /
`memcpy_fromio`) with interrupts disabled, and can never yield. Training
freezes mid-step with no error, no exception, no OOM.

→ **Not fixed** — it is an Intel `xe`/GuC driver bug, not something this
codebase can repair. The checkpointing change in this PR is damage
limitation: it bounds how much work a hang destroys.

## What has been ruled out for Failure B

With evidence, across five boots:

| Hypothesis | Verdict |
|---|---|
| GuC firmware too old | **Disproven.** Boot −3 was clean for two days on the old 70.45.2; the first-ever fault also occurred on 70.45.2. Upgrading to 70.65.0 made lockups 4.9× *more* frequent. |
| GT1 / media tile / `action 3003` | **Weakened.** Explained Run 11 (4193 GT1 events) but Run 12 had **zero** GT1 events and zero `action 3003`. |
| Kernel too old | Contributing at most — 6.17.0-41 is the current distro candidate, and boot −3 was clean on it. |
| ReBAR, system RAM, NaN divergence, thermal/PCIe | Ruled out — see §9 and §10. |
| Deep CPU C-states racing the GPU IRQ | **Current leading candidate, untested.** All 721 lockups landed on one core with `cpuidle_enter_state` in the trace, C3 exit latency 1 ms, `intel_idle.max_cstate` unclamped. Circumstantial. |

## Practical guidance

- **Cap runs at 1000 steps** on this hardware. Runs at 2000/3000/5000 have all
  failed; every model that actually worked was trained at 1000.
- **Restart the container before loading a model.** A separate VRAM/RAM leak
  leaves an idle worker holding ~26 GB; a restart reclaims ~18 GB.
- **There is no true resume.** `base_model` loads weights only — no optimizer
  or LR-scheduler state — so a rescued checkpoint is not equivalent to
  continuing a run.
- Numbers in these reports come from `journalctl -k`, container job logs, the
  `jobs`/`models` tables, and direct `torch.load` verification of checkpoints.
  Where a figure is inferred rather than measured, the reports say so.
