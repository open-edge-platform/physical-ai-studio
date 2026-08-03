# Run 11 — `490a12e3` — launched 2026-07-30 22:07 PDT

First run on the fixed GuC firmware (70.65.0). 3000 steps, checkpoints at
1000 / 2000 / 3000 so all three can be tested tomorrow.

## Config

| | |
|---|---|
| job id | `490a12e3-c5c5-4df1-82f9-7c59a2e40421` |
| model name | Pi0.5 Sorting 150eps 3000steps (fw 70.65.0) |
| dataset | `f6e480e2` — so101_sorting_colored_marks, 150 eps, 100,309 frames |
| policy | pi05, **from base** (`base_model_id: null`) |
| batch / workers | 8 / 4 |
| max_steps | 3000 |
| val_split | 0.1 |
| precision | bf16-mixed |

## Measured at launch

- **3.47 steps/min** — full speed (target ~3.6; a VRAM-starved run does ~0.3)
- VRAM 32,640 MiB — model resident on the GPU
- loss at step 12: 0.329, descending from 0.615 at step 1
- 0 kernel faults

## Checkpoint ETAs

| ckpt | elapsed | expected |
|---|---|---|
| step 1000 | +4.8 h | Fri ~02:55 |
| step 2000 | +9.6 h | Fri ~07:44 |
| step 3000 | +14.4 h | Fri ~12:32 |

## IMPORTANT — stop manually at step 3000

`local.py:107` saves an extra final `model.ckpt` **after** `fit()` returns,
on top of the three interval checkpoints. That 24.5 GB save is what OOMed
before (24.5 GB ckpt + training RAM > 32 GB host RAM). Once step 3000's
checkpoint is on disk, **stop the job from the GUI** rather than letting it
complete naturally.

Disk: 146 GB free at launch. 3 x 23 GB = 69 GB, leaving ~77 GB. A 4th
(final) save would drop that to ~54 GB.

## Fault monitor is running

```
local-changes/gpu-fault-watch-490a12e3.log
```

Background `journalctl -kf` filtered to `LOCKUP|GuC|G2H|reset started|DEVICE_LOST`.
**An empty file is the good outcome.** Run 10's first lockup came 1h32m in
and it kept reporting healthy steps for 7 more hours — the fault was
invisible from inside the job, which is why this log exists.

Check it with:
```bash
wc -l local-changes/gpu-fault-watch-490a12e3.log   # 0 = healthy
```

Any content means the card is degrading: stop the run and keep the last
checkpoint rather than losing everything.

## Gotcha hit at launch — worth remembering

VRAM read **15,860 MiB** before launch, not 34. The inference worker from
the arm testing (host PID 4527, child of backend 3982) was still holding
~15.5 GB. `docker restart` released it (15,860 -> 44 MiB). Launching without
that check would have spilled to swap and run 10-15x slower.

The gate check is not optional, and it is not just about *stopped training
runs* — **inference model loads leak too**.

## What this run is testing

Whether 3000 steps produces smoother, more confident arm motion. Note the
existing evidence runs the other way: every 1000-step run has beaten every
3000+ run on val loss on this box.

| model | steps | val loss |
|---|---|---|
| `15842eae` | 1000 | **0.1187** best |
| `7c1f0a94` | 1000 | 0.1540 |
| `28c2e9ca` | 3385 | 0.1327 |
| `c5d4500a` | 3675 | 0.1917 worst |

Having all three checkpoints means the comparison is direct: test 1000 vs
2000 vs 3000 on the arm and let the hardware settle it. If 1000 wins again,
that is a real finding about this box and this dataset size.

## Before testing any of these tomorrow

1. **Fix the wrist camera mount first** (blue threadlocker). The screws
   loosen on bin contact, and the wrist camera is a policy input — a
   drifting camera silently invalidates trial data. Round 2's
   "last block always fails" pattern is equally consistent with camera
   drift accumulating over a long scene.
2. Re-check VRAM is ~34 MiB before each model load.
3. `pkill firefox` — or launch it with `LIBGL_ALWAYS_SOFTWARE=1
   MOZ_WEBRENDER=0`, since Firefox grabs `renderD128` = the training GPU.
4. Skip the OpenVINO exports until the conversion defect is diagnosed —
   OpenVINO could not pick at all on `7c1f0a94` while torch worked.
