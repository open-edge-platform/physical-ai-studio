# Run 12 — `862b848f` — FAILED at step 1750 / 5000 (2026-08-01)

Companion to §10 of `firmware-fix-report-2026-07-30.html`. Read the §9
addendum there first — it established that firmware age was never the cause
and localized Run 11's faults to GT1 / `action 3003`.

**Run 12 breaks that GT1 pattern.** Same lockup site, but zero GT1
involvement. See "The signature moved AGAIN" below.

## Config

| | |
|---|---|
| job id | `862b848f-3122-47da-abc2-d329f502194a` |
| dataset | `f6e480e2` — so101_sorting_colored_marks, 150 eps (135 train / 15 val) |
| policy | pi05, from base (`base_model_id: null`) |
| batch / workers | 8 / 4 |
| max_steps | **5000** |
| precision | bf16-mixed |
| GPU | **B70** (`xpu:0`, 34.2 GB) — `device: null` → Lightning takes device 0 |
| GuC firmware | 70.65.0 |
| kernel | 6.17.0-41 (boot −1) |
| launched | 2026-07-31 21:51 UTC |
| died | 2026-08-01 06:28 UTC — 8h36m in |

## Timeline (container logs UTC, kernel log PDT — 7h offset)

| UTC | Event |
|---|---|
| 07-31 21:51 | Run starts |
| 08-01 02:34–02:53 | Validation at step 1000, `val/loss=0.14838`, 24.5 GB ckpt written |
| 08-01 06:28:00 | **Last training step: 1750/5000**, loss 0.0657 |
| 08-01 06:28:51 | Last `JOB_UPDATE`. Log goes silent. |
| 08-01 06:29:30 | **First hard lockup on CPU 4** — 90 s after the last step |
| → 08-03 00:39 | Hung **42 hours**. 721 hard + 3520 soft lockups, all CPU 4. |
| 08-03 00:39 | Container restarted → job marked `failed` |

```
status   failed
progress 35
message  "Job aborted due to application shutdown"   <- describes the RESTART, not the cause
```

Unlike Run 11 (faulted 9 min in, froze at 1473), Run 12 ran **8.5 h clean**
then froze. First fault and freeze are the same event here.

## THE SIGNATURE MOVED AGAIN — Run 12 is not GT1

§9 concluded "every one of the 2,524 driver errors is on GT1" and "action
3003 — 100% of failures". **That does not hold for Run 12.** Counted in both
the watch log and the kernel journal:

| Marker | Run 11 `490a12e3` | Run 12 `862b848f` | journal boot −1 |
|---|---|---|---|
| Hard lockups | 398 | **1030** (721 in journal) | 721 |
| Soft lockups | 1435 | 0 in watch log | **3520** |
| `GT1` mentions | **4193** | **0** | **0** |
| `GT0` mentions | 20 | 2 | 0 |
| `action 3003` | **1699** | **2** | **0** |
| `G2H` lines | 1257 | 2 | 0 |

The two `action 3003` lines in Run 12's watch log are at **09:48 UTC — over
3 hours AFTER the 06:28 freeze** — and both marked `done`. They are not the
failure.

**So the media-tile / GT1 theory from §9 does not explain Run 12.** Either
there are two distinct lockup mechanisms, or GT1 was always a symptom rather
than the cause. §9's LEADING candidate needs downgrading.

Caveat on the count asymmetry: Run 12's watch log has 0 soft lockups while
the journal has 3520, so the watch script was dropping lines. The `GT1: 0`
result is nonetheless confirmed independently in the journal.

## Root cause — as far as evidence goes

Same wedge site as before: a CPU stuck in the `xe` GuC message-ring read,
inside the interrupt handler, interrupts disabled. It cannot yield, the core
is gone, and training blocks on a GPU submit that never completes.

```
RIP: 0010:g2h_read+0xb1/0x420 [xe]
Call Trace:
 <IRQ>
 xe_guc_ct_fast_path+0x72/0x140 [xe]
 xe_guc_irq_handler+0xaa/0xc0 [xe]
 gt_irq_handler+0x3cb/0x430 [xe]
 dg1_irq_handler+0x156/0x270 [xe]
 </IRQ>
 <TASK>
 RIP: 0010:cpuidle_enter_state+0xca/0x6e0     <- CPU4 was ASLEEP when the IRQ arrived
 cpuidle_enter+0x30/0x50
 call_cpuidle+0x22/0x60
 Comm: swapper/4                               <- the idle task
```

RIP distribution across all 721 journal lockups:

```
1831  smp_call_function_single      <- other cores piling up behind CPU4 (collateral)
1328  smp_call_function_many_cond   <- same
1132  cpuidle_enter_state           <- CPU4 asleep when IRQ landed
1100  memcpy_fromio                 <- the MMIO read that never returns
  60  g2h_read
```

Run 12 shows **both** prior signatures at once: `memcpy_fromio` (Run 10's)
and `g2h_read` (Run 11's). §9 treated those as distinguishing; they are not.

### New leading theory: deep C-states race the GPU IRQ

Not considered in §9. All 721 lockups are on **CPU 4 only** — the single
core servicing that GPU IRQ.

```
C3_ACPI   latency=1048us   usage=9,139,916    <- 1 ms exit latency, heavily used
intel_idle.max_cstate = 9                     <- unclamped
pcie_aspm.policy = default                    <- not "performance"
```

One-core concentration + `cpuidle_enter_state` in the trace + 1 ms wake
latency inside an MMIO read is a plausible race window. **Circumstantial,
not proven** — the deeper defect is that the driver has no timeout on that
mailbox read. Clamping may only make it rarer.

## Cross-run table (extends §9's per-boot table)

| Boot | Window | GuC | Kernel | Hard | Soft | GT1 errs | Run | Died at |
|---|---|---|---|---|---|---|---|---|
| −4 | Jul 20→27 | 70.45.2 | -40 | 0 | 0 | 0 | — | 30 OOM kills (Failure A) |
| −3 | Jul 27→29 | 70.45.2 | -41 | 0 | 0 | 0 | — | clean |
| −2 | Jul 29→30 | 70.45.2 | -41 | 81 | 480 | 239 | 10 `8108c5ba` | step **1750**/2000 |
| −1 | Jul 30→31 | 70.65.0 | -41 | 398 | 1435 | 4193 | 11 `490a12e3` | step 1473/3000 |
| −1 | Jul 31→Aug 2 | 70.65.0 | -41 | **721** | **3520** | **0** | 12 `862b848f` | step **1750**/5000 |

Runs 11 and 12 share boot −1 (Jul 31 10:44 → Aug 2 17:34).

**Runs 10 and 12 both died at step 1750** — different firmware, different
`max_steps` (2000 vs 5000), same step. Run 11 died at 1473. Two of three at
exactly 1750 is worth chasing but is NOT yet a proven pattern; 1750 is also
just where a ~8.5 h run happens to land.

## RULED OUT — with evidence

| Suspect | Evidence against |
|---|---|
| **Firmware** | Already disproven in §9. Run 12 confirms: 70.65.0, worse than ever. Model `01a71dc5` trained fine on the same firmware. |
| **GT1 / media tile / action 3003** | **Newly weakened.** 0 GT1 mentions, 0 `action 3003` in Run 12's journal. §9's leading candidate does not explain this run. |
| **Kernel too old** | 6.17.0-41 IS the apt candidate; nothing newer available. (§9 already noted -41 is necessary-not-sufficient — boot −3 was clean on -41.) |
| **ReBAR misconfigured** | B70 BAR = 32 GB, B50 = 16 GB. Both correct. |
| **System RAM / OOM** | Zero OOM in boot −1. Died at 35%, nowhere near the final save. This is Failure B, not A. |
| **NaN divergence** | `gradient_clip_val=1.0` held. Loss 0.0657 at the last step. |
| **The B50** | Idle. `device: null` → `xpu:0` → torch enumerates 0 as B70. |

### Which card wedged is UNPROVEN

Boot −1's journal has **zero** `[drm]`/`xe 0000:..` lines, so nothing ties
the stuck handler to a BDF. Current mapping (affinity `0-15` on both, so it
floats):

```
IRQ 179 -> 0000:03:00.0  (B70, training GPU)  currently CPU 8
IRQ 184 -> 0000:07:00.0  (B50, idle)          currently CPU 6
```

Every lockup was CPU 4 — neither of today's assignments.

**Watch-script gap:** it grepped only lockup/tainted and dropped drm/xe
lines (and missed 3520 soft lockups). Next run:

```bash
journalctl -kf | grep -iE "xe |drm|guc|reset|lockup|GT[01]"
```

## SALVAGE — one checkpoint, exported

Steps 1000–1750 are gone. Verified intact by loading: `zip ok: True`,
`global_step: 1000`, 819 tensors.

```
source  /app/storage/cache/862b848f-.../stepstep=001000.ckpt  (24.5 GB)
model   1d1670fe-efc1-4a8e-98be-5532549e1304
name    "Pi0.5 Sorting 150eps step1000 rescued (val 0.148)"
export  exports/torch/pi05.pt  (9.4 GB)  — torch only
```

Registered by hand (DB backed up to `physicalai.db.bak-before-1d1670fe`),
confirmed live at `localhost:7860/api/projects/.../models`.

### Leaderboard — nothing has beaten `15842eae`

| Model | Eps | global_step | val/loss | Backends |
|---|---|---|---|---|
| `15842eae` | 91 | 1001 | **0.1187** BEST | openvino, torch |
| **`1d1670fe`** (Run 12, rescued) | 150 | 1000 | **0.1484** | torch |
| `7c1f0a94` | 150 | 1000 | 0.1540 | openvino, torch |
| `01a71dc5` | 150 | 1000 | 0.1720 | torch |
| `c5d4500a` | 40 | 3676 | 0.1917 | openvino, torch |

Registry names misstate steps: `7c1f0a94` is named "step1000" but metrics
run to 1749 — the CHECKPOINT is at 1000, metrics kept logging past it. Same
for `01a71dc5` (metrics to 1449).

Run 12 gained nothing on val loss. 91 episodes still beats 150.

## FIXES — ranked, none applied yet

Needs `sudo` from a real terminal (prefix `!` in Claude Code).

**1. Clamp C-states + ASPM — new, highest value, needs reboot**

```bash
sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_idle.max_cstate=1 processor.max_cstate=1 pcie_aspm.policy=performance/' /etc/default/grub
sudo update-grub && sudo reboot
```

Costs idle power and heat. Reversible.

**2. Pin GPU IRQs off CPU 0/4 — no reboot, damage limitation only**

```bash
echo 2 | sudo tee /proc/irq/179/smp_affinity_list   # B70 -> CPU2
echo 3 | sudo tee /proc/irq/184/smp_affinity_list   # B50 -> CPU3
```

Won't stop the wedge; keeps a stuck core from starving RCU/dockerd — which
is what made the box unusable for 42 h (§9 saw the same starvation).

**3. Shorten engine job timeout — free, low value**

```
/sys/class/drm/card1/device/tile0/gt0/engines/rcs/job_timeout_ms = 5000 (max 10000)
```

A timeout exists and did not save this run — the wedge is in the IRQ
handler, which never yields to let it fire.

**4. Checkpoint every 250 steps + progress watchdog — not a fix, but the
only thing that makes a long run survivable**

`val_check_interval=1000` (local.py:106) meant a wedge at 1750 cost 750
steps. At 250 it costs ≤250. Add a watchdog: no `Training progress` line for
~10 min → kill the job. Run 12 hung silently for **42 hours**.

Remember the baked-source gotcha — `docker cp` to BOTH paths
(RESUME-HERE-2026-07-30.md §6).

**No true resume exists.** `base_model` loads weights only
(local.py:58-59) — no optimizer or LR-scheduler state.

## Open questions

1. **Why did GT1 vanish?** Highest value. §9's leading theory rests on GT1 /
   `action 3003`; Run 12 has neither. Two mechanisms, or was GT1 always a
   symptom?
2. **Why step 1750 twice** (Runs 10 and 12, different firmware)? Test with
   `val_check_interval=250` — if it still dies at 1750, it's step-linked;
   if it dies elsewhere, it's time-linked (~8.5 h).
3. Does the C-state clamp help or just delay?
4. Which card's IRQ wedges? Needs the fixed watch command.
5. Does `15842eae` (0.1187, 91 eps) beat `1d1670fe` (0.1484, 150 eps) on the
   REAL ARM? More episodes keep losing on val loss — does val loss even
   predict arm performance here?

## Side observations

- **RAM leak again** (§9's Failure A, mid-session): 29/30 GB used, 20 GB
  swapped, 1.2 GB available, one idle python at 26.4 GB. `docker restart`
  reclaimed 18 GB. Do this before loading any model.
- The 24.5 GB copy pushed IO pressure to 65% (`full avg10=63.20`), load
  13.55, swap back to 20 GB — expected, transient.
- Disk: 153 GB free / 915 GB (83%). `models/` 144 GB, `cache/` 23 GB.
