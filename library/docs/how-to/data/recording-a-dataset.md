# How to record a dataset for imitation learning

A goal-oriented guide for recording demonstration datasets that actually train
well. It codifies the [LeRobot AGENT_GUIDE §5](https://github.com/huggingface/lerobot/blob/main/AGENT_GUIDE.md)
data recommendations.

> **Good data beats clever models.** Every hour spent on rig setup and recording
> discipline saves several hours of training and debugging.

This guide covers **what to record and why**. For the click-by-click mechanics
of the recorder and the episode viewer, see the Physical AI Studio guides:

- [Recording Datasets](../../../../application/docs/05-recording-datasets.md) —
  creating a dataset, the record loop, reviewing and exporting episodes.
- [Training Policies](../../../../application/docs/06-training-policies.md) —
  training a model from the Models page.

Physical AI Studio identifies datasets by their own dataset ID, not a Hugging
Face repo — there is nothing to configure there before you start.

---

## The single most important rule: record cyclic episodes

**An episode must end in the same state it started in.**

```text
BAD   home -> approach -> grasp -> move to target -> release              [stop recording]
GOOD  home -> approach -> grasp -> move to target -> release -> return home -> settle ~1s   [stop recording]
```

If you stop recording at the target and reset the arm manually before the next
episode, the dataset never shows the model what to do after a successful
execution. At inference the policy finishes one repetition and is immediately in
a state it has never seen — it cannot chain repetitions, and behaviour becomes
undefined.

Add a deliberate ~1 s settle at home before you click accept, so the terminal
state is consistent across episodes. There's a second, less obvious reason this
matters for some policies — see [Why cyclic episodes matter](#why-cyclic-episodes-matter-technical-detail)
at the end of this guide.

---

## Before you record

Fix these first. More episodes will not compensate for a broken setup.

| Check                       | Why                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| Rig and cameras bolted down | Any camera movement between sessions invalidates every earlier episode.                       |
| Object/background contrast  | If you can barely see the object in the camera feed, neither can the policy.                  |
| Lighting                    | Diffuse and consistent. No moving shadows. Lighting matters more than resolution.             |
| Camera coverage             | Can _you_ do the task from the camera views alone? Check every target position, not just one. |
| Two cameras                 | 1 fixed overhead/front + 1 wrist. Multi-view consistently outperforms single-view.            |
| Practice runs               | Do 5-10 unrecorded runs first. Hesitant, inconsistent demos teach hesitation.                 |

---

## Staged recording: add diversity one axis at a time

The most common mistake is jumping straight to the full task. If you vary object
position **and** distractor count **and** target position across 50 episodes,
every axis ends up with a handful of samples and the policy generalises to none
of them.

| Stage | Episodes | What changes                                                          | Purpose                                     |
| ----- | -------- | --------------------------------------------------------------------- | ------------------------------------------- |
| **1** | ~50      | **One** object, **one** fixed target, position varied in a small zone | Constrained baseline. Proves the rig works. |
| **2** | ~30      | Add distractor objects, target still fixed                            | Teaches object selection.                   |
| **3** | ~40      | 3-4 target positions, distractors present                             | Teaches spatial generalisation.             |

**Train after Stage 1 before recording Stage 2.** A quick ACT run tells you
whether the problem is your rig or your model, in minutes rather than after
recording 70 more episodes on a broken setup. See
[How long should I train?](#how-long-should-i-train) below.

### Sizing

Target roughly **45,000+ frames** total (episode count x average episode
length x fps), which is a solid reference point for a first task. Episode
_count_ and frame _count_ are different budgets — short cyclic episodes need
more episodes to reach the same frame total.

| Setting          | Recommendation                                             |
| ---------------- | ---------------------------------------------------------- |
| Episodes         | 50 to start, scale to 100-300 after the first training run |
| Episode length   | Whatever the cyclic task takes; 15-45 s is typical         |
| FPS              | 30                                                         |
| Cameras          | 2 (1 fixed + 1 wrist)                                      |
| Task description | Short, specific, action-phrased sentence                   |

---

## Consistency

Within and across episodes, keep the same:

- grasp pose and approach vector
- movement timing and speed
- **object selection rule** when multiple candidates are present

That last one matters more than people expect. Five identical objects plus a
generic task description makes the target genuinely ambiguous in the data — the
policy has to infer a selection rule from a handful of examples. Pick one rule
(always nearest to the base, always leftmost, ...) and stick to it, or make the
task description itself disambiguating.

Optimise for speed only **after** the strategy is dialled in. Never trade
quality for speed.

---

## Task descriptions

Language-conditioned policies (Pi0.5, SmolVLA) read the task string you enter
when creating the dataset or confirming the task before an episode. Make it
specific:

```text
BAD   "pick up the object and put it in the box"
GOOD  "pick up a brown cube and put it in the black box"
```

Costs nothing to write, gives the language encoder something to work with.

---

## Recording and reviewing in Physical AI Studio

Full click-by-click walkthrough with screenshots:
[Recording Datasets](../../../../application/docs/05-recording-datasets.md).
Summary of the loop:

1. Reset your physical scene.
2. Enter or confirm the task.
3. Click **Start episode**.
4. Perform the demonstration — cyclic, per the rule above.
5. Click **Accept** or **Discard**.

Use **Discard** freely. A discarded episode costs seconds; a bad episode baked
into the dataset costs a training run.

After recording, use the built-in episode viewer to review each episode —
replay the video and check the joint-state graph for anything that looks wrong
(jumps, stalls, an object out of frame). Remove any episode that doesn't look
right before you train.

---

## How long should I train?

This differs by policy, mainly because of whether the policy starts from a
pretrained backbone.

| Policy      | Pretrained backbone?      | Typical epochs                                                     | Notes                                                                                                                 |
| ----------- | ------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| ACT         | No — trained from scratch | No pretrained schedule; use validation loss to decide when to stop | Fast per step, good first baseline.                                                                                   |
| Diffusion   | No — trained from scratch | Longer than ACT                                                    | Benefits from more training than ACT on the same data.                                                                |
| SmolVLA     | Yes — pretrained VLM      | 5-10 epochs                                                        | Unfreezing the vision encoder often improves results on specialized tasks, at the cost of more VRAM and slower steps. |
| Pi0 / Pi0.5 | Yes — pretrained VLA      | 5-10 epochs                                                        | Memory-heavy; a small dataset needs far fewer epochs than a large pretraining-scale one.                              |

For any pretrained policy, more epochs is not automatically better on a small,
task-specific dataset: watch `val/loss`, not just the training loss, and stop
once it plateaus rather than running a fixed large step count out of habit.

### Epoch/step conversion

Duration is easiest to reason about in epochs (one full pass over your
dataset), independent of batch size:

```text
steps_per_epoch = ceil(total_frames / batch_size)
total_steps      = epochs * steps_per_epoch
```

### Training with the `physicalai` CLI

```bash
# Train with a policy config, overriding epochs directly
physicalai fit --config configs/physicalai/pi05.yaml --trainer.max_epochs 10

# Or specify a raw step budget instead
physicalai fit --config configs/physicalai/act.yaml --trainer.max_steps 40000
```

See [CLI Training](../training/cli.md) for the full set of override flags
(batch size, precision, devices, callbacks).

### Training from the Studio GUI

The Models page training settings ask for a **training steps** value (see
[Training Policies](../../../../application/docs/06-training-policies.md)),
not epochs directly. Use the conversion above with your dataset's total frame
count and chosen batch size to translate a target epoch count into that field.

---

## Do not merge incompatible datasets

If you re-record because of a design change (e.g. adding the cyclic return),
**create a new dataset** rather than adding the new episodes to the old one.
Mixing non-cyclic episodes into a cyclic dataset reintroduces the "stop moving
at the target" behaviour you were trying to remove. Keep the old dataset around
as a baseline for comparison instead of deleting it.

---

## Troubleshooting

| Symptom                         | Likely cause                                                                              |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| Policy ignores the object       | Camera framing, contrast, or lighting — not the model                                     |
| Policy flaps / oscillates       | Inconsistent demos, or needs more training                                                |
| Fails at one specific stage     | Record 10-20 more episodes targeting that stage                                           |
| Works at trained positions only | Under-sampled diversity — record more episodes per position                               |
| Completes once then freezes     | Non-cyclic episodes (see [above](#the-single-most-important-rule-record-cyclic-episodes)) |
| Picks a random object each time | No consistent selection rule in the demonstrations                                        |

---

## Why cyclic episodes matter (technical detail)

Chunked-action policies (ACT, Pi0, Pi0.5, SmolVLA, ...) request `chunk_size`
future actions for every recorded frame. Near the end of an episode those
frames don't exist, so the underlying dataset library clamps the query to the
last frame and repeats the final action instead:

```python
# lerobot/datasets/dataset_reader.py
key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
```

This is flagged with an `action_is_pad` marker, but **not every policy masks it
out of the training loss**:

| Policy  | Masks padded actions out of the loss? |
| ------- | ------------------------------------- |
| ACT     | Yes                                   |
| SmolVLA | Yes (upstream)                        |
| Pi0     | No                                    |
| Pi0.5   | No                                    |

For Pi0/Pi0.5, the repeated final action is trained on as if it were real
supervision. With a 50-step action chunk and short episodes, this can end up
being a meaningful fraction of all action targets.

- Non-cyclic episodes -> the policy is trained to freeze **at the target**. Bad.
- Cyclic episodes -> the policy is trained to hold **at home**. Safe, correct,
  and exactly what you want between repetitions.

This is the underlying reason cyclic episodes are a requirement rather than a
preference for these policies — though it's good practice for every policy
regardless, since it's also what makes chained, repeated execution possible at
all.
