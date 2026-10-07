# Incidents

Both programs were written with AI coding assistants, both looked reasonable on review, and both used the same 30 Hz position-streaming idiom as LeRobot replay and `RobotRuntime`.

## 1. Diagnostic hold loop: elbow_flex gearbox stripped

**Context.** A bench SO-101 was being checked against a recorded dataset. A replay script sent the episode's recorded actions at the dataset's 30 fps. To freeze the arm at the episode's start pose, the replay loop was reused with a fixed frame:

```python
for k in range(90):                       # 3 s ramp: bounded, target moves -> fine
    r.send_action(cur + (A[0] - cur) * (k + 1) / 90, goal_time=1/30); time.sleep(1/30)
while True:                               # unbounded, constant target -> destroys the servo
    r.send_action(A[HOLD_FRAME], goal_time=1/30); time.sleep(1/30)
```

**Target.** The pose was `[13.4, -98.8, 100.0, 89.0, -22.4, 2.3]`. Elbow_flex at exactly 100.0 is the calibrated mechanical stop, and shoulder_lift at −98.8 was just short of its own.

**Outcome.** The elbow servo stalled against its stop under a command stream that kept releasing its overload protection. With power off, the joint moved only about 10° by hand: a stripped gearbox, and the servo was scrap.

**Would have prevented it:**

- Clamping to ±95.
- Sending the hold target once.
- A tracking-error stall check.
- A timeout.

## 2. Workshop tic-tac-toe app: servo smoked during play

**Context.** A PySide6 game app drove an SO-101 between a park pose, a board-detection pose, and per-square place poses. A code audit found these issues:

- **Shipped poses on the stops.** Park had elbow_flex at 100.0; detection had shoulder_lift at −98.6 and elbow_flex at 99.2. Every human turn ran park → detection → park as two 3 s interpolated moves. All 180 commands put both joints within 2 units of their stops, each command slightly different, so protection was released on every tick. Between turns, the arm held park with the elbow commanded to 100.0.
- **Capture dialog.** A 33 ms `QTimer` mirrored the leader onto the follower with no timeout. Clicking "Capture" frees a hand, so the leader gets set down and slumps toward its stops, and the follower follows it.
- **Captured poses were the leader's position.** A pose captured while the follower was blocked became a target the follower could never reach.
- **No way to stop.** Torque stayed on at park between turns and after exit. Motion ran on the GUI thread with no stop control.

**Outcome.** The servo that smoked was shoulder_pan, as reported by the operator. The leading explanation is that the folded arm caught the base clamp during the roughly 75-unit pan swing between park and detection. The rest of each 3 s move kept streaming new targets at the blocked joint. This is unconfirmed: no contact marks were checked and the swing was not re-tested.

**Field fix shipped:**

- A ±90 clamp.
- A 0.5-unit mirror deadband and a 60 s capture timeout.
- Re-targeting joints that didn't arrive to their measured positions after each move.
- Saving the follower's measured position on capture.
- Clamping the pose file.

The policy-driven pick path was not covered.

**Would have caught the pan case directly:** a per-tick tracking-error stall check, which needs no knowledge of where the clamp is.
