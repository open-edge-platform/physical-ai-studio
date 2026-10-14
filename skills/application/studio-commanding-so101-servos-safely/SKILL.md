---
name: studio-commanding-so101-servos-safely
description: Writes, reviews, and debugs code that commands SO-101 arms (Feetech STS3215 servos) so it cannot burn out a servo, covering end-stop clamping, bounded ramps, stopping the command stream once a target is reached, stall detection, and passive diagnostics. Use whenever code calls physicalai.robot.SO101 send_action, connect, or set_torque; wires RobotRuntime to an SO-101 or adds a RuntimeCallback safety filter; uses scservo_sdk, feetech-servo-sdk, or LeRobot's FeetechMotorsBus; or implements a hold, replay, move-to-pose, pose-capture, or leader-to-follower mirror loop (while True with sleep(1/fps), a 33 ms QTimer). Also use when stored poses or dataset actions sit near +/-100, or when a servo is hot, smoking, buzzing, stalled, or reports err=32.
license: Apache-2.0
---

# Commanding SO-101 Servos Safely

Two STS3215 servos have been destroyed by AI-generated control code that looked correct. The first was a hold loop and the second a leader-to-follower mirror. Each re-sent a position target every 33 ms to a joint that could not reach it. See [`references/incidents.md`](references/incidents.md).

The loop shape itself is the standard idiom. LeRobot's replay paces at the dataset's fps, its teleoperate loop defaults to 60 Hz, and `RobotRuntime` examples use `fps=30`. Streaming is correct while the target moves every tick. It becomes destructive when the target is stationary and unreachable: a calibrated end stop, the table, the base clamp, or the robot's own body. Every new command releases the servo's overload protection, so the stream keeps the joint at stall torque. Nothing in the API distinguishes "tracking a trajectory" from "pushing on a wall"; the calling code has to.

## Facts that shape the code

These come from the driver at `src/physicalai/robot/so101/so101.py` in [openvinotoolkit/physicalai](https://github.com/openvinotoolkit/physicalai) and from the Feetech STS memory table. See [`references/feetech-protection.md`](references/feetech-protection.md) for register details and a passive read script.

- **Overload protection is released by any new command.** It trips when load stays above Overload Torque (register 36, default 80%) for Protection Time (register 35, default 2 s). The servo then drops to Protective Torque (register 34, default 20%) until the next command. A 30 Hz stream re-arms a tripped servo within 33 ms.
- **Thermal cutoff can't save a hard stall.** Commands can't clear it, but it reads case temperature, and windings and plastic gears fail first.
- **±100 is the mechanical stop.** Normalized ±100 (0/100 for the gripper) maps to calibration `range_min`/`range_max`, which were recorded by sweeping each joint to its stop. The driver clips there with no margin.
- **`send_action(action, goal_time=...)` ignores `goal_time`.** A single distant target moves at full acceleration (`_configure_servos` writes 254). Ramping is the caller's job, which is why callers reach for a streaming loop.
- **`connect()` is not passive.** It writes configuration registers with torque off, then enables torque for followers. A follower snaps to its last goal; an arm connected as `role="leader"` goes limp.
- **`disconnect()` on a follower holds position.** It re-targets every joint to where it actually is and leaves torque on, so the arm stays powered after the process exits.
- **Only the gripper gets protection settings.** `_configure_servos` writes torque, current, and overload limits for the gripper alone. The five arm joints run factory defaults.
- **A fresh arm is half-calibrated.** If EEPROM ranges are still 0–4095, ±100 means a full revolution and there are no soft limits. Sweep the arm before any teleop or policy run.

## Rules

1. **Clamp every arm-joint command off the stop.** Use `ARM_LIMIT = 95` only after checking the margin in degrees (see Checks); aim for at least about 5°. Clamp the gripper to 0–100.
2. **Stream only while the target is moving.** Ramp a move over a bounded number of ticks, send the final target once, and stop sending.
3. **Never re-send a constant target in a loop.** To hold a pose, send it once; the servo holds on its own and its protection stays effective.
4. **Watch tracking error whenever you stream.** If an arm joint stays more than a few units from its command while not moving, stop streaming, re-target the stuck joint to its measured position, and surface an error.
5. **Leave the gripper out of stall detection.** Squeezing an object is an intended stall, and the driver already limits gripper torque.
6. **Bound every loop.** Give each one a duration or timeout. A leader-to-follower mirror also needs a deadband, so it stops sending when the leader is still.
7. **Store what the follower actually reached.** Save measured follower positions as poses, never leader or commanded values.
8. **Read passively when diagnosing.** Use raw register reads, not `connect()` (see the reference).
9. **Keep motion stoppable without pulling power.** Run it off the GUI thread and check a stop flag every tick. After stopping, send one release command.

## Workflow

1. **Inventory every command path.** Search the code for `send_action`, sync writes, `QTimer`, `setInterval`, `while True`, and `sleep(1/` patterns. For each loop, record its rate, its exit condition, and whether its target changes every iteration.

   - Done when: every loop that commands the arm is listed, and none is unbounded or sends a constant target.

2. **Clamp all inputs.** Route every command through one `clamp()`, including stored poses, dataset actions, policy outputs, and the teleop mirror. Scan stored poses and datasets for arm values at or beyond `ARM_LIMIT` (see Checks).

   - Done when: no path reaches the bus unclamped, and flagged poses are edited in the file so what is stored matches what the arm does.

3. **Replace jumps and holds with a guarded move.** Use the `guarded_move` below for move-to-pose, park, and "go to start" commands. A hold is the guarded move's single final command, followed by nothing.

   - Done when: no code holds a pose by re-sending it.

4. **Guard streaming loops.** For replay, mirror, and custom policy loops, apply the same tracking-error check per tick. For `RobotRuntime`, add `JointSafetyCallback` from [`references/runtime-safety-callback.md`](references/runtime-safety-callback.md) and run inside `with runtime:`, so an aborted run ends in `SO101.disconnect()`, which holds the measured pose.

   - Done when: a deliberately blocked joint aborts the loop within about half a second. Test it by holding the joint by hand with a hand on the power switch.

5. **Bound and make stoppable.** Add timeouts, a mirror deadband (about 0.5 units), and a stop control.

   - Done when: every loop exits on its own, and the operator can stop motion without cutting power.

6. **Validate on the arm.** Run each stored pose once with a hand on the power switch. Then stop the app, check nothing holds the port (`lsof /dev/ttyACM0`), and run the passive read script with the arm at rest.
   - Done when: every arm joint at rest shows a goal-minus-present gap of only a few ticks, with low load and no overload status bit.
   - If a joint shows small error but high load, it is pushing on something inside the stall threshold. Widen the clamp margin or move the pose.

## Guarded move (scripts and apps)

```python
import time
import numpy as np

ARM, GRIPPER = slice(0, 5), 5   # pan, lift, elbow, wrist_flex, wrist_roll | gripper
ARM_LIMIT = 95.0                # normalized; +/-100 is the calibrated mechanical stop
STALL_ERR = 5.0                 # units of commanded-minus-measured that count as "not arriving"
STALL_SPEED = 0.3               # units per tick below which a joint counts as "not moving"
STALL_TICKS = 10                # consecutive ticks (~0.33 s at 30 Hz) before declaring a stall


class StallError(RuntimeError):
    """An arm joint stopped short of its command; the push has been released."""


def clamp(action):
    a = np.asarray(action, dtype=np.float32).copy()
    a[ARM] = np.clip(a[ARM], -ARM_LIMIT, ARM_LIMIT)
    a[GRIPPER] = np.clip(a[GRIPPER], 0.0, 100.0)
    return a


def _measure(robot):
    return np.asarray(robot.get_observation().joint_positions, dtype=np.float32)


def _release(robot, cmd, meas):
    """Re-target joints that did not arrive to where they are; keep the others and the grip."""
    hold = cmd.copy()
    pinned = np.abs(cmd[ARM] - meas[ARM]) > STALL_ERR
    hold[ARM] = np.where(pinned, meas[ARM], cmd[ARM])
    robot.send_action(hold)


def guarded_move(robot, target, duration_s, fps=30.0):
    """Ramp to target, send one final command, stop. Raise StallError (push released) on a stall."""
    target = clamp(target)
    start = _measure(robot)
    n = max(1, round(duration_s * fps))
    prev = start
    count = np.zeros(5, dtype=int)
    for k in range(1, n + 1):
        t0 = time.perf_counter()
        cmd = clamp(start + (target - start) * (k / n))
        robot.send_action(cmd)
        meas = _measure(robot)
        err, speed = np.abs(cmd[ARM] - meas[ARM]), np.abs(meas[ARM] - prev[ARM])
        prev = meas
        count = np.where((err > STALL_ERR) & (speed < STALL_SPEED), count + 1, 0)
        if count.max() >= STALL_TICKS:
            _release(robot, cmd, meas)
            raise StallError(f"joints {np.flatnonzero(count >= STALL_TICKS).tolist()} stalled: "
                             f"cmd {cmd[ARM].round(1)}, measured {meas[ARM].round(1)}")
        time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))
    time.sleep(0.3)               # let the servos settle; nothing is sent here
    meas = _measure(robot)
    _release(robot, target, meas)  # the single final command
    return meas
```

Tune `STALL_ERR` and `STALL_TICKS` by logging tracking error on a known-good run. Gravity-loaded joints (shoulder_lift, elbow_flex) sit a unit or two below their command at rest, so the threshold must clear that.

Position-only detection cannot see a joint pushing with less than `STALL_ERR` of error, such as a pose just short of a badly calibrated stop. The clamp margin and the at-rest load check in step 6 cover that case.

## Checks

```bash
# Arm values at or past the clamp in any JSON pose file (finds every 6-number list).
python - poses.json <<'EOF'
import json, sys
LIMIT = 95.0
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
def walk(n, p="$"):
    if isinstance(n, dict):
        for k, v in n.items(): yield from walk(v, f"{p}.{k}")
    elif isinstance(n, list):
        if len(n) == 6 and all(isinstance(x, (int, float)) for x in n): yield p, n
        else:
            for i, v in enumerate(n): yield from walk(v, f"{p}[{i}]")
for p, v in walk(json.load(open(sys.argv[1]))):
    hits = [f"{NAMES[i]}={v[i]:.1f}" for i in range(5) if abs(v[i]) >= LIMIT]
    if hits: print(f"{p}: {', '.join(hits)}")
EOF

# Degrees per normalized unit and the margin a clamp leaves (LeRobot-format calibration JSON).
python -c "
import json, sys; L = 95; c = json.load(open(sys.argv[1]))
for k, v in c.items():
    if k == 'gripper': continue
    span = v['range_max'] - v['range_min']; d = span * 360 / 4096 / 200
    bad = '  <-- 0-4095 factory default: sweep this arm first' if (v['range_min'], v['range_max']) == (0, 4095) else ''
    print(f'{k:14} {d:.2f} deg/unit, margin at {L}: {(100 - L) * d:.1f} deg{bad}')
" follower_calibration.json
```

Recorded demonstrations routinely contain ±100. A park pose at episode start and end often has shoulder_lift on its rail. So clamp on replay and on policy output, not only on hand-written poses.

## Other arms and stacks

- **LeRobot.** Its SO follower config has `max_relative_target`, a per-step delta cap that defaults to `None`. That caps step size but is not a stall guard.
- **Trossen WidowX AI.** The physicalai driver caps per-step deltas (`src/physicalai/robot/trossen/widowxai.py`). The streaming and stall rules above still apply to any position-controlled arm.
- **Upstream precedent.** `RobotRuntime._return_to_initial_state` in `src/physicalai/runtime/core.py` already aborts its return move on tracking error and holds the measured pose. Its main control loop does not; `JointSafetyCallback` adds that.
