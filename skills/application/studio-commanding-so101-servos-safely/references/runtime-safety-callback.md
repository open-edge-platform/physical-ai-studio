# Safety callback for RobotRuntime

`RobotRuntime` (`src/physicalai/runtime/core.py` in openvinotoolkit/physicalai) streams the policy's actions at `fps`. Out of the box it neither clamps SO-101 actions short of the stops nor watches for stalls; its return-to-initial-state move does check tracking error, but the main loop does not.

The runtime offers two hooks that make it possible to add both checks:

- **`on_action_ready(action=..., step=...)`** may transform the action. An exception raised here ends the run, because a failed safety filter means the action can no longer be trusted.
- **`on_tick(event)`** receives `event.robot_state`, the observation read at the start of that tick. Exceptions raised here are only logged. So detection happens in `on_tick` and the abort happens in `on_action_ready`.

Per tick, the order is: read the observation, `on_action_ready`, send, `on_action_sent`, `on_tick`. The measurement seen in `on_tick` therefore reflects the command sent on the previous tick.

```python
import numpy as np


class JointSafetyCallback:
    """Clamp every SO-101 action off the end stops; end the run if an arm joint stalls."""

    def __init__(self, arm_limit=95.0, stall_err=5.0, stall_speed=0.3, stall_ticks=10):
        self.arm_limit, self.stall_err = arm_limit, stall_err
        self.stall_speed, self.stall_ticks = stall_speed, stall_ticks
        self._prev_meas = None
        self._prev_cmd = None   # the command the latest measurement reflects
        self._last_cmd = None
        self._count = np.zeros(5, dtype=int)

    def on_action_ready(self, *, action, step):
        if self._count.max() >= self.stall_ticks:
            joints = np.flatnonzero(self._count >= self.stall_ticks).tolist()
            raise RuntimeError(f"JointSafetyCallback: arm joint(s) {joints} stalled at step {step}")
        a = np.asarray(action, dtype=np.float32).copy()
        a[:5] = np.clip(a[:5], -self.arm_limit, self.arm_limit)
        a[5] = np.clip(a[5], 0.0, 100.0)   # gripper: clamp only, no stall check
        return a

    def on_action_sent(self, *, action, step):
        self._prev_cmd, self._last_cmd = self._last_cmd, np.asarray(action, dtype=np.float32)

    def on_tick(self, event):
        meas = np.asarray(event.robot_state.joint_positions, dtype=np.float32)
        if self._prev_meas is not None and self._prev_cmd is not None:
            err = np.abs(self._prev_cmd[:5] - meas[:5])
            speed = np.abs(meas[:5] - self._prev_meas[:5])
            self._count = np.where((err > self.stall_err) & (speed < self.stall_speed), self._count + 1, 0)
        self._prev_meas = meas
```

## Wiring

```python
runtime = RobotRuntime(robot=robot, action_source=source, fps=30,
                       cameras=cameras, callbacks=[JointSafetyCallback()])
with runtime:                     # exit -> SO101.disconnect() -> joints re-targeted to measured pose
    runtime.run(duration_s=60)
```

When the callback raises, `run()` shuts down the action source. Exiting the `with` block then disconnects the robot, and a follower's `disconnect()` re-targets every joint to its present position, which releases the push.

Without the `with` block, call `runtime.disconnect()` in a `finally`. If you pass `return_to_initial_state=True`, the runtime's return move has its own tracking-error abort.

For YAML runtime bundles (`physicalai run --config runtime.yaml`), the callback must be importable by class path. Follow how `LowPassFilterCallback` is declared in `src/physicalai/runtime/callbacks/low_pass.py` (`@export_config`), and see the physicalai skill `physicalai-runtime-working-with-config`.

## Tuning and limits

- **Stall timing.** At 30 Hz, `stall_ticks=10` aborts about 0.3–0.4 s after a joint falls `stall_err` behind and stops moving.
- **Tune from a known-good run.** Before tightening the thresholds, log `err` on a known-good policy run. Policies trained on human demonstrations lead the arm, so normal tracking error can be several units during fast motion.
- **The gripper is excluded on purpose.** A policy squeezing a cube is a stall by design.
- **What it cannot catch.** A joint pushing with less than `stall_err` of error goes unnoticed. The clamp margin and the passive at-rest load check cover that case.
