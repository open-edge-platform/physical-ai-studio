# SO-101 manual alignment report — 14 November 2026

Both arms use `q_cad = cad_at_zero_rad + cad_sign * q_encoder_rad`. Encoder radians use 4096 periods per turn and the midpoint of each recorded interval. Present_Position already contains the hardware homing offset.

Both alignment flags are confirmed for **manual visual matching**. Physical pose accuracy has not been measured. The leader elbow sign was explicitly confirmed by the user despite shoulder co-movement; strict-isolation failures remain in the evidence folder.

## Recorded mapping

| Arm | Joint | Physical direction for increasing value | CAD sign | CAD angle at encoder zero (rad) |
|---|---|---|---:|---:|
| follower | shoulder_pan | clockwise viewed from above | +1 | -0.308330138 |
| follower | shoulder_lift | lower upper arm away from upright toward the extended pose | +1 | +1.582717062 |
| follower | elbow_flex | lower forearm and gripper at the elbow | +1 | -1.704729894 |
| follower | wrist_flex | tilt gripper tip downward, forearm fixed | +1 | +0.047655670 |
| follower | wrist_roll | counter-clockwise viewed from behind gripper toward fingertips | +1 | -0.227796147 |
| follower | gripper | Increasing raw ticks / calibrated range value opens the jaw | separate opening coordinate | — |
| leader | shoulder_pan | clockwise viewed from above | +1 | -0.299126254 |
| leader | shoulder_lift | lower upper arm away from upright toward the extended pose | +1 | +1.566610264 |
| leader | elbow_flex | lower forearm at elbow | +1 | -1.650273576 |
| leader | wrist_flex | tilt whole gripper assembly downward, forearm fixed | +1 | +0.013908092 |
| leader | wrist_roll | counter-clockwise viewed from behind gripper toward fingertips | +1 | -3.082534393 |
| leader | gripper | Increasing raw ticks / calibrated range value opens the jaw | separate opening coordinate | — |

## Reference and evidence

The target is [upright-cad-target.json](reference_poses/upright-cad-target.json): CAD angles `[0, 93.1, -89.5, 1.5, 90]` degrees. Each arm was manually placed to match it by eye, then read with torque off. No automatic movement or register writes were made. The colored moving jaw identifies wrist-roll orientation. The earlier 180-degree display correction was applied; the original observation is preserved separately.

- [Follower capture](measurements/manual-direction-session/follower-upright-reference-01.json)
- [Leader capture](measurements/manual-direction-session/leader-upright-reference-01.json)
- Individual accepted/rejected direction observations and raw before/after snapshots are retained in `measurements/manual-direction-session/`.
- The original LeRobot calibration files and their `calibration_sha256` bindings are unchanged.
- All five arm signs are +1 on both arms. Gripper opening is separate from the 6x5 arm Jacobian.

## Model checks

The generated [follower URDF](../models/so101-follower-calibrated.urdf) and [leader URDF](../models/so101-leader-calibrated.urdf) have five bounded arm joints. CAD hub and gripper joints remain fixed for arm FK/IK. Position intervals come from each arm’s own calibration. They are kinematics-only models with massless inertial placeholders; effort/velocity values are unbounded because those quantities were not measured.

- Each actual aligned model passed Frax FK/Jacobian comparisons and finite-difference Jacobian checks at 10 configurations.
- Generated URDF position limits match the per-arm calibration.
- Both calibration/kinematics test runs passed: 8 passed, 1 optional backend skipped per environment; LeRobot and Frax were each checked in the interpreter where installed.
- Numerical agreement is software consistency, not a physical accuracy estimate.
- Tool/TCP and camera mounting transforms are not measured by this procedure. The default FK endpoint remains the Body3 wrist-roll frame.
- No open-loop physical motion test has been executed by these tools.

See [follower validation](../models/so101-follower-validation.json) and [leader validation](../models/so101-leader-validation.json).
