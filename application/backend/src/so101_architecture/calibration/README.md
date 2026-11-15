# LeRobot calibration of the two SO-101 arms

These files are the LeRobot 0.7.0 calibration of the two SO-101 arms on the Intel server. They were made on 2026-11-13 with `lerobot-calibrate`.

The separate physical/CAD alignment was captured on 2026-11-14 through manual
visual reference matching. Both arms' signs and zero offsets are recorded in
`kinematics_alignment.json`; the original LeRobot files below are unchanged.
See [ALIGNMENT_REPORT.md](ALIGNMENT_REPORT.md) for the mapping table, evidence,
generated URDFs, checks, and measurement limitations.

| File | Arm | LeRobot settings |
|---|---|---|
| `robots/so_follower/follower.json` | Follower (12 V) | `--robot.type=so101_follower --robot.id=follower` |
| `teleoperators/so_leader/leader.json` | Leader (5 V) | `--teleop.type=so101_leader --teleop.id=leader` |

The folders have the same layout as the LeRobot calibration folder. On the server, LeRobot reads the files from `~/.cache/huggingface/lerobot/calibration/`.

## Use these files

Do one of these:

- Copy the files into `~/.cache/huggingface/lerobot/calibration/` on the server. Keep the same subfolders.
- Set `HF_LEROBOT_CALIBRATION` to this folder before you run LeRobot:

  ```bash
  export HF_LEROBOT_CALIBRATION=/path/to/repo/calibration
  ```

The `id` value in the LeRobot command (`follower`, `leader`) must be the same as the file name.

## Servo IDs of the leader arm

On 2026-11-13 the servo IDs of the leader arm were in reverse order (ID 1 was the gripper). They were changed in the servo memory to the order that LeRobot uses:

| Joint | ID |
|---|---|
| `shoulder_pan` | 1 |
| `shoulder_lift` | 2 |
| `elbow_flex` | 3 |
| `wrist_flex` | 4 |
| `wrist_roll` | 5 |
| `gripper` | 6 |

`leader.json` is correct only for this ID order. If a servo is replaced or the IDs change, calibrate the arm again.

## Values

Values are raw servo ticks (0–4095, 4096 ticks = 360°). LeRobot always gives `wrist_roll` the full range (0–4095).

For model coordinates, the installed LeRobot 0.7 DEGREES conversion uses
`(Present_Position - range_midpoint) * 360 / 4095`. This software convention is
different from the encoder's 4096 periods per turn. Homing offsets are already
applied by the servo when it reports `Present_Position`; do not apply them twice.
These files alone do not locate that midpoint pose in the CAD model. See
[../kinematics.md](../kinematics.md#per-arm-calibration-and-cad-alignment) for the
explicit CAD alignment mapping and per-arm URDF exporter.

| Joint | Follower range | Leader range |
|---|---|---|
| `shoulder_pan` | 708–2830 | 728–2934 |
| `shoulder_lift` | 1195–3542 | 818–3174 |
| `elbow_flex` | 791–2991 | 858–3059 |
| `wrist_flex` | 597–2949 | 972–3252 |
| `wrist_roll` | 0–4095 | 0–4095 |
| `gripper` | 1329–2829 | 1118–2642 |

## Measured follower reference

The snapshot in [reference_poses/follower-arm-measurement.json](reference_poses/follower-arm-measurement.json)
records a non-overlapping closed gripper at raw 1339 (LeRobot opening value about
0.67), and an approximate wrist-roll reference of -90 degrees in the CAD viewer
at raw 3208. See [../kinematics.md](../kinematics.md#recorded-follower-reference)
for the provisional visual fit and the remaining geometric alignment measurements.
