# SO-101 Example Scripts

Quick hardware verification scripts. No extra dependencies beyond `physicalai[so101]`.

## read_joints

Read and display live joint positions. Torque is off — move the arm by hand to verify readings.

```bash
python examples/so101/read_joints.py --port /dev/ttyUSB0
```

## move_joints

Move each joint by a small offset in radians (default ±0.08 rad) one at a time to verify actuation and wiring. Torque is released when done. A calibration file is required.

```bash
python examples/so101/move_joints.py --port /dev/ttyUSB0 --calibration path/to/calibration.json
```

Use `--offset 0.15` (radians) for larger movements or `--delay 1.0` for more time to observe.

## Finding your port

```bash
ls /dev/cu.usb*          # macOS
ls /dev/ttyUSB*          # Linux
```

> **macOS users:** Always use `/dev/cu.*` instead of `/dev/tty.*`. The `tty`
> variant may hang on open (uninterruptible).
