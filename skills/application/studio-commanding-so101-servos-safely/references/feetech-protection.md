# Feetech STS3215 protection and passive diagnostics

## Registers

Addresses and names follow LeRobot's `STS_SMS_SERIES_CONTROL_TABLE` (`lerobot/motors/feetech/tables.py`). Defaults and meanings follow the Feetech STS memory table; the vendor manual is linked from `STS3215Addr` in physicalai's `src/physicalai/robot/so101/constants.py`. Read the values from your own arm instead of assuming the defaults.

| Addr   | Name                     | Bytes | Default  | Meaning                                                                                                                                                                                                                                                                   |
| ------ | ------------------------ | ----- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9 / 11 | Min / Max_Position_Limit | 2     | 0 / 4095 | Calibrated range, stored in EEPROM by the calibration sweep. Values of 0 / 4095 mean the arm was never swept.                                                                                                                                                             |
| 16     | Max_Torque_Limit         | 2     | 1000     | 0.1% units. physicalai sets 500 on the gripper only.                                                                                                                                                                                                                      |
| 28     | Protection_Current       | 2     | —        | physicalai sets 250 on the gripper only.                                                                                                                                                                                                                                  |
| 31     | Homing_Offset            | 2     | —        | Sign-magnitude with the sign in bit 11, not two's complement.                                                                                                                                                                                                             |
| 34     | Protective_Torque        | 1     | 20       | % torque output after overload protection trips.                                                                                                                                                                                                                          |
| 35     | Protection_Time          | 1     | 200      | 10 ms units (200 = 2 s): how long load must exceed the overload threshold before tripping.                                                                                                                                                                                |
| 36     | Overload_Torque          | 1     | 80       | % load threshold that starts the protection timer. physicalai and LeRobot set 25 on the gripper. LeRobot's code comment describes this as "torque when overloaded", which is register 34's job per the memory table. Verify on hardware before relying on either reading. |
| 40     | Torque_Enable            | 1     | 0        | 1 = on.                                                                                                                                                                                                                                                                   |
| 42     | Goal_Position            | 2     | —        | Last commanded target, in raw ticks.                                                                                                                                                                                                                                      |
| 56     | Present_Position         | 2     | —        | Raw ticks.                                                                                                                                                                                                                                                                |
| 60     | Present_Load             | 2     | —        | Magnitude in 0.1% of max torque; bit 10 is direction (verify on hardware).                                                                                                                                                                                                |
| 62     | Present_Voltage          | 1     | —        | 0.1 V units.                                                                                                                                                                                                                                                              |
| 63     | Present_Temperature      | 1     | —        | °C, case sensor.                                                                                                                                                                                                                                                          |
| 65     | Status                   | 1     | —        | Error bits. Bit 5 (32) is overload; it matches `err=32` in physicalai's torque-write warnings.                                                                                                                                                                            |

## How streaming defeats protection

Feetech documents overload protection as released by the next command. With a target the joint cannot reach, the sequence is:

1. Load exceeds the threshold, and after Protection Time the output drops to Protective Torque.
2. The next streamed command, at most 33 ms later at 30 Hz, releases the protection and the servo pushes at full torque again.
3. Whether a new command also restarts the timer before the first trip is not documented. Either way, the joint spends nearly all its time at stall torque.

When the command stream stops, the protection behaves as designed: after Protection Time the servo backs off to Protective Torque and stays there.

Latched status bits can make later writes fail with `err=32`. To clear them, power-cycle the servo bus itself; replugging USB is not enough.

## Passive read script

Run this with nothing else holding the port. Studio's robot owner process keeps the serial port open even when idle, so stop the backend first and check with `lsof /dev/ttyACM0`.

The script only reads registers. It never calls `connect()`, writes nothing, and does not change torque. It needs `feetech-servo-sdk`, which `physicalai[so101]` installs.

```python
#!/usr/bin/env python3
"""Passive SO-101 servo snapshot: python servo_snapshot.py /dev/ttyACM0"""
import sys
from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex", 4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
REGS = [("torque", 40, 1), ("goal", 42, 2), ("present", 56, 2), ("load", 60, 2), ("volt", 62, 1),
        ("temp", 63, 1), ("status", 65, 1), ("ovl_thr%", 36, 1), ("prot_t", 35, 1), ("prot_tq%", 34, 1),
        ("min", 9, 2), ("max", 11, 2)]

port = PortHandler(PORT)
if not (port.openPort() and port.setBaudRate(1_000_000)):
    sys.exit(f"cannot open {PORT} (is Studio's backend or another process holding it?)")
ph = PacketHandler(0)
print("joint          " + " ".join(f"{n:>8}" for n, _, _ in REGS) + "  gap")
try:
    for sid, name in NAMES.items():
        vals = {}
        for reg, addr, size in REGS:
            read = ph.read1ByteTxRx if size == 1 else ph.read2ByteTxRx
            v, res, _err = read(port, sid, addr)
            vals[reg] = v if res == COMM_SUCCESS else None
        if vals["load"] is not None:
            vals["load"] = (vals["load"] & 0x3FF) / 10.0      # % of max torque, direction bit dropped
        if vals["volt"] is not None:
            vals["volt"] = vals["volt"] / 10.0
        gap = (abs(vals["goal"] - vals["present"])
               if None not in (vals["goal"], vals["present"]) else None)
        print(f"{name:14} " + " ".join(f"{'-' if vals[r] is None else vals[r]:>8}" for r, _, _ in REGS)
              + f"  {'-' if gap is None else gap}")
finally:
    port.closePort()
```

At rest, a healthy joint shows a `gap` (goal minus present, in ticks) of a few ticks, low load, and `status` 0. A joint with a large gap and steady load is pushing against something. A joint with a small gap but high load is pushing inside the stall threshold. If `min`/`max` read 0/4095, the arm has never been swept.
