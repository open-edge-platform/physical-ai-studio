"""Offline LeRobot tick/radian conversion; never opens a servo or writes registers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "calibration").is_dir() else Path(__file__).resolve().parents[1]
MOTOR_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
PROFILES = {
    "follower": ROOT / "calibration/robots/so_follower/follower.json",
    "leader": ROOT / "calibration/teleoperators/so_leader/leader.json",
}
ALIGNMENT_PATH = ROOT / "calibration/kinematics_alignment.json"
# Encoder geometry uses 4096 periods/revolution. LeRobot 0.7 DEGREES uses
# resolution - 1 (4095); bridge those two conventions explicitly below.
RADIANS_PER_TICK = 2 * np.pi / 4096


def vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return value


class ServoCalibration:
    """Present_Position ticks after the servo's stored homing offset is applied.

    Applying homing_offset again here would count the hardware adjustment twice.
    The gripper is a separate 0..100 opening coordinate, not an arm angle.
    """

    def __init__(self, arm, path=None):
        if arm not in PROFILES:
            raise ValueError("arm must be follower or leader")
        self.arm = arm
        self.path = Path(path) if path is not None else PROFILES[arm]
        self.records = json.loads(self.path.read_text())
        self.fingerprint = hashlib.sha256(
            json.dumps(self.records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for i, name in enumerate((*MOTOR_NAMES, "gripper"), 1):
            entry = self.records[name]
            if entry["id"] != i or not 0 <= entry["range_min"] < entry["range_max"] <= 4095:
                raise ValueError(f"Invalid calibration for {name}")
        self.tick_min = np.array([self.records[n]["range_min"] for n in MOTOR_NAMES], float)
        self.tick_max = np.array([self.records[n]["range_max"] for n in MOTOR_NAMES], float)
        self.zero_ticks = (self.tick_min + self.tick_max) / 2
        self.lower = (self.tick_min - self.zero_ticks) * RADIANS_PER_TICK
        self.upper = (self.tick_max - self.zero_ticks) * RADIANS_PER_TICK

    def ticks_to_radians(self, ticks):
        return (vector(ticks, 5, "ticks") - self.zero_ticks) * RADIANS_PER_TICK

    def radians_to_ticks(self, q):
        """Nearest encoder tick; no clamping or hidden limit enforcement."""
        return np.rint(vector(q, 5, "q") / RADIANS_PER_TICK + self.zero_ticks).astype(int)

    def lerobot_degrees_to_radians(self, degrees):
        return np.deg2rad(vector(degrees, 5, "degrees")) * (4095 / 4096)

    def radians_to_lerobot_degrees(self, q):
        return np.rad2deg(vector(q, 5, "q")) * (4096 / 4095)

    def validate_limits(self, q):
        q = vector(q, 5, "q")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("Arm coordinates exceed the recorded calibration interval")
        return q

    def gripper_ticks_to_percent(self, tick):
        value = float(tick)
        if not np.isfinite(value):
            raise ValueError("gripper tick must be finite")
        entry = self.records["gripper"]
        return (value - entry["range_min"]) * 100 / (entry["range_max"] - entry["range_min"])

    def alignment(self, path=ALIGNMENT_PATH):
        data = json.loads(Path(path).read_text())
        if data.get("version") != 1:
            raise ValueError("Unsupported kinematics alignment version")
        profile = data["arms"][self.arm]
        if profile.get("confirmed") is not True:
            raise ValueError(f"{self.arm}: CAD/servo zero and direction alignment is not confirmed")
        if profile.get("calibration_sha256") != self.fingerprint:
            raise ValueError(f"{self.arm}: alignment does not match the current calibration")
        # Null means unknown: NumPy would turn null into NaN, rejected by vector().
        offset = vector([profile["cad_at_zero_rad"][n] for n in MOTOR_NAMES], 5, "cad_at_zero_rad")
        signs = vector([profile["cad_sign"][n] for n in MOTOR_NAMES], 5, "cad_sign")
        if not np.isin(signs, [-1, 1]).all():
            raise ValueError("Each cad_sign must be +1 or -1")
        return offset, signs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=PROFILES, default="follower")
    parser.add_argument("--export-urdf", type=Path)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT_PATH)
    args = parser.parse_args()
    cal = ServoCalibration(args.arm)
    print("Coordinates: encoder radians (4096 periods/turn); input ticks already homed.")
    for name, zero, lower, upper in zip(MOTOR_NAMES, cal.zero_ticks, cal.lower, cal.upper):
        print(f"{name:15s} zero_ticks={zero:7.1f} lower_rad={lower:+.9f} upper_rad={upper:+.9f}")
    if args.export_urdf:
        from so101_kinematics import CalibratedSO101Kinematics

        model = CalibratedSO101Kinematics(args.arm, alignment_path=args.alignment)
        model.export_arm_urdf(args.export_urdf)
        print(args.export_urdf)


if __name__ == "__main__":
    main()
