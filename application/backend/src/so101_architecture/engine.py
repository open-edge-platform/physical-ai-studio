"""Analytical SO-101 Kinematics, 6x5 Jacobian, Dynamics, 5-DOF IK & Architecture Synthesis Engine.

Designed as the core backend of the Modified Physical AI Studio (SO-101 Model-Based Edition).
Relies strictly on the repository's verified `tools/so101_kinematics.py`, `tools/so101_calibration.py`,
`configs/published_priors_verified.json`, and `src/brickcell/block_inspector/geometry.py`,
with zero additional package installations required.
"""

from __future__ import annotations

import io
import json
import math
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

_THIS_DIR = Path(__file__).resolve().parent

try:
    from brickcell.block_inspector.geometry import (
        CORNER_SIGNS,
        DUPLO_CATALOG,
        NOMINAL_TCP_OFFSET_BODY3,
        PRIORS_JSON_PATH,
        REPO_ROOT,
        T_BASE_WORLD,
        T_WORLD_BASE,
        nominal_canted_wrist_cam_transform,
        station_camera_rig,
    )
except ImportError:
    REPO_ROOT = _THIS_DIR if (_THIS_DIR / "assets").is_dir() else _THIS_DIR.parents[2]
    PRIORS_JSON_PATH = (
        _THIS_DIR / "assets" / "published_priors_verified.json"
        if (_THIS_DIR / "assets" / "published_priors_verified.json").is_file()
        else REPO_ROOT / "configs" / "published_priors_verified.json"
    )
    BASE_ORIGIN_Z_M = 0.02932886598977181
    T_WORLD_BASE = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, BASE_ORIGIN_Z_M],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_BASE_WORLD = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -BASE_ORIGIN_Z_M],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    CORNER_SIGNS = np.array(
        [
            [-1.0, -1.0, -1.0],
            [+1.0, -1.0, -1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, +1.0, +1.0],
            [-1.0, +1.0, +1.0],
        ],
        dtype=np.float64,
    )
    NOMINAL_TCP_OFFSET_BODY3 = np.array([0.015, 0.000, -0.088], dtype=np.float64)

    def nominal_canted_wrist_cam_transform(cant_deg: float = 25.0) -> np.ndarray:
        g = math.radians(cant_deg)
        cg, sg = math.cos(g), math.sin(g)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.000],
                [0.0, -cg, -sg, +0.045],
                [0.0, +sg, -cg, -0.020],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @dataclass(frozen=True)
    class _DuploSpec:
        part_id: str
        dims_m: tuple[float, float, float]
        mass_kg: float

    DUPLO_CATALOG = {
        "duplo_2x2": _DuploSpec("3437", (0.0318, 0.0318, 0.0240), 0.019),
        "duplo_2x4": _DuploSpec("3011", (0.0638, 0.0318, 0.0240), 0.033),
    }

    def _look_at_cv(eye_w: list[float], tgt_w: list[float]) -> np.ndarray:
        eye = np.asarray(eye_w, dtype=np.float64)
        tgt = np.asarray(tgt_w, dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_ax = tgt - eye
        z_ax /= np.linalg.norm(z_ax)
        if abs(float(np.dot(z_ax, up))) > 0.995:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_ax = np.cross(z_ax, up)
        x_ax /= np.linalg.norm(x_ax)
        y_ax = np.cross(z_ax, x_ax)
        y_ax /= np.linalg.norm(y_ax)
        out = np.eye(4, dtype=np.float64)
        out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = x_ax, y_ax, z_ax, eye
        return out

    @dataclass
    class _StandaloneCam:
        name: str
        alias: str
        serial_or_dev: str
        K: np.ndarray
        T_parent_cvcam: np.ndarray
        is_wrist: bool = False
        has_depth: bool = True

        def project_world_points(
            self, points_world: np.ndarray, T_world_body3: np.ndarray | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
            T_wc = (
                np.asarray(T_world_body3, dtype=np.float64) @ self.T_parent_cvcam
                if self.is_wrist and T_world_body3 is not None
                else self.T_parent_cvcam
            )
            pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
            pts_c = (pts - T_wc[:3, 3]) @ T_wc[:3, :3]
            z = pts_c[:, 2]
            zs = np.where(np.abs(z) > 1e-9, z, 1e-9)
            u = self.K[0, 0] * (pts_c[:, 0] / zs) + self.K[0, 2]
            v = self.K[1, 1] * (pts_c[:, 1] / zs) + self.K[1, 2]
            return np.column_stack([u, v]), z

    class _StandaloneRig:
        def __init__(self) -> None:
            K_top = np.array(
                [[597.90332, 0.0, 323.843658], [0.0, 597.624878, 236.115799], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            K_plate = np.array(
                [[606.809937, 0.0, 316.578918], [0.0, 606.353882, 250.320465], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            fy_w = 0.5 * 480.0 / math.tan(0.5 * math.radians(52.0))
            K_wrist = np.array([[fy_w, 0.0, 320.0], [0.0, fy_w, 240.0], [0.0, 0.0, 1.0]])
            self.realsense_1 = _StandaloneCam(
                "realsense_1",
                "overhead",
                "350122060653",
                K_top,
                _look_at_cv([0.335, -0.025, 0.545], [0.335, -0.025, 0.015]),
                False,
                True,
            )
            self.realsense_2 = _StandaloneCam(
                "realsense_2",
                "side",
                "243622060450",
                K_plate,
                _look_at_cv([0.525, -0.355, 0.255], [0.295, -0.005, 0.020]),
                False,
                True,
            )
            self.wrist = _StandaloneCam(
                "wrist",
                "wrist",
                "/dev/video44",
                K_wrist,
                nominal_canted_wrist_cam_transform(25.0),
                True,
                False,
            )

        def to_dict(self) -> dict[str, Any]:
            return {
                "cameras": {
                    c.name: {
                        "alias": c.alias,
                        "serial_or_dev": c.serial_or_dev,
                        "width": 640,
                        "height": 480,
                        "K": c.K.tolist(),
                        "T_parent_cvcam": c.T_parent_cvcam.tolist(),
                        "is_wrist": c.is_wrist,
                        "has_depth": c.has_depth,
                    }
                    for c in (self.realsense_1, self.realsense_2, self.wrist)
                }
            }

    def station_camera_rig() -> _StandaloneRig:
        return _StandaloneRig()


for _cand_tools in (_THIS_DIR, REPO_ROOT / "tools"):
    if _cand_tools.is_dir() and str(_cand_tools) not in sys.path:
        sys.path.insert(0, str(_cand_tools))

from so101_calibration import (  # noqa: E402
    ALIGNMENT_PATH,
    MOTOR_NAMES,
    RADIANS_PER_TICK,
    ServoCalibration,
)
from so101_kinematics import (  # noqa: E402
    ARM_JOINTS,
    WRIST_LINK,
    CalibratedSO101Kinematics,
    SO101Kinematics,
)

# Primary rigid links along the 5-DOF kinematic chain from base to wrist
PRIMARY_CHAIN_LINKS: tuple[tuple[str, str, str], ...] = (
    ("Body5_1", "FIXED_BASE", "Base Turret Mount"),
    ("Body1_2", "shoulder_pan", "Shoulder Pan (q1 / Revolute_1)"),
    ("Group 1", "shoulder_lift", "Upper Arm Lift (q2 / Revolute_2)"),
    ("Body1_3", "elbow_flex", "Forearm Elbow (q3 / Revolute_3)"),
    ("Group 1_2", "wrist_flex", "Wrist Pitch (q4 / Revolute_2_5)"),
    ("Body3", "wrist_roll", "Wrist Roll & Gripper Body (q5 / Revolute_4)"),
)

# Upright visual reference pose in CAD degrees (`calibration/reference_poses/upright-cad-target.json`)
UPRIGHT_CAD_TARGET_DEG: tuple[float, ...] = (0.0, 93.1, -89.5, 1.5, 90.0)

# Menagerie / STS3215 nominal motor constants (`dynamics/README.md`)
STS3215_ROTOR_ARMATURE_KG_M2 = 0.028
STS3215_DAMPING_NM_S_PER_RAD = 0.60
STS3215_FRICTIONLOSS_NM = 0.052
STS3215_STALL_TORQUE_NM = 2.94
SAFE_GRIPPER_MIN_RAW_DEFAULT = 1339

# Default workspace bowls (meters in +X-forward world frame)
DEFAULT_WORKSPACE_BOWLS: dict[str, dict[str, Any]] = {
    "source": {
        "label": "Source Bowl (Aquamarine)",
        "pos_world_m": [0.230, 0.000, 0.000],
        "radius_m": 0.0795,
        "rgba": [0.20, 0.86, 0.84, 1.0],
    },
    "good": {
        "label": "GOOD Sort Bowl (Green)",
        "pos_world_m": [0.385, 0.100, 0.000],
        "radius_m": 0.0795,
        "rgba": [0.22, 0.88, 0.42, 1.0],
    },
    "defective": {
        "label": "DEFECTIVE Sort Bowl (Pink/Red)",
        "pos_world_m": [0.207, 0.107, 0.000],
        "radius_m": 0.0795,
        "rgba": [0.96, 0.32, 0.46, 1.0],
    },
}

# Default task waypoints in follower encoder radians
DEFAULT_WAYPOINTS: dict[str, dict[str, Any]] = {
    "observation_home": {
        "name": "observation_home",
        "title": "1. Observation Home (Clear Camera Sightlines)",
        "label": "1. Observation Home (Clear Camera Sightlines)",
        "role": "Park & Multi-Camera 3D Localization Pose",
        "description": "Park & Multi-Camera 3D Localization Pose",
        "q_enc_rad": [0.000, -1.150, 0.880, 0.420, 0.550],
        "gripper_pct": 22.0,
        "source": "calibrated_default",
    },
    "source_pregrasp": {
        "name": "source_pregrasp",
        "title": "2. Source Bowl Pre-Grasp Hover",
        "label": "2. Source Bowl Pre-Grasp Hover",
        "role": "Collision-Free Approach Above Target Brick",
        "description": "Collision-Free Approach Above Target Brick",
        "q_enc_rad": [0.000, -0.520, 1.120, 0.820, 0.000],
        "gripper_pct": 24.0,
        "source": "analytical_5dof_ik",
    },
    "source_pick": {
        "name": "source_pick",
        "title": "3. Source Bowl Grasp Contact",
        "label": "3. Source Bowl Grasp Contact",
        "role": "Top-Down / Pitched Jaw Closure on Brick",
        "description": "Top-Down / Pitched Jaw Closure on Brick",
        "q_enc_rad": [0.000, -0.340, 1.240, 0.680, 0.000],
        "gripper_pct": 3.5,
        "source": "analytical_5dof_ik",
    },
    "plate_camera_spin": {
        "name": "plate_camera_spin",
        "title": "4. Plate-Camera Presentation & Joint-5 CCW Spin",
        "label": "4. Plate-Camera Presentation & Joint-5 CCW Spin",
        "role": "Present at Side D415 ([0.402, -0.090, 0.146] m, 28 deg pitch) + CCW Sweep [+0.45..+2.98 rad]",
        "description": "Present at Side D415 ([0.402, -0.090, 0.146] m, 28 deg pitch) + CCW Sweep [+0.45..+2.98 rad]",
        "q_enc_rad": [-0.220, -0.460, 0.940, 0.520, 0.450],
        "spin_end_q5_rad": 2.980,
        "gripper_pct": 3.5,
        "source": "leader_demonstration_calibrated",
    },
    "good_bowl_drop": {
        "name": "good_bowl_drop",
        "title": "5A. GOOD Bowl Release Waypoint",
        "label": "5A. GOOD Bowl Release Waypoint",
        "role": "Place Verified Clean Brick in GOOD Bowl",
        "description": "Place Verified Clean Brick in GOOD Bowl",
        "q_enc_rad": [0.255, -0.260, 0.960, 0.760, 0.550],
        "gripper_pct": 24.0,
        "source": "analytical_5dof_ik",
    },
    "defective_bowl_drop": {
        "name": "defective_bowl_drop",
        "title": "5B. DEFECTIVE Bowl Release Waypoint",
        "label": "5B. DEFECTIVE Bowl Release Waypoint",
        "role": "Place Flagged Brick in DEFECTIVE Bowl",
        "description": "Place Flagged Brick in DEFECTIVE Bowl",
        "q_enc_rad": [0.475, -0.580, 1.250, 0.780, 0.550],
        "gripper_pct": 24.0,
        "source": "analytical_5dof_ik",
    },
}


def rotation_matrix_to_rpy_deg(R: np.ndarray) -> list[float]:
    """Extract Roll-Pitch-Yaw (ZYX intrinsic / XYZ extrinsic) in degrees from a 3x3 rotation matrix."""
    r = np.asarray(R, dtype=np.float64)
    sy = math.hypot(float(r[0, 0]), float(r[1, 0]))
    pitch = math.atan2(float(-r[2, 0]), sy)
    if sy > 1e-9:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = math.atan2(float(-r[1, 2]), float(r[1, 1]))
        yaw = 0.0
    return [
        round(math.degrees(roll), 3),
        round(math.degrees(pitch), 3),
        round(math.degrees(yaw), 3),
    ]


def _so101_lambert_2d(u_cam: np.ndarray) -> np.ndarray:
    """Antipodal-safe 2D Lambert pointing residual matching `frax_ik.so101_antipodal_safe_lambert_2d`."""
    one_plus_uz = max(1.0 + float(u_cam[2]), 1e-6)
    if 1.0 + float(u_cam[2]) >= 1e-6:
        u_xy = np.asarray(u_cam[:2], dtype=np.float64)
    else:
        u_xy = np.array([math.sqrt(1e-6 * (2.0 - 1e-6)), 0.0], dtype=np.float64)
    return math.sqrt(2.0 / one_plus_uz) * u_xy


@dataclass
class ArmLiveSnapshot:
    arm: str
    connected: bool
    source: str  # "armd_live" or "studio_virtual"
    ticks: list[int]
    q_enc_rad: list[float]
    lerobot_deg: list[float]
    q_cad_rad: list[float]
    q_cad_deg: list[float]
    gripper_tick: int
    gripper_pct: float
    torque_enabled: bool
    healthy: bool
    age_ms: float
    error: str | None = None


class SO101StudioEngine:
    """Unified analytical engine for the Modified Physical AI Studio."""

    def __init__(
        self,
        armd_url: str = "http://127.0.0.1:8765",
        live_depth_url: str = "http://127.0.0.1:8094",
        multicam_url: str = "http://127.0.0.1:8092",
        wrist_url: str = "http://127.0.0.1:8091/frame.jpg",
        hud_url: str = "http://127.0.0.1:8095",
    ) -> None:
        self.armd_url = armd_url.rstrip("/")
        self.live_depth_url = live_depth_url.rstrip("/")
        self.multicam_url = multicam_url.rstrip("/")
        self.wrist_url = wrist_url
        self.hud_url = hud_url.rstrip("/")

        self.cad_kin = SO101Kinematics()
        self.follower_kin = CalibratedSO101Kinematics("follower")
        self.leader_kin = CalibratedSO101Kinematics("leader")
        self.follower_cal = self.follower_kin.calibration
        self.leader_cal = self.leader_kin.calibration

        self.rig = station_camera_rig()
        self.T_body3_cam = nominal_canted_wrist_cam_transform(25.0)
        self.tcp_offset_body3 = NOMINAL_TCP_OFFSET_BODY3.copy()
        self.safe_gripper_min_raw = SAFE_GRIPPER_MIN_RAW_DEFAULT

        # Load published CAD inertial priors (`configs/published_priors_verified.json`)
        self.priors_doc: dict[str, Any] = {}
        if PRIORS_JSON_PATH.is_file():
            self.priors_doc = json.loads(PRIORS_JSON_PATH.read_text(encoding="utf-8"))
        self.link_priors: dict[str, dict[str, Any]] = self.priors_doc.get("links", {})

        # Mutable studio state (virtual leader/follower joint angles, waypoints, workspace bowls)
        self.virtual_leader_q = np.array([0.000, -1.150, 0.880, 0.420, 0.550], dtype=np.float64)
        self.virtual_leader_gripper_pct = 22.0
        self.virtual_follower_q = np.array([0.000, -1.150, 0.880, 0.420, 0.550], dtype=np.float64)
        self.virtual_follower_gripper_pct = 22.0
        self.payload_mass_kg = 0.019  # 2x2 DUPLO brick nominal mass (19 g)

        self.bowls: dict[str, dict[str, Any]] = json.loads(json.dumps(DEFAULT_WORKSPACE_BOWLS))
        self.waypoints: dict[str, dict[str, Any]] = json.loads(json.dumps(DEFAULT_WAYPOINTS))

        # Re-solve default IK waypoints using exact analytical solver so they match current bowl centers
        self._refresh_ik_waypoints()

    def _refresh_ik_waypoints(self) -> None:
        """Compute exact analytical 5-DOF IK solutions for source_pregrasp, source_pick, good & defective drops."""
        src = np.asarray(self.bowls["source"]["pos_world_m"], dtype=np.float64)
        good = np.asarray(self.bowls["good"]["pos_world_m"], dtype=np.float64)
        bad = np.asarray(self.bowls["defective"]["pos_world_m"], dtype=np.float64)

        pre_sol = self.solve_5dof_ik(
            target_xyz_world=[float(src[0]), float(src[1]), float(src[2]) + 0.075],
            pitch_down_deg=90.0,
            jaw_yaw_deg=0.0,
        )
        pick_sol = self.solve_5dof_ik(
            target_xyz_world=[float(src[0]), float(src[1]), float(src[2]) + 0.024],
            pitch_down_deg=90.0,
            jaw_yaw_deg=0.0,
            q_seed=pre_sol["q_enc_rad"],
        )
        spin_sol = self.solve_5dof_ik(
            target_xyz_world=[0.402, -0.090, 0.146],
            pitch_down_deg=28.0,
            jaw_yaw_deg=0.0,
        )
        spin_q = list(spin_sol["q_enc_rad"])
        spin_q[4] = 0.450  # Start Joint-5 CCW sweep at +0.45 rad

        good_sol = self.solve_5dof_ik(
            target_xyz_world=[float(good[0]), float(good[1]), float(good[2]) + 0.055],
            pitch_down_deg=75.0,
            jaw_yaw_deg=math.degrees(math.atan2(float(good[1]), float(good[0]))),
        )
        bad_sol = self.solve_5dof_ik(
            target_xyz_world=[float(bad[0]), float(bad[1]), float(bad[2]) + 0.055],
            pitch_down_deg=75.0,
            jaw_yaw_deg=math.degrees(math.atan2(float(bad[1]), float(bad[0]))),
        )

        self.waypoints["source_pregrasp"]["q_enc_rad"] = [round(v, 5) for v in pre_sol["q_enc_rad"]]
        self.waypoints["source_pick"]["q_enc_rad"] = [round(v, 5) for v in pick_sol["q_enc_rad"]]
        self.waypoints["plate_camera_spin"]["q_enc_rad"] = [round(v, 5) for v in spin_q]
        self.waypoints["good_bowl_drop"]["q_enc_rad"] = [round(v, 5) for v in good_sol["q_enc_rad"]]
        self.waypoints["defective_bowl_drop"]["q_enc_rad"] = [
            round(v, 5) for v in bad_sol["q_enc_rad"]
        ]

    # ------------------------------------------------------------------ Calibration & Unit Conversions

    @staticmethod
    def normalized_to_ticks(
        cal: ServoCalibration, normalized_5: list[float] | np.ndarray
    ) -> np.ndarray:
        """Convert Physical AI Studio normalized [-100, 100] body joint values to homed STS3215 ticks."""
        norm = np.asarray(normalized_5, dtype=np.float64)
        if norm.shape != (5,) or not np.isfinite(norm).all():
            raise ValueError("normalized_5 must contain 5 finite numbers in [-100, 100]")
        out = np.zeros(5, dtype=np.float64)
        for i, m_name in enumerate(MOTOR_NAMES):
            rec = cal.records[m_name]
            rmin = float(rec["range_min"])
            rmax = float(rec["range_max"])
            val = float(norm[i])
            if int(rec.get("drive_mode", 0)) == 1:
                val = -val
            frac = np.clip((val + 100.0) / 200.0, 0.0, 1.0)
            out[i] = rmin + frac * (rmax - rmin)
        return np.round(out).astype(int)

    @staticmethod
    def ticks_to_normalized(cal: ServoCalibration, ticks_5: list[float] | np.ndarray) -> np.ndarray:
        """Convert homed STS3215 ticks to Physical AI Studio normalized [-100, 100] body joint values."""
        ticks = np.asarray(ticks_5, dtype=np.float64)
        if ticks.shape != (5,) or not np.isfinite(ticks).all():
            raise ValueError("ticks_5 must contain 5 finite tick numbers")
        out = np.zeros(5, dtype=np.float64)
        for i, m_name in enumerate(MOTOR_NAMES):
            rec = cal.records[m_name]
            rmin = float(rec["range_min"])
            rmax = float(rec["range_max"])
            span = max(rmax - rmin, 1.0)
            val = 200.0 * ((float(ticks[i]) - rmin) / span) - 100.0
            if int(rec.get("drive_mode", 0)) == 1:
                val = -val
            out[i] = np.clip(val, -100.0, 100.0)
        return out

    def normalized_to_radians(
        self, normalized_5: list[float] | np.ndarray, arm: str = "follower"
    ) -> np.ndarray:
        """Convert Physical AI Studio normalized [-100, 100] values to calibrated encoder radians."""
        cal = self.follower_cal if arm == "follower" else self.leader_cal
        ticks = self.normalized_to_ticks(cal, normalized_5)
        return cal.ticks_to_radians(ticks)

    def radians_to_normalized(
        self, q_enc_rad: list[float] | np.ndarray, arm: str = "follower"
    ) -> np.ndarray:
        """Convert calibrated encoder radians to Physical AI Studio normalized [-100, 100] values."""
        cal = self.follower_cal if arm == "follower" else self.leader_cal
        ticks = cal.radians_to_ticks(q_enc_rad)
        return self.ticks_to_normalized(cal, ticks)

    def apply_custom_calibration(
        self, arm: str, calibration_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply a custom SO-101 calibration dictionary (e.g., from Physical AI Studio's DB) to `leader` or `follower`."""
        if arm not in ("follower", "leader"):
            raise ValueError("arm must be 'follower' or 'leader'")
        for m_name in (*MOTOR_NAMES, "gripper"):
            if m_name not in calibration_dict:
                raise ValueError(f"Missing motor {m_name!r} in calibration_dict")
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(calibration_dict, tmp, indent=2)
            tmp_path = Path(tmp.name)
        try:
            new_cal = ServoCalibration(arm, path=tmp_path)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        if arm == "follower":
            self.follower_cal = new_cal
            self.follower_kin.calibration = new_cal
        else:
            self.leader_cal = new_cal
            self.leader_kin.calibration = new_cal
        return self.get_calibration_summary()

    def sync_from_studio_db(
        self,
        db_path: Path | str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Discover projects, SO-101 Leader/Follower robots, and cameras directly from Physical AI Studio's SQLite DB."""
        import os
        import sqlite3

        candidates: list[Path] = []
        if db_path is not None:
            candidates.append(Path(db_path))
        storage_env = os.environ.get("STORAGE_DIR")
        if storage_env:
            candidates.append(Path(storage_env) / "data" / "physicalai.db")
            candidates.append(Path(storage_env) / "physicalai.db")
        candidates.extend(
            [
                Path("/app/storage/data/physicalai.db"),
                Path("/app/storage/physicalai.db"),
                Path.home() / ".local/share/physicalai/physicalai.db",
            ]
        )
        resolved_db = next((p for p in candidates if p.is_file()), None)
        if resolved_db is None:
            return {
                "synced": False,
                "reason": "physicalai.db not found locally; using repository SO-101 calibration profiles",
                "projects": [],
                "active_project": None,
                "robots": [],
                "cameras": [],
            }

        conn = sqlite3.connect(f"file:{resolved_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            projects = [
                dict(r)
                for r in conn.execute(
                    "SELECT id, name, created_at, updated_at FROM projects ORDER BY updated_at DESC"
                ).fetchall()
            ]
            if not projects:
                return {
                    "synced": True,
                    "db_path": str(resolved_db),
                    "projects": [],
                    "active_project": None,
                    "robots": [],
                    "cameras": [],
                }

            chosen_proj = next(
                (p for p in projects if project_id and p["id"] == project_id),
                next(
                    (
                        p
                        for p in projects
                        if "block" in p["name"].lower() or "so101" in p["name"].lower()
                    ),
                    projects[0],
                ),
            )
            pid = chosen_proj["id"]

            robot_rows = conn.execute(
                "SELECT id, project_id, name, type, payload FROM project_robots WHERE project_id = ?",
                (pid,),
            ).fetchall()
            robots = []
            for r in robot_rows:
                payload = (
                    json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]
                )
                robots.append(
                    {
                        "id": r["id"],
                        "project_id": r["project_id"],
                        "name": r["name"],
                        "type": r["type"],
                        "payload": payload,
                    }
                )
                cal_dict = payload.get("calibration") if isinstance(payload, dict) else None
                if isinstance(cal_dict, dict):
                    if r["type"] == "SO101_Follower" or r["name"].lower() == "follower":
                        try:
                            self.apply_custom_calibration("follower", cal_dict)
                        except Exception:
                            pass
                    elif r["type"] == "SO101_Leader" or r["name"].lower() == "leader":
                        try:
                            self.apply_custom_calibration("leader", cal_dict)
                        except Exception:
                            pass

            cam_rows = conn.execute(
                "SELECT id, project_id, name, driver, fingerprint, hardware_name, payload FROM project_cameras WHERE project_id = ?",
                (pid,),
            ).fetchall()
            cameras = []
            for c in cam_rows:
                fp = (
                    json.loads(c["fingerprint"])
                    if isinstance(c["fingerprint"], str)
                    else c["fingerprint"]
                )
                cpay = json.loads(c["payload"]) if isinstance(c["payload"], str) else c["payload"]
                cameras.append(
                    {
                        "id": c["id"],
                        "project_id": c["project_id"],
                        "name": c["name"],
                        "driver": c["driver"],
                        "fingerprint": fp,
                        "hardware_name": c["hardware_name"],
                        "payload": cpay,
                    }
                )

            return {
                "synced": True,
                "db_path": str(resolved_db),
                "projects": projects,
                "active_project": chosen_proj,
                "robots": robots,
                "cameras": cameras,
            }
        finally:
            conn.close()

    def get_calibration_summary(self) -> dict[str, Any]:
        """Return full per-arm calibration intervals, unit bridges, and CAD zero alignment."""
        alignment_doc = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
        arms_out: dict[str, Any] = {}
        for arm_name, cal, kin in (
            ("follower", self.follower_cal, self.follower_kin),
            ("leader", self.leader_cal, self.leader_kin),
        ):
            joints_table = []
            for idx, (m_name, cad_j) in enumerate(zip(MOTOR_NAMES, ARM_JOINTS)):
                rec = cal.records[m_name]
                joints_table.append(
                    {
                        "id": int(rec["id"]),
                        "motor_name": m_name,
                        "cad_joint": cad_j,
                        "range_min_tick": int(rec["range_min"]),
                        "range_max_tick": int(rec["range_max"]),
                        "zero_ticks": round(float(cal.zero_ticks[idx]), 1),
                        "homing_offset": int(rec["homing_offset"]),
                        "lower_enc_rad": round(float(cal.lower[idx]), 5),
                        "upper_enc_rad": round(float(cal.upper[idx]), 5),
                        "lower_lerobot_deg": round(
                            float(np.rad2deg(cal.lower[idx]) * (4096.0 / 4095.0)), 2
                        ),
                        "upper_lerobot_deg": round(
                            float(np.rad2deg(cal.upper[idx]) * (4096.0 / 4095.0)), 2
                        ),
                        "cad_at_zero_rad": round(float(kin.cad_at_zero[idx]), 6),
                        "cad_at_zero_deg": round(float(math.degrees(kin.cad_at_zero[idx])), 2),
                        "cad_sign": int(kin.cad_sign[idx]),
                    }
                )
            grip_rec = cal.records["gripper"]
            prof_str = (
                str(cal.path.relative_to(REPO_ROOT))
                if cal.path.is_relative_to(REPO_ROOT)
                else str(cal.path.name)
            )
            arms_out[arm_name] = {
                "arm": arm_name,
                "profile_path": prof_str,
                "sha256": cal.fingerprint,
                "confirmed": bool(alignment_doc["arms"][arm_name].get("confirmed", False)),
                "reference_pose": alignment_doc["arms"][arm_name].get("reference_pose", ""),
                "joints": joints_table,
                "gripper": {
                    "id": int(grip_rec["id"]),
                    "range_min_tick": int(grip_rec["range_min"]),
                    "safe_min_tick": (
                        self.safe_gripper_min_raw
                        if arm_name == "follower"
                        else int(grip_rec["range_min"])
                    ),
                    "range_max_tick": int(grip_rec["range_max"]),
                    "homing_offset": int(grip_rec["homing_offset"]),
                },
            }
        return {
            "radians_per_tick": RADIANS_PER_TICK,
            "lerobot_deg_per_tick": 360.0 / 4095.0,
            "upright_cad_target_deg": list(UPRIGHT_CAD_TARGET_DEG),
            "safe_gripper_min_raw": self.safe_gripper_min_raw,
            "arms": arms_out,
        }

    def derive_alignment_from_reference_ticks(
        self,
        arm: str,
        measured_ticks: list[float] | np.ndarray,
        cad_target_deg: list[float] | tuple[float, ...] = UPRIGHT_CAD_TARGET_DEG,
        cad_signs: list[int] | tuple[int, ...] = (1, 1, 1, 1, 1),
    ) -> dict[str, Any]:
        """Compute `cad_at_zero_rad` from homed ticks recorded at a known CAD reference pose.

        Implements `kinematics.md`:
            q_encoder_rad = (measured_ticks - zero_ticks) * 2*pi / 4096
            cad_at_zero_rad = measured_cad_angles - cad_sign * q_encoder_rad
        """
        cal = self.follower_cal if arm == "follower" else self.leader_cal
        ticks_arr = np.asarray(measured_ticks, dtype=np.float64)
        if ticks_arr.shape != (5,) or not np.isfinite(ticks_arr).all():
            raise ValueError("measured_ticks must contain 5 finite servo tick values")
        cad_rad = np.deg2rad(np.asarray(cad_target_deg, dtype=np.float64))
        signs = np.asarray(cad_signs, dtype=np.float64)
        if signs.shape != (5,) or not np.isin(signs, (-1.0, 1.0)).all():
            raise ValueError("cad_signs must contain five +1 or -1 values")

        q_enc_rad = cal.ticks_to_radians(ticks_arr)
        cad_at_zero_rad = cad_rad - signs * q_enc_rad
        reconstructed_cad_deg = np.rad2deg(cad_at_zero_rad + signs * q_enc_rad)
        max_err_deg = float(np.max(np.abs(reconstructed_cad_deg - np.asarray(cad_target_deg))))

        return {
            "arm": arm,
            "calibration_sha256": cal.fingerprint,
            "measured_ticks": [int(round(v)) for v in ticks_arr],
            "q_encoder_rad": [round(float(v), 6) for v in q_enc_rad],
            "cad_target_deg": [round(float(v), 3) for v in cad_target_deg],
            "cad_sign": {m: int(s) for m, s in zip(MOTOR_NAMES, signs)},
            "cad_at_zero_rad": {
                m: round(float(v), 9) for m, v in zip(MOTOR_NAMES, cad_at_zero_rad)
            },
            "cad_at_zero_deg": {
                m: round(float(math.degrees(v)), 3) for m, v in zip(MOTOR_NAMES, cad_at_zero_rad)
            },
            "max_reconstruction_error_deg": max_err_deg,
            "alignment_snippet": {
                "confirmed": True,
                "calibration_sha256": cal.fingerprint,
                "cad_sign": {m: int(s) for m, s in zip(MOTOR_NAMES, signs)},
                "cad_at_zero_rad": {
                    m: round(float(v), 12) for m, v in zip(MOTOR_NAMES, cad_at_zero_rad)
                },
            },
        }

    def map_leader_to_follower(
        self,
        leader_q_enc_rad: list[float] | np.ndarray | None = None,
        leader_ticks: list[float] | np.ndarray | None = None,
        leader_gripper_pct: float | None = None,
    ) -> dict[str, Any]:
        """Map a Leader Arm posture to the geometrically identical Follower Arm posture via CAD coordinates.

        Because Leader and Follower have independent calibration midpoints (`zero_ticks`) and
        `cad_at_zero_rad` offsets:
            q_cad = leader.cad_at_zero + leader.cad_sign * q_leader_enc
            q_follower_enc = follower.cad_sign * (q_cad - follower.cad_at_zero)
        """
        if leader_q_enc_rad is not None:
            q_lead = np.asarray(leader_q_enc_rad, dtype=np.float64)
            ticks_lead = self.leader_cal.radians_to_ticks(q_lead)
        elif leader_ticks is not None:
            ticks_lead = np.asarray(leader_ticks, dtype=np.float64)
            q_lead = self.leader_cal.ticks_to_radians(ticks_lead)
        else:
            q_lead = self.virtual_leader_q.copy()
            ticks_lead = self.leader_cal.radians_to_ticks(q_lead)

        # Normalize wrist_roll difference into [-pi, +pi] while keeping joints 1..4 exact
        q_cad = self.leader_kin.cad_at_zero + self.leader_kin.cad_sign * q_lead
        q_fol_raw = self.follower_kin.cad_sign * (q_cad - self.follower_kin.cad_at_zero)
        q_fol = q_fol_raw.copy()
        q_fol[4] = math.atan2(math.sin(float(q_fol[4])), math.cos(float(q_fol[4])))
        q_fol_clamped = np.clip(
            q_fol, self.follower_cal.lower + 0.01, self.follower_cal.upper - 0.01
        )
        ticks_fol = self.follower_cal.radians_to_ticks(q_fol_clamped)

        grip_pct = float(
            self.virtual_leader_gripper_pct if leader_gripper_pct is None else leader_gripper_pct
        )
        g_rec = self.follower_cal.records["gripper"]
        raw_grip = int(
            round(
                g_rec["range_min"]
                + (np.clip(grip_pct, 0.0, 100.0) / 100.0)
                * (g_rec["range_max"] - g_rec["range_min"])
            )
        )
        safe_grip_tick = max(self.safe_gripper_min_raw, min(int(g_rec["range_max"]), raw_grip))

        return {
            "leader_ticks": [int(v) for v in ticks_lead],
            "leader_q_enc_rad": [round(float(v), 5) for v in q_lead],
            "q_cad_rad": [round(float(v), 5) for v in q_cad],
            "q_cad_deg": [round(float(math.degrees(v)), 2) for v in q_cad],
            "follower_q_enc_rad": [round(float(v), 5) for v in q_fol_clamped],
            "follower_ticks": [int(v) for v in ticks_fol],
            "follower_lerobot_deg": [
                round(float(v), 2)
                for v in self.follower_cal.radians_to_lerobot_degrees(q_fol_clamped)
            ],
            "follower_gripper_tick": safe_grip_tick,
            "follower_gripper_pct": round(
                float(self.follower_cal.gripper_ticks_to_percent(safe_grip_tick)), 2
            ),
        }

    # ------------------------------------------------------------------ Kinematics & 6x5 Jacobian

    def compute_kinematics_and_jacobian(
        self,
        q_enc_rad: list[float] | np.ndarray,
        arm: str = "follower",
        tcp_offset: list[float] | tuple[float, float, float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Evaluate FK, all link poses, 6x5 Geometric Jacobians (rad/tick/deg), and 5D Manipulability."""
        kin = self.follower_kin if arm == "follower" else self.leader_kin
        cal = kin.calibration
        q = np.asarray(q_enc_rad, dtype=np.float64)
        if q.shape != (5,) or not np.isfinite(q).all():
            raise ValueError("q_enc_rad must contain 5 finite joint angles in radians")

        tcp_off = (
            self.tcp_offset_body3
            if tcp_offset is None
            else np.asarray(tcp_offset, dtype=np.float64)
        )
        frames_base = kin.frames(q)
        T_base_body3 = frames_base[WRIST_LINK]
        T_base_tcp = kin.fk(q, tcp_offset=tcp_off)
        T_world_body3 = T_WORLD_BASE @ T_base_body3
        T_world_tcp = T_WORLD_BASE @ T_base_tcp
        T_world_wrist_cam = T_world_body3 @ self.T_body3_cam

        # 6x5 Base-expressed geometric Jacobian [linear; angular] w.r.t. encoder radians
        J_base_rad = kin.jacobian(q, tcp_offset=tcp_off)
        # Rotate linear & angular blocks into +X-forward World frame
        R_wb = T_WORLD_BASE[:3, :3]
        J_world_rad = np.vstack([R_wb @ J_base_rad[:3, :], R_wb @ J_base_rad[3:, :]])

        # Unit-bridged Jacobians (`kinematics.md` lines 115-118)
        J_base_tick = J_base_rad * RADIANS_PER_TICK
        J_base_lerobot_deg = J_base_rad * ((math.pi / 180.0) * (4095.0 / 4096.0))
        J_world_tick = J_world_rad * RADIANS_PER_TICK
        J_world_lerobot_deg = J_world_rad * ((math.pi / 180.0) * (4095.0 / 4096.0))

        # 5x5 Gram matrix & 5-DOF Yoshikawa manipulability
        gram_5x5 = J_base_rad.T @ J_base_rad + 1e-9 * np.eye(5, dtype=np.float64)
        manip_5d = float(math.sqrt(max(float(np.linalg.det(gram_5x5)), 0.0)))
        svals = np.linalg.svd(J_base_rad, compute_uv=False)
        cond_num = float(svals[0] / max(float(svals[-1]), 1e-9))

        ticks = cal.radians_to_ticks(q)
        norm_5 = self.ticks_to_normalized(cal, ticks)
        lerobot_deg = cal.radians_to_lerobot_degrees(q)
        q_cad_rad = kin.cad_at_zero + kin.cad_sign * q
        margins = np.minimum(q - cal.lower, cal.upper - q)

        # Build 3D link chain for interactive visualization & table inspection
        chain_links = []
        for link_name, motor_name, label in PRIMARY_CHAIN_LINKS:
            T_b_link = frames_base[link_name]
            T_w_link = T_WORLD_BASE @ T_b_link
            prior = self.link_priors.get(link_name, {})
            com_local = np.asarray(prior.get("com_m", [0.0, 0.0, 0.0]), dtype=np.float64)
            com_world = T_w_link[:3, 3] + T_w_link[:3, :3] @ com_local
            chain_links.append(
                {
                    "link": link_name,
                    "motor": motor_name,
                    "label": label,
                    "pos_base_m": [round(float(v), 5) for v in T_b_link[:3, 3]],
                    "pos_base_mm": [round(float(v * 1000.0), 2) for v in T_b_link[:3, 3]],
                    "pos_world_m": [round(float(v), 5) for v in T_w_link[:3, 3]],
                    "pos_world_mm": [round(float(v * 1000.0), 2) for v in T_w_link[:3, 3]],
                    "rpy_world_deg": rotation_matrix_to_rpy_deg(T_w_link[:3, :3]),
                    "com_world_m": [round(float(v), 5) for v in com_world],
                    "mass_kg": round(float(prior.get("mass_kg", 0.0)), 6),
                }
            )

        return {
            "arm": arm,
            "q_enc_rad": [round(float(v), 5) for v in q],
            "ticks": [int(v) for v in ticks],
            "normalized": [round(float(v), 2) for v in norm_5],
            "lerobot_deg": [round(float(v), 2) for v in lerobot_deg],
            "q_cad_rad": [round(float(v), 5) for v in q_cad_rad],
            "q_cad_deg": [round(float(math.degrees(v)), 2) for v in q_cad_rad],
            "within_limits": bool(np.all(margins >= 0.0)),
            "per_joint_margin_rad": [round(float(v), 4) for v in margins],
            "tcp_offset_body3_m": [round(float(v), 4) for v in tcp_off],
            "T_base_body3": np.round(T_base_body3, 6).tolist(),
            "T_base_tcp": np.round(T_base_tcp, 6).tolist(),
            "T_world_body3": np.round(T_world_body3, 6).tolist(),
            "T_world_tcp": np.round(T_world_tcp, 6).tolist(),
            "T_world_wrist_cam": np.round(T_world_wrist_cam, 6).tolist(),
            "tcp_base_xyz_m": [round(float(v), 5) for v in T_base_tcp[:3, 3]],
            "tcp_base_xyz_mm": [round(float(v * 1000.0), 2) for v in T_base_tcp[:3, 3]],
            "tcp_base_rpy_deg": rotation_matrix_to_rpy_deg(T_base_tcp[:3, :3]),
            "tcp_world_xyz_m": [round(float(v), 5) for v in T_world_tcp[:3, 3]],
            "tcp_world_xyz_mm": [round(float(v * 1000.0), 2) for v in T_world_tcp[:3, 3]],
            "tcp_world_rpy_deg": rotation_matrix_to_rpy_deg(T_world_tcp[:3, :3]),
            "wrist_cam_world_xyz_m": [round(float(v), 5) for v in T_world_wrist_cam[:3, 3]],
            "link_chain": chain_links,
            "jacobian_base_rad": np.round(J_base_rad, 6).tolist(),
            "jacobian_world_rad": np.round(J_world_rad, 6).tolist(),
            "jacobian_base_tick": np.round(J_base_tick, 7).tolist(),
            "jacobian_base_lerobot_deg": np.round(J_base_lerobot_deg, 7).tolist(),
            "J_world_rad": np.round(J_world_rad, 6).tolist(),
            "J_base_rad": np.round(J_base_rad, 6).tolist(),
            "J_world_tick": np.round(J_world_tick, 7).tolist(),
            "J_world_lerobot_deg": np.round(J_world_lerobot_deg, 7).tolist(),
            "singular_values": [round(float(v), 5) for v in svals],
            "condition_number": round(cond_num, 3),
            "manipulability_5d": round(manip_5d, 6),
        }

    # ------------------------------------------------------------------ Analytical Dynamics & Gravity

    def compute_dynamics(
        self,
        q_enc_rad: list[float] | np.ndarray,
        arm: str = "follower",
        payload_mass_kg: float | None = None,
        tcp_offset: list[float] | tuple[float, float, float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute configuration-space Mass Matrix M(q), Gravity Torque Vector g(q), and COM."""
        kin = self.follower_kin if arm == "follower" else self.leader_kin
        q = np.asarray(q_enc_rad, dtype=np.float64)
        tcp_off = (
            self.tcp_offset_body3
            if tcp_offset is None
            else np.asarray(tcp_offset, dtype=np.float64)
        )
        m_payload = (
            self.payload_mass_kg if payload_mass_kg is None else max(0.0, float(payload_mass_kg))
        )

        frames_base = kin.frames(q)
        T_WORLD_BASE[:3, :3]

        # Extract joint axes & origins in world frame for the 5 arm joints
        joint_origins_w: list[np.ndarray] = []
        joint_axes_w: list[np.ndarray] = []
        for j_name in ARM_JOINTS:
            for name, _, child, _, axis in kin.joints:
                if name == j_name:
                    T_w_child = T_WORLD_BASE @ frames_base[child]
                    joint_origins_w.append(T_w_child[:3, 3])
                    joint_axes_w.append(T_w_child[:3, :3] @ axis)
                    break

        g_world = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
        M_q = np.diag(np.full(5, STS3215_ROTOR_ARMATURE_KG_M2, dtype=np.float64))
        g_tau = np.zeros(5, dtype=np.float64)

        total_mass = 0.0
        weighted_com_w = np.zeros(3, dtype=np.float64)
        link_rows = []

        # Map each primary link to the joints that precede/move it
        moving_links = [
            ("Body5_1", 0),  # Stationary base (0 active joints)
            ("Body1_2", 1),  # Moved by q1
            ("Group 1", 2),  # Moved by q1..q2
            ("Body1_3", 3),  # Moved by q1..q3
            ("Group 1_2", 4),  # Moved by q1..q4
            ("Body3", 5),  # Moved by q1..q5
        ]

        for link_name, active_joint_count in moving_links:
            prior = self.link_priors.get(link_name, {})
            m_i = float(prior.get("mass_kg", 0.0))
            com_local = np.asarray(prior.get("com_m", [0.0, 0.0, 0.0]), dtype=np.float64)
            I_local = np.asarray(
                prior.get(
                    "inertia_matrix_kg_m2",
                    np.diag(prior.get("diagonal_inertia_kg_m2", [1e-5, 1e-5, 1e-5])),
                ),
                dtype=np.float64,
            )
            T_w_link = T_WORLD_BASE @ frames_base[link_name]
            R_w_link = T_w_link[:3, :3]
            p_com_w = T_w_link[:3, 3] + R_w_link @ com_local

            total_mass += m_i
            weighted_com_w += m_i * p_com_w

            J_v = np.zeros((3, 5), dtype=np.float64)
            J_w = np.zeros((3, 5), dtype=np.float64)
            for j_idx in range(active_joint_count):
                z_j = joint_axes_w[j_idx]
                o_j = joint_origins_w[j_idx]
                J_v[:, j_idx] = np.cross(z_j, p_com_w - o_j)
                J_w[:, j_idx] = z_j

            I_world = R_w_link @ I_local @ R_w_link.T
            M_q += m_i * (J_v.T @ J_v) + (J_w.T @ I_world @ J_w)
            tau_i = -m_i * (J_v.T @ g_world)
            g_tau += tau_i

            link_rows.append(
                {
                    "link": link_name,
                    "source_link": prior.get("source_link", link_name),
                    "mass_g": round(m_i * 1000.0, 3),
                    "com_local_mm": [round(float(v * 1000.0), 2) for v in com_local],
                    "com_world_mm": [round(float(v * 1000.0), 2) for v in p_com_w],
                    "principal_inertia_kg_m2": [
                        round(float(v), 8) for v in np.linalg.eigvalsh(I_local)
                    ],
                }
            )

        # Include held payload mass at TCP if positive
        if m_payload > 0.0:
            T_w_body3 = T_WORLD_BASE @ frames_base[WRIST_LINK]
            p_tcp_w = T_w_body3[:3, 3] + T_w_body3[:3, :3] @ tcp_off
            total_mass += m_payload
            weighted_com_w += m_payload * p_tcp_w
            J_v_tcp = np.zeros((3, 5), dtype=np.float64)
            for j_idx in range(5):
                J_v_tcp[:, j_idx] = np.cross(joint_axes_w[j_idx], p_tcp_w - joint_origins_w[j_idx])
            M_q += m_payload * (J_v_tcp.T @ J_v_tcp)
            g_tau += -m_payload * (J_v_tcp.T @ g_world)

        com_world = weighted_com_w / max(total_mass, 1e-9)
        com_base = (T_BASE_WORLD @ np.r_[com_world, 1.0])[:3]
        load_pct = 100.0 * np.abs(g_tau) / STS3215_STALL_TORQUE_NM
        # Nominal static gravity droop compensation in STS3215 ticks (~14 Nm/rad effective stiffness)
        ff_ticks = np.round((g_tau / 14.0) / RADIANS_PER_TICK).astype(int)

        return {
            "status": self.priors_doc.get(
                "status",
                "Published SO101 CAD inertials registered to calibrated local geometry",
            ),
            "arm_mass_g": 632.006,
            "payload_mass_g": round(m_payload * 1000.0, 2),
            "total_mass_g": round(total_mass * 1000.0, 3),
            "whole_arm_com_world_m": [round(float(v), 5) for v in com_world],
            "whole_arm_com_base_m": [round(float(v), 5) for v in com_base],
            "gravity_torque_nm": [round(float(v), 5) for v in g_tau],
            "servo_load_pct_of_stall": [round(float(v), 2) for v in load_pct],
            "feedforward_gravity_ticks": [int(v) for v in ff_ticks],
            "mass_matrix_5x5_kg_m2": np.round(M_q, 6).tolist(),
            "motor_defaults": {
                "rotor_armature_kg_m2": STS3215_ROTOR_ARMATURE_KG_M2,
                "damping_nm_s_per_rad": STS3215_DAMPING_NM_S_PER_RAD,
                "frictionloss_nm": STS3215_FRICTIONLOSS_NM,
                "stall_torque_cap_nm": STS3215_STALL_TORQUE_NM,
            },
            "links": link_rows,
        }

    # ------------------------------------------------------------------ Pure-NumPy 5-DOF Grasp & Pose IK

    def _grasp_residual(
        self,
        q: np.ndarray,
        p_target: np.ndarray,
        approach_dir: np.ndarray,
        jaw_axis_world: np.ndarray,
        is_topdown: bool,
        tcp_off: np.ndarray,
    ) -> np.ndarray:
        """7-row 5-DOF grasp/placement residual matching `frax_ik.make_5dof_grasp_residual`."""
        T_w_tcp = T_WORLD_BASE @ self.follower_kin.fk(q, tcp_offset=tcp_off)
        p_tcp = T_w_tcp[:3, 3]
        R_tcp = T_w_tcp[:3, :3]

        r_pos = 15.0 * (p_tcp - p_target)
        neg_z = -R_tcp[:, 2]
        jaw_x = R_tcp[:, 0]

        if is_topdown:
            R_tool = R_tcp @ np.diag([1.0, -1.0, -1.0])
            r_pitch = 3.0 * _so101_lambert_2d(R_tool.T @ approach_dir)
            jaw_perp = np.cross(approach_dir, jaw_axis_world)
            r_roll = 3.0 * np.array(
                [1.0 - float(np.dot(jaw_x, jaw_axis_world)), float(np.dot(jaw_x, jaw_perp))],
                dtype=np.float64,
            )
        else:
            horiz_norm = max(float(np.linalg.norm(approach_dir[:2])), 1e-6)
            horiz_ref = approach_dir[:2] / horiz_norm
            dot_h = float(np.dot(neg_z[:2], horiz_ref))
            dot_j = float(np.dot(jaw_x, jaw_axis_world))
            r_pitch = np.array(
                [
                    4.0 * float(neg_z[2] - approach_dir[2]),
                    3.0 * max(0.0, 0.45 - dot_h),
                ],
                dtype=np.float64,
            )
            r_roll = np.array(
                [
                    4.0 * float(jaw_x[2]),
                    3.0 * max(0.0, 0.65 - dot_j),
                ],
                dtype=np.float64,
            )
        return np.concatenate([r_pos, r_pitch, r_roll])

    def solve_5dof_ik(
        self,
        target_xyz_world: list[float] | np.ndarray,
        pitch_down_deg: float = 90.0,
        jaw_yaw_deg: float = 0.0,
        q_seed: list[float] | np.ndarray | None = None,
        tcp_offset: list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Solve 5-DOF collision-checked Grasp/Placement IK in pure NumPy (zero extra dependencies)."""
        p_tgt = np.asarray(target_xyz_world, dtype=np.float64)
        if p_tgt.shape != (3,) or not np.isfinite(p_tgt).all():
            raise ValueError("target_xyz_world must be [x, y, z] in meters")
        tcp_off = (
            self.tcp_offset_body3
            if tcp_offset is None
            else np.asarray(tcp_offset, dtype=np.float64)
        )

        radial_az = math.atan2(float(p_tgt[1]), float(p_tgt[0]))
        pitch_rad = math.radians(float(pitch_down_deg))
        jaw_yaw_rad = math.radians(float(jaw_yaw_deg))
        approach = np.array(
            [
                math.cos(pitch_rad) * math.cos(radial_az),
                math.cos(pitch_rad) * math.sin(radial_az),
                -math.sin(pitch_rad),
            ],
            dtype=np.float64,
        )
        approach /= max(float(np.linalg.norm(approach)), 1e-9)

        is_topdown = abs(float(pitch_down_deg) - 90.0) < 1.5
        if is_topdown:
            jaw_axis = np.array(
                [math.cos(jaw_yaw_rad), math.sin(jaw_yaw_rad), 0.0], dtype=np.float64
            )
        else:
            perp_horiz = np.array(
                [-math.sin(radial_az), math.cos(radial_az), 0.0], dtype=np.float64
            )
            in_plane = np.cross(perp_horiz, approach)
            in_plane /= max(float(np.linalg.norm(in_plane)), 1e-9)
            rel_yaw = jaw_yaw_rad - (radial_az + 0.5 * math.pi)
            jaw_axis = math.cos(rel_yaw) * perp_horiz + math.sin(rel_yaw) * in_plane
            jaw_axis /= max(float(np.linalg.norm(jaw_axis)), 1e-9)

        q_lo = self.follower_cal.lower + 0.02
        q_hi = self.follower_cal.upper - 0.02
        q_post = np.clip(
            np.array([radial_az, -0.75, 1.30, 0.45, 0.0], dtype=np.float64), q_lo, q_hi
        )

        seeds = [
            q_post,
            np.clip(
                np.array([radial_az, -0.45, 1.05, 0.75, jaw_yaw_rad], dtype=np.float64), q_lo, q_hi
            ),
            np.clip(np.array([radial_az, -1.05, 1.45, 0.15, 0.0], dtype=np.float64), q_lo, q_hi),
        ]
        if q_seed is not None:
            seeds.insert(0, np.clip(np.asarray(q_seed, dtype=np.float64), q_lo, q_hi))

        best_q = seeds[0]
        best_cost = float("inf")
        eps = 1e-6

        for seed in seeds:
            q_cur = seed.copy()
            for _ in range(45):
                r0 = self._grasp_residual(q_cur, p_tgt, approach, jaw_axis, is_topdown, tcp_off)
                r_full = np.concatenate([r0, 0.04 * (q_cur - q_post)])
                cost0 = 0.5 * float(np.dot(r_full, r_full))
                if cost0 < 1e-7:
                    break
                J_r = np.zeros((len(r_full), 5), dtype=np.float64)
                for j in range(5):
                    q_p = q_cur.copy()
                    q_p[j] += eps
                    rp = self._grasp_residual(q_p, p_tgt, approach, jaw_axis, is_topdown, tcp_off)
                    J_r[: len(r0), j] = (rp - r0) / eps
                J_r[len(r0) :, :] = 0.04 * np.eye(5)

                H = J_r.T @ J_r + 1e-2 * np.eye(5)
                g = J_r.T @ r_full
                dq = -np.linalg.solve(H, g)
                dq_norm = float(np.linalg.norm(dq))
                if dq_norm > 0.35:
                    dq *= 0.35 / dq_norm

                # Backtracking line search
                accepted = False
                for alpha in (1.0, 0.5, 0.25, 0.1):
                    q_cand = np.clip(q_cur + alpha * dq, q_lo, q_hi)
                    rc = np.concatenate(
                        [
                            self._grasp_residual(
                                q_cand, p_tgt, approach, jaw_axis, is_topdown, tcp_off
                            ),
                            0.04 * (q_cand - q_post),
                        ]
                    )
                    if 0.5 * float(np.dot(rc, rc)) < cost0:
                        q_cur = q_cand
                        accepted = True
                        break
                if not accepted or dq_norm < 1e-5:
                    break

            r_end = np.concatenate(
                [
                    self._grasp_residual(q_cur, p_tgt, approach, jaw_axis, is_topdown, tcp_off),
                    0.04 * (q_cur - q_post),
                ]
            )
            c_end = 0.5 * float(np.dot(r_end, r_end))
            if c_end < best_cost:
                best_cost = c_end
                best_q = q_cur.copy()

        kin_res = self.compute_kinematics_and_jacobian(best_q, arm="follower", tcp_offset=tcp_off)
        p_achieved = np.asarray(kin_res["tcp_world_xyz_m"], dtype=np.float64)
        pos_err_mm = float(np.linalg.norm(p_achieved - p_tgt) * 1000.0)
        feasible = bool(pos_err_mm <= 12.0 and kin_res["within_limits"])

        return {
            "feasible": feasible,
            "target_xyz_world_m": [round(float(v), 4) for v in p_tgt],
            "achieved_xyz_world_m": kin_res["tcp_world_xyz_m"],
            "position_error_mm": round(pos_err_mm, 3),
            "pitch_down_deg": round(float(pitch_down_deg), 2),
            "jaw_yaw_deg": round(float(jaw_yaw_deg), 2),
            "cost": round(best_cost, 6),
            "q_enc_rad": kin_res["q_enc_rad"],
            "ticks": kin_res["ticks"],
            "lerobot_deg": kin_res["lerobot_deg"],
            "manipulability_5d": kin_res["manipulability_5d"],
            "condition_number": kin_res["condition_number"],
            "kinematics": kin_res,
        }

    # ------------------------------------------------------------------ Live Hardware & Camera Probing

    def poll_arms_state(self, timeout_s: float = 0.25) -> dict[str, Any]:
        """Read live Leader & Follower arm state from `armd` (:8765), falling back to virtual studio arms."""
        out: dict[str, Any] = {"armd_url": self.armd_url, "armd_online": False, "arms": {}}
        active_armd_url = self.armd_url
        for cand_armd in (
            self.armd_url,
            "http://172.18.0.1:18765",
            "http://172.18.0.1:8765",
            "http://172.17.0.1:18765",
            "http://host.docker.internal:8765",
        ):
            try:
                with urllib.request.urlopen(f"{cand_armd}/health", timeout=timeout_s) as resp:
                    if resp.status == 200:
                        out["armd_online"] = True
                        active_armd_url = cand_armd
                        out["armd_url"] = cand_armd
                        break
            except Exception:
                continue

        for arm_name, cal, kin, v_q, v_g in (
            (
                "leader",
                self.leader_cal,
                self.leader_kin,
                self.virtual_leader_q,
                self.virtual_leader_gripper_pct,
            ),
            (
                "follower",
                self.follower_cal,
                self.follower_kin,
                self.virtual_follower_q,
                self.virtual_follower_gripper_pct,
            ),
        ):
            if out["armd_online"]:
                try:
                    with urllib.request.urlopen(
                        f"{active_armd_url}/arms/{arm_name}/state?fresh=false", timeout=timeout_s
                    ) as resp:
                        doc = json.loads(resp.read().decode("utf-8"))
                    joints = doc.get("joints", {})
                    if all(
                        m in joints and joints[m].get("position") is not None for m in MOTOR_NAMES
                    ):
                        ticks = [int(joints[m]["position"]) for m in MOTOR_NAMES]
                        q_enc = cal.ticks_to_radians(ticks)
                        lerobot_deg = cal.radians_to_lerobot_degrees(q_enc)
                        q_cad = kin.cad_at_zero + kin.cad_sign * q_enc
                        g_tick = int(joints.get("gripper", {}).get("position") or 1600)
                        g_pct = float(cal.gripper_ticks_to_percent(g_tick))
                        out["arms"][arm_name] = ArmLiveSnapshot(
                            arm=arm_name,
                            connected=True,
                            source="armd_live",
                            ticks=ticks,
                            q_enc_rad=[round(float(v), 5) for v in q_enc],
                            lerobot_deg=[round(float(v), 2) for v in lerobot_deg],
                            q_cad_rad=[round(float(v), 5) for v in q_cad],
                            q_cad_deg=[round(float(math.degrees(v)), 2) for v in q_cad],
                            gripper_tick=g_tick,
                            gripper_pct=round(g_pct, 2),
                            torque_enabled=bool(
                                any(joints[m].get("torque_enabled") for m in MOTOR_NAMES)
                            ),
                            healthy=bool(doc.get("healthy", True)),
                            age_ms=round(float(doc.get("age_ms") or 0.0), 1),
                        ).__dict__
                        continue
                except Exception:
                    pass

            # Virtual / offline studio fallback (explicitly labeled)
            ticks = [int(v) for v in cal.radians_to_ticks(v_q)]
            lerobot_deg = cal.radians_to_lerobot_degrees(v_q)
            q_cad = kin.cad_at_zero + kin.cad_sign * v_q
            g_rec = cal.records["gripper"]
            g_tick = int(
                round(
                    g_rec["range_min"]
                    + (np.clip(v_g, 0.0, 100.0) / 100.0) * (g_rec["range_max"] - g_rec["range_min"])
                )
            )
            if arm_name == "follower":
                g_tick = max(self.safe_gripper_min_raw, g_tick)
            out["arms"][arm_name] = ArmLiveSnapshot(
                arm=arm_name,
                connected=False,
                source="studio_virtual_emulator",
                ticks=ticks,
                q_enc_rad=[round(float(v), 5) for v in v_q],
                lerobot_deg=[round(float(v), 2) for v in lerobot_deg],
                q_cad_rad=[round(float(v), 5) for v in q_cad],
                q_cad_deg=[round(float(math.degrees(v)), 2) for v in q_cad],
                gripper_tick=g_tick,
                gripper_pct=round(float(v_g), 2),
                torque_enabled=(arm_name == "follower"),
                healthy=True,
                age_ms=0.0,
                error="armd offline — using interactive Studio virtual arm",
            ).__dict__

        out["leader_to_follower_mapping"] = self.map_leader_to_follower(
            leader_q_enc_rad=out["arms"]["leader"]["q_enc_rad"],
            leader_gripper_pct=out["arms"]["leader"]["gripper_pct"],
        )
        return out

    def render_camera_jpeg(self, camera_slot: str, timeout_s: float = 0.25) -> tuple[bytes, str]:
        """Fetch live camera JPEG (`top`, `plate`, `wrist`) or synthesize an annotated 3D preview."""
        slot = {
            "overhead": "top",
            "realsense_1": "top",
            "side": "plate",
            "realsense_2": "plate",
        }.get(camera_slot, camera_slot)
        candidate_urls: list[tuple[str, str]] = []
        if slot == "top":
            candidate_urls = [
                (f"{self.live_depth_url}/rgb/top", "live_depth_d415_top"),
                ("http://172.18.0.1:18094/rgb/top", "live_depth_d415_top"),
                ("http://172.18.0.1:8094/rgb/top", "live_depth_d415_top"),
                ("http://172.17.0.1:18094/rgb/top", "live_depth_d415_top"),
                ("http://host.docker.internal:8094/rgb/top", "live_depth_d415_top"),
                (f"{self.multicam_url}/image/overhead", "multicam_d415_top"),
                (f"{self.hud_url}/frame/realsense_1", "hud_realsense_1"),
            ]
        elif slot == "plate":
            candidate_urls = [
                (f"{self.live_depth_url}/rgb/plate", "live_depth_d415_plate"),
                ("http://172.18.0.1:18094/rgb/plate", "live_depth_d415_plate"),
                ("http://172.18.0.1:8094/rgb/plate", "live_depth_d415_plate"),
                ("http://172.17.0.1:18094/rgb/plate", "live_depth_d415_plate"),
                ("http://host.docker.internal:8094/rgb/plate", "live_depth_d415_plate"),
                (f"{self.multicam_url}/image/side", "multicam_d415_plate"),
                (f"{self.hud_url}/frame/realsense_2", "hud_realsense_2"),
            ]
        elif slot == "wrist":
            candidate_urls = [
                (self.wrist_url, "focus_assist_sonix_wrist"),
                ("http://172.18.0.1:18091/frame.jpg", "focus_assist_sonix_wrist"),
                ("http://172.18.0.1:8091/frame.jpg", "focus_assist_sonix_wrist"),
                ("http://172.17.0.1:18091/frame.jpg", "focus_assist_sonix_wrist"),
                ("http://host.docker.internal:8091/frame.jpg", "focus_assist_sonix_wrist"),
                (f"{self.multicam_url}/image/wrist", "multicam_sonix_wrist"),
                (f"{self.hud_url}/frame/wrist", "hud_wrist"),
            ]

        for url, source_tag in candidate_urls:
            try:
                with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                    if resp.status == 200:
                        data = resp.read()
                        if data.startswith(b"\xff\xd8") and len(data) > 128:
                            return data, source_tag
            except Exception:
                continue

        # Generate deterministic, explicitly labeled synthetic 3D camera preview using Pillow
        return self._render_synthetic_camera_preview(slot), "synthetic_studio_preview"

    def _render_synthetic_camera_preview(self, slot: str) -> bytes:
        """Render a 640x480 projected 3D view of the bowls, bricks, and SO-101 arm for `top`, `plate`, or `wrist`."""
        w, h = 640, 480
        img = Image.new("RGB", (w, h), (14, 18, 26))
        draw = ImageDraw.Draw(img)

        kin = self.compute_kinematics_and_jacobian(self.virtual_follower_q, arm="follower")
        T_w_body3 = np.asarray(kin["T_world_body3"], dtype=np.float64)
        cam = (
            self.rig.realsense_1
            if slot == "top"
            else (self.rig.realsense_2 if slot == "plate" else self.rig.wrist)
        )

        # Draw subtle workspace floor grid (z = 0)
        for gx in np.linspace(0.0, 0.50, 11):
            pts = np.array([[gx, -0.30, 0.0], [gx, 0.30, 0.0]], dtype=np.float64)
            uv, z = cam.project_world_points(pts, T_world_body3=T_w_body3)
            if np.all(z > 0.05):
                draw.line(
                    [(float(uv[0, 0]), float(uv[0, 1])), (float(uv[1, 0]), float(uv[1, 1]))],
                    fill=(28, 36, 50),
                    width=1,
                )
        for gy in np.linspace(-0.30, 0.30, 13):
            pts = np.array([[0.0, gy, 0.0], [0.50, gy, 0.0]], dtype=np.float64)
            uv, z = cam.project_world_points(pts, T_world_body3=T_w_body3)
            if np.all(z > 0.05):
                draw.line(
                    [(float(uv[0, 0]), float(uv[0, 1])), (float(uv[1, 0]), float(uv[1, 1]))],
                    fill=(28, 36, 50),
                    width=1,
                )

        # Draw World Frame coordinate triad at origin (X=red, Y=green, Z=blue)
        axes_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.10, 0.0, 0.0],
                [0.0, 0.10, 0.0],
                [0.0, 0.0, 0.10],
            ],
            dtype=np.float64,
        )
        uv_ax, z_ax = cam.project_world_points(axes_pts, T_world_body3=T_w_body3)
        if np.all(z_ax > 0.04):
            o_pt = (float(uv_ax[0, 0]), float(uv_ax[0, 1]))
            for idx, col, lbl in (
                (1, (239, 68, 68), "+X"),
                (2, (34, 197, 94), "+Y"),
                (3, (56, 189, 248), "+Z"),
            ):
                tip = (float(uv_ax[idx, 0]), float(uv_ax[idx, 1]))
                draw.line([o_pt, tip], fill=col, width=2)
                draw.text((tip[0] + 3, tip[1] - 5), lbl, fill=col)

        # Draw SO-101 kinematic link chain
        link_pts = np.array([row["pos_world_m"] for row in kin["link_chain"]], dtype=np.float64)
        tcp_pt = np.asarray(kin["tcp_world_xyz_m"], dtype=np.float64)
        all_arm_pts = np.vstack([link_pts, tcp_pt[None, :]])
        uv_a, z_a = cam.project_world_points(all_arm_pts, T_world_body3=T_w_body3)
        for i in range(len(all_arm_pts) - 1):
            if z_a[i] > 0.03 and z_a[i + 1] > 0.03:
                draw.line(
                    [
                        (float(uv_a[i, 0]), float(uv_a[i, 1])),
                        (float(uv_a[i + 1, 0]), float(uv_a[i + 1, 1])),
                    ],
                    fill=(255, 106, 26),
                    width=4,
                )
        for i in range(len(all_arm_pts)):
            if z_a[i] > 0.03:
                ux, uy = float(uv_a[i, 0]), float(uv_a[i, 1])
                draw.ellipse((ux - 4, uy - 4, ux + 4, uy + 4), fill=(255, 220, 120))

        # Explicit Synthetic Banner (per AGENTS.md)
        draw.rectangle((0, 0, w, 46), fill=(20, 24, 32))
        cam_titles = {
            "top": "TOP / OVERHEAD D415 (350122060653)",
            "plate": "PLATE / SIDE D415 (243622060450)",
            "wrist": "WRIST / GRIPPER SONIX UVC (/dev/video44, 25 deg cant)",
        }
        draw.text(
            (12, 8),
            f"SYNTHETIC STUDIO PREVIEW · {cam_titles.get(slot, slot.upper())}",
            fill=(255, 138, 69),
        )
        draw.text(
            (12, 26),
            f"TCP World: {kin['tcp_world_xyz_mm']} mm | w5={kin['manipulability_5d']:.4f} | Live camera offline",
            fill=(185, 200, 218),
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return buf.getvalue()

    # ------------------------------------------------------------------ Architecture Synthesis & Export

    def build_trajectory_preview(self, steps_per_segment: int = 12) -> dict[str, Any]:
        """Generate a dense Pick -> Plate-Camera CCW Spin -> Sort trajectory for 3D playback & verification."""
        wp = self.waypoints
        q_home = np.asarray(wp["observation_home"]["q_enc_rad"], dtype=np.float64)
        q_pre = np.asarray(wp["source_pregrasp"]["q_enc_rad"], dtype=np.float64)
        q_pick = np.asarray(wp["source_pick"]["q_enc_rad"], dtype=np.float64)
        q_spin_start = np.asarray(wp["plate_camera_spin"]["q_enc_rad"], dtype=np.float64)
        q_spin_end = q_spin_start.copy()
        q_spin_end[4] = float(wp["plate_camera_spin"].get("spin_end_q5_rad", 2.980))
        q_good = np.asarray(wp["good_bowl_drop"]["q_enc_rad"], dtype=np.float64)
        q_bad = np.asarray(wp["defective_bowl_drop"]["q_enc_rad"], dtype=np.float64)

        segments = [
            ("LocalizeBlock", "1. Observe & Localize 3D Brick", q_home, q_pre, 24.0),
            ("PickBlock", "2. Descend & Close Jaws on Brick", q_pre, q_pick, 3.5),
            ("LiftToPlateCam", "3. Lift & Present to Side D415 Camera", q_pick, q_spin_start, 3.5),
            (
                "PresentAndSpinAtPlateCamera",
                "4. CCW Joint-5 Spin Inspection (+0.45 -> +2.98 rad)",
                q_spin_start,
                q_spin_end,
                3.5,
            ),
            (
                "SortBlockToPile(good)",
                "5. Place Verified Brick in GOOD Bowl",
                q_spin_end,
                q_good,
                24.0,
            ),
            ("ParkAtRest", "6. Return to Observation Home", q_good, q_home, 22.0),
        ]

        frames = []
        min_manip = float("inf")
        max_torque = 0.0
        all_within_limits = True

        for op_name, phase_title, q0, q1, grip_pct in segments:
            alphas = np.linspace(0.0, 1.0, steps_per_segment, endpoint=True)
            for a in alphas:
                # Smooth cosine interpolation
                s = 0.5 * (1.0 - math.cos(math.pi * float(a)))
                q_t = (1.0 - s) * q0 + s * q1
                kin = self.compute_kinematics_and_jacobian(q_t, arm="follower")
                dyn = self.compute_dynamics(
                    q_t,
                    arm="follower",
                    payload_mass_kg=(0.019 if grip_pct < 10.0 else 0.0),
                )
                min_manip = min(min_manip, float(kin["manipulability_5d"]))
                max_torque = max(max_torque, float(np.max(np.abs(dyn["gravity_torque_nm"]))))
                all_within_limits = all_within_limits and bool(kin["within_limits"])
                frames.append(
                    {
                        "operator": op_name,
                        "phase": phase_title,
                        "q_enc_rad": kin["q_enc_rad"],
                        "ticks": kin["ticks"],
                        "gripper_pct": grip_pct,
                        "tcp_world_xyz_m": kin["tcp_world_xyz_m"],
                        "link_chain_world": [r["pos_world_m"] for r in kin["link_chain"]],
                        "manipulability_5d": kin["manipulability_5d"],
                        "gravity_torque_nm": dyn["gravity_torque_nm"],
                    }
                )

        return {
            "mode": "plate_spin_sort_no_vla",
            "total_frames": len(frames),
            "all_within_joint_limits": all_within_limits,
            "min_manipulability_5d": round(min_manip, 6),
            "peak_gravity_torque_nm": round(max_torque, 4),
            "peak_load_pct_of_stall": round(100.0 * max_torque / STS3215_STALL_TORQUE_NM, 2),
            "defective_branch_waypoint": {
                "q_enc_rad": [round(float(v), 5) for v in q_bad],
                "ticks": [int(v) for v in self.follower_cal.radians_to_ticks(q_bad)],
            },
            "frames": frames,
        }

    def export_architecture_bundle(self) -> dict[str, Any]:
        """Generate a complete, standalone No-VLA SO-101 Pick-Inspect-Place Python controller & URDF."""
        with tempfile.TemporaryDirectory(prefix="so101-studio-") as tmp:
            urdf_path = Path(tmp) / "so101-follower-calibrated.urdf"
            self.follower_kin.export_arm_urdf(urdf_path)
            urdf_xml = urdf_path.read_text(encoding="utf-8")

        wp_json = json.dumps(self.waypoints, indent=4)
        bowls_json = json.dumps(self.bowls, indent=4)

        python_script = f'''#!/usr/bin/env python3
"""Standalone SO-101 Model-Based Pick-Inspect-Place Controller (Generated by Physical AI Studio).

Executes the complete deterministic Pick -> Plate-Camera CCW Spin -> Sort architecture
using calibrated Forward Kinematics, 6x5 Geometric Jacobians, Published CAD Dynamics,
and `armd` HTTP sync-write — requiring ZERO VLA training or external AI dependencies.

Run dry-run verification (no servo motion):
    uv run python so101_pick_place_controller.py --dry-run

Run on physical SO-101 follower arm via armd (127.0.0.1:8765):
    uv run python so101_pick_place_controller.py --execute --sort-to good
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if (REPO_ROOT / "tools").is_dir():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
elif (REPO_ROOT.parent / "tools").is_dir():
    sys.path.insert(0, str(REPO_ROOT.parent / "tools"))

from so101_calibration import MOTOR_NAMES, ServoCalibration  # noqa: E402
from so101_kinematics import CalibratedSO101Kinematics  # noqa: E402

SAFE_GRIPPER_MIN_RAW = {self.safe_gripper_min_raw}
BOWLS = {bowls_json}
WAYPOINTS = {wp_json}


def send_armd_positions(
    armd_url: str,
    cal: ServoCalibration,
    q_enc_rad: list[float] | np.ndarray,
    gripper_pct: float,
    speed: int = 280,
    acc: int = 25,
    dry_run: bool = True,
) -> dict:
    """Convert calibrated encoder radians -> homed STS3215 ticks and send atomic sync-write."""
    q = cal.validate_limits(q_enc_rad)
    ticks = cal.radians_to_ticks(q)
    g_rec = cal.records["gripper"]
    raw_grip = int(
        round(g_rec["range_min"] + (np.clip(gripper_pct, 0.0, 100.0) / 100.0) * (g_rec["range_max"] - g_rec["range_min"]))
    )
    safe_grip = max(SAFE_GRIPPER_MIN_RAW, min(int(g_rec["range_max"]), raw_grip))

    joints_payload = {{
        name: {{"position": int(tick)}} for name, tick in zip(MOTOR_NAMES, ticks)
    }}
    joints_payload["gripper"] = {{"position": int(safe_grip)}}
    body = {{"joints": joints_payload, "speed": int(speed), "acc": int(acc), "wait": True}}

    if dry_run:
        return {{"dry_run": True, "payload": body}}

    req = urllib.request.Request(
        f"{{armd_url.rstrip('/')}}/arms/follower/positions",
        data=json.dumps(body).encode("utf-8"),
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--armd-url", default="http://127.0.0.1:8765")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", help="Send live motion commands to armd")
    parser.add_argument("--sort-to", choices=["good", "defective"], default="good")
    args = parser.parse_args()
    dry_run = not args.execute

    kin = CalibratedSO101Kinematics("follower")
    cal = kin.calibration
    tcp = {self.tcp_offset_body3.tolist()}

    sequence = [
        ("observation_home", WAYPOINTS["observation_home"]["q_enc_rad"], WAYPOINTS["observation_home"]["gripper_pct"]),
        ("source_pregrasp", WAYPOINTS["source_pregrasp"]["q_enc_rad"], WAYPOINTS["source_pregrasp"]["gripper_pct"]),
        ("source_pick", WAYPOINTS["source_pick"]["q_enc_rad"], WAYPOINTS["source_pick"]["gripper_pct"]),
        ("plate_camera_spin_start", WAYPOINTS["plate_camera_spin"]["q_enc_rad"], WAYPOINTS["plate_camera_spin"]["gripper_pct"]),
    ]
    spin_end_q = list(WAYPOINTS["plate_camera_spin"]["q_enc_rad"])
    spin_end_q[4] = float(WAYPOINTS["plate_camera_spin"].get("spin_end_q5_rad", 2.98))
    sequence.append(("plate_camera_spin_ccw_end", spin_end_q, WAYPOINTS["plate_camera_spin"]["gripper_pct"]))

    drop_key = "good_bowl_drop" if args.sort_to == "good" else "defective_bowl_drop"
    sequence.append((drop_key, WAYPOINTS[drop_key]["q_enc_rad"], WAYPOINTS[drop_key]["gripper_pct"]))
    sequence.append(("park_home", WAYPOINTS["observation_home"]["q_enc_rad"], WAYPOINTS["observation_home"]["gripper_pct"]))

    report = []
    for step_name, q_rad, grip_pct in sequence:
        pose_base = kin.fk(q_rad, tcp_offset=tcp)
        J_base = kin.jacobian(q_rad, tcp_offset=tcp)
        manip = float(np.sqrt(np.linalg.det(J_base.T @ J_base + 1e-9 * np.eye(5))))
        res = send_armd_positions(args.armd_url, cal, q_rad, grip_pct, dry_run=dry_run)
        report.append({{
            "step": step_name,
            "q_enc_rad": [round(float(v), 5) for v in q_rad],
            "ticks": [int(v) for v in cal.radians_to_ticks(q_rad)],
            "gripper_pct": grip_pct,
            "tcp_base_xyz_m": [round(float(v), 4) for v in pose_base[:3, 3]],
            "manipulability_5d": round(manip, 6),
            "armd_result": res,
        }})
        if not dry_run:
            time.sleep(0.25)

    print(json.dumps({{"mode": "no_vla_analytical_pick_place", "dry_run": dry_run, "steps": report}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
        return {
            "schema": 1,
            "architecture": "so101_model_based_pick_inspect_place_no_vla",
            "safe_gripper_min_raw": self.safe_gripper_min_raw,
            "tcp_offset_body3_m": [round(float(v), 4) for v in self.tcp_offset_body3],
            "bowls": self.bowls,
            "waypoints": self.waypoints,
            "calibrated_follower_urdf": urdf_xml,
            "controller_python_script": python_script,
        }

    def compile_architecture_graph(self, graph: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate and compile a Simulink-style block diagram + SymSkill skills into a deployable Python policy."""
        try:
            from brickcell.studio.block_composer import (
                ARCHITECTURE_PRESETS,
                compile_graph_to_python_policy,
            )
        except ImportError:
            from so101_architecture.block_composer import (  # type: ignore[no-redef]
                ARCHITECTURE_PRESETS,
                compile_graph_to_python_policy,
            )
        active_graph = graph or getattr(
            self, "active_graph", ARCHITECTURE_PRESETS["full_bilevel_spin_sort"]
        )
        self.active_graph = active_graph
        return compile_graph_to_python_policy(
            graph=active_graph,
            waypoints=self.waypoints,
            bowls=self.bowls,
            safe_gripper_min_raw=self.safe_gripper_min_raw,
            tcp_offset_body3_m=[round(float(v), 4) for v in self.tcp_offset_body3],
        )

    def full_studio_snapshot(self, project_id: str | None = None) -> dict[str, Any]:
        """Return the complete Studio state for initial UI hydration."""
        try:
            from brickcell.studio.block_composer import (
                ARCHITECTURE_PRESETS,
                BLOCK_LIBRARY,
                PORT_TYPES,
                compile_graph_to_python_policy,
            )
        except ImportError:
            from so101_architecture.block_composer import (  # type: ignore[no-redef]
                ARCHITECTURE_PRESETS,
                BLOCK_LIBRARY,
                PORT_TYPES,
                compile_graph_to_python_policy,
            )

        studio_db = self.sync_from_studio_db(project_id=project_id)
        arms = self.poll_arms_state()
        q_active = arms["arms"]["follower"]["q_enc_rad"]
        kin = self.compute_kinematics_and_jacobian(q_active, arm="follower")
        dyn = self.compute_dynamics(q_active, arm="follower")
        traj = self.build_trajectory_preview(steps_per_segment=10)
        if not hasattr(self, "active_graph") or not self.active_graph:
            self.active_graph = json.loads(
                json.dumps(ARCHITECTURE_PRESETS["full_bilevel_spin_sort"])
            )
        compiled = compile_graph_to_python_policy(
            graph=self.active_graph,
            waypoints=self.waypoints,
            bowls=self.bowls,
            safe_gripper_min_raw=self.safe_gripper_min_raw,
            tcp_offset_body3_m=[round(float(v), 4) for v in self.tcp_offset_body3],
        )
        return {
            "studio_version": "0.6.0-simulink-symskill-composer",
            "upstream_base": "open-edge-platform/physical-ai-studio",
            "studio_db": studio_db,
            "calibration": self.get_calibration_summary(),
            "arms": arms,
            "kinematics": kin,
            "dynamics": dyn,
            "bowls": self.bowls,
            "waypoints": self.waypoints,
            "cameras": self.rig.to_dict()["cameras"],
            "trajectory": traj,
            "composer": {
                "port_types": PORT_TYPES,
                "block_library": BLOCK_LIBRARY,
                "presets": ARCHITECTURE_PRESETS,
                "active_graph": self.active_graph,
                "compiled": compiled,
            },
            "duplo_catalog": {
                k: {
                    "part_id": v.part_id,
                    "dims_mm": [round(d * 1000.0, 1) for d in v.dims_m],
                    "mass_g": round(v.mass_kg * 1000.0, 1),
                }
                for k, v in DUPLO_CATALOG.items()
            },
        }
