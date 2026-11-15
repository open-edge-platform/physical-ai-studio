"""SO-101 reference FK/Jacobian and a five-joint Frax model loader.

Distances are metres, joint angles are radians from the exported CAD pose.
The TCP is a fixed offset in Body3 (wrist-roll child), not a moving fingertip.
These coordinates are not calibrated servo register values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

ARM_JOINTS = (
    "Revolute_1",
    "Revolute_2",
    "Revolute_3",
    "Revolute_2_5",
    "Revolute_4",
)
WRIST_LINK = "Body3"
_HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = (
    (_HERE / "SO-101.zip")
    if (_HERE / "SO-101.zip").is_file()
    else (Path(__file__).resolve().parents[1] / "SO-101.zip")
)


def _vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return value


def _rotation(axis, angle):
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    return out


class SO101Kinematics:
    """Base-expressed geometric Jacobian with row order [linear; angular]."""

    def __init__(self, archive=DEFAULT_ARCHIVE):
        with zipfile.ZipFile(archive) as z:
            model = json.loads(z.read("robotics_ws_model/kinematics.json"))
            self._urdf = z.read("robotics_ws_model/robot.urdf")
        if model["couplings"] or model["cuts"]:
            raise ValueError("This reference expects the uncoupled exported tree")
        self.base = model["base"]
        homes = {}
        for link in model["links"]:
            h = np.eye(4)
            h[:3, :3] = link.get("rotation", np.eye(3))
            h[:3, 3] = link["origin"]
            homes[link["name"]] = h
        self.joints = []
        seen = {self.base}
        for j in model["joints"]:
            if j["type"] != "revolute" or j["parent"] not in seen:
                raise ValueError("Expected topologically ordered revolute joints")
            child, parent = j["child"], j["parent"]
            axis = homes[child][:3, :3].T @ np.array(j["coordinates"][0]["axis"])
            axis /= np.linalg.norm(axis)
            self.joints.append(
                (j["name"], parent, child, np.linalg.inv(homes[parent]) @ homes[child], axis)
            )
            seen.add(child)
        if not set(ARM_JOINTS) <= {j[0] for j in self.joints}:
            raise ValueError("The archive does not contain the expected arm joints")

    def frames(self, q):
        """All link frames; non-arm hub and gripper coordinates remain zero."""
        angles = dict(zip(ARM_JOINTS, _vector(q, 5, "q")))
        frames = {self.base: np.eye(4)}
        for name, parent, child, home, axis in self.joints:
            frames[child] = frames[parent] @ home @ _rotation(axis, angles.get(name, 0.0))
        return frames

    def fk(self, q, tcp_offset=(0.0, 0.0, 0.0)):
        """Base-to-TCP pose; TCP axes are aligned with Body3 axes."""
        tcp = np.eye(4)
        tcp[:3, 3] = _vector(tcp_offset, 3, "tcp_offset")
        return self.frames(q)[WRIST_LINK] @ tcp

    def jacobian(self, q, tcp_offset=(0.0, 0.0, 0.0)):
        """6x5 matrix mapping rad/s to [m/s; rad/s] in the base frame."""
        offset = _vector(tcp_offset, 3, "tcp_offset")
        frames = self.frames(q)
        wrist = frames[WRIST_LINK]
        point = wrist[:3, 3] + wrist[:3, :3] @ offset
        columns = {}
        for name, _, child, _, axis in self.joints:
            if name in ARM_JOINTS:
                t = frames[child]
                direction = t[:3, :3] @ axis
                columns[name] = np.r_[np.cross(direction, point - t[:3, 3]), direction]
        return np.column_stack([columns[name] for name in ARM_JOINTS])

    def export_arm_urdf(self, destination):
        """Make a kinematics-only Frax URDF, locking the other 12 CAD joints."""
        root = ET.fromstring(self._urdf)
        for link in root.findall("link"):
            for tag in ("visual", "collision"):
                for child in link.findall(tag):
                    link.remove(child)
            # The CAD export has no inertias. Frax requires inertial elements
            # even for FK/IK; zero values mark this as a massless kinematic model.
            if link.find("inertial") is None:
                inertial = ET.SubElement(link, "inertial")
                ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
                ET.SubElement(inertial, "mass", value="0")
                ET.SubElement(
                    inertial, "inertia", ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0"
                )
        for joint in root.findall("joint"):
            if joint.get("name") not in ARM_JOINTS:
                joint.set("type", "fixed")
                for tag in ("axis", "limit", "dynamics", "mimic"):
                    for child in joint.findall(tag):
                        joint.remove(child)
        ET.indent(root)
        ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)

    def load_frax(self, tcp_offset=(0.0, 0.0, 0.0)):
        """Load installed Frax's Manipulator with five independent coordinates.

        The temporary URDF has no mesh paths, so the original archive does not
        need extracting. Frax fully parses the file during construction.
        """
        from frax import Manipulator

        ee_offset = np.eye(4)
        ee_offset[:3, 3] = _vector(tcp_offset, 3, "tcp_offset")
        with tempfile.TemporaryDirectory(prefix="so101-frax-") as tmp:
            path = Path(tmp) / "so101-arm.urdf"
            self.export_arm_urdf(path)
            # Frax caches 1 / total_mass even when only FK/IK is used. The
            # massless export deliberately has no valid dynamics/COM model.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="divide by zero encountered in scalar divide",
                    module=r"frax\.core\.robot",
                )
                return Manipulator(str(path), joint_ordering=list(ARM_JOINTS), ee_offset=ee_offset)


class CalibratedSO101Kinematics(SO101Kinematics):
    """FK/Jacobian in a specific arm's midpoint-zero encoder radians.

    Calibration ranges alone cannot determine CAD pose alignment. Construction
    requires a confirmed per-arm CAD angle and sign mapping for all five joints.
    """

    def __init__(self, arm, archive=DEFAULT_ARCHIVE, alignment_path=None):
        from so101_calibration import ALIGNMENT_PATH, ServoCalibration

        super().__init__(archive)
        self.calibration = ServoCalibration(arm)
        alignment_file = alignment_path or ALIGNMENT_PATH
        offsets, signs = self.calibration.alignment(alignment_file)
        metadata = json.loads(Path(alignment_file).read_text(encoding="utf-8"))["arms"][arm]
        model_hash = metadata.get("reference_model_sha256")
        if model_hash and model_hash != hashlib.sha256(Path(archive).read_bytes()).hexdigest():
            raise ValueError("Alignment reference does not match the CAD model archive")
        self.cad_at_zero = offsets
        self.cad_sign = signs
        updated = []
        for name, parent, child, home, axis in self.joints:
            if name in ARM_JOINTS:
                index = ARM_JOINTS.index(name)
                home = home @ _rotation(axis, offsets[index])
                axis = axis * signs[index]
            updated.append((name, parent, child, home, axis))
        self.joints = updated

    def export_arm_urdf(self, destination):
        super().export_arm_urdf(destination)
        root = ET.parse(destination).getroot()
        root.set("name", f"so101_{self.calibration.arm}_calibrated_kinematics")
        rows = {row[0]: row for row in self.joints}
        for joint in root.findall("joint"):
            name = joint.get("name")
            if name not in ARM_JOINTS:
                continue
            index = ARM_JOINTS.index(name)
            _, _, _, home, axis = rows[name]
            r = home[:3, :3]
            pitch = np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0]))
            if np.hypot(r[0, 0], r[1, 0]) > 1e-12:
                roll, yaw = np.arctan2(r[2, 1], r[2, 2]), np.arctan2(r[1, 0], r[0, 0])
            else:
                roll, yaw = np.arctan2(-r[1, 2], r[1, 1]), 0.0
            origin = joint.find("origin")
            origin.set("xyz", " ".join(format(v, ".17g") for v in home[:3, 3]))
            origin.set("rpy", " ".join(format(v, ".17g") for v in [roll, pitch, yaw]))
            joint.find("axis").set("xyz", " ".join(format(v, ".17g") for v in axis))
            joint.set("type", "revolute")
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            # Only encoder position intervals were calibrated. Dynamics/speed
            # remain unspecified; this is a kinematics-only model.
            limit.attrib.update(
                lower=format(self.calibration.lower[index], ".17g"),
                upper=format(self.calibration.upper[index], ".17g"),
                effort="inf",
                velocity="inf",
            )
        ET.indent(root)
        ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=float, nargs=5, default=[0.0] * 5, metavar="RAD")
    parser.add_argument("--tcp", type=float, nargs=3, default=[0.0] * 3, metavar="M")
    parser.add_argument("--frax", action="store_true", help="also compare installed Frax")
    parser.add_argument(
        "--arm", choices=("follower", "leader"), help="use confirmed calibrated encoder coordinates"
    )
    parser.add_argument(
        "--alignment", type=Path, help="per-arm CAD alignment file (requires --arm)"
    )
    args = parser.parse_args()
    if args.alignment and not args.arm:
        parser.error("--alignment requires --arm")
    model = (
        CalibratedSO101Kinematics(args.arm, alignment_path=args.alignment)
        if args.arm
        else SO101Kinematics()
    )
    np.set_printoptions(precision=8, suppress=True)
    print("Joint order:", ", ".join(ARM_JOINTS))
    print("Base-to-TCP pose:\n", model.fk(args.q, args.tcp))
    print("Geometric Jacobian [linear; angular]:\n", model.jacobian(args.q, args.tcp))
    if args.frax:
        robot = model.load_frax(args.tcp)
        print("Frax nv:", robot.nv)
        print("Frax Jacobian:\n", np.asarray(robot.ee_jacobian(np.array(args.q))))


if __name__ == "__main__":
    main()
