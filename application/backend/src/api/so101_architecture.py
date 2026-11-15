# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FastAPI router for the SO-101 Model-Based Pick-and-Place Architecture Workbench in Physical AI Studio."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from so101_architecture.engine import SO101StudioEngine
from so101_architecture.inspection import MAX_INSPECTION_BODY, inspection_evidence

router = APIRouter(prefix="/api/so101", tags=["SO-101 Architecture"])

_UI_EXT_PATH = Path(__file__).resolve().parent.parent / "so101_architecture" / "ui_extension.js"


@lru_cache(maxsize=1)
def get_so101_engine() -> SO101StudioEngine:
    """Return singleton SO101StudioEngine instance inside Physical AI Studio."""
    return SO101StudioEngine()


class KinematicsRequest(BaseModel):
    arm: str = Field(default="follower", description="Target arm: 'follower' or 'leader'")
    q_enc_rad: list[float] | None = Field(
        default=None, description="5 calibrated encoder angles in radians"
    )
    ticks: list[float] | None = Field(default=None, description="5 homed STS3215 servo tick values")
    lerobot_deg: list[float] | None = Field(
        default=None, description="5 LeRobot joint values in degrees"
    )
    normalized: list[float] | None = Field(
        default=None,
        description="5 Physical AI Studio normalized [-100, 100] joint values",
    )
    gripper_pct: float | None = Field(
        default=None, description="Gripper opening percentage [0, 100]"
    )
    payload_mass_kg: float | None = Field(
        default=None, description="Held payload mass at TCP in kg"
    )
    tcp_offset: list[float] | None = Field(
        default=None, description="TCP offset [x, y, z] in Body3 frame (m)"
    )


class IKRequest(BaseModel):
    target_xyz_world: list[float] = Field(
        ..., description="Target TCP [x, y, z] in World frame (meters)"
    )
    pitch_down_deg: float = Field(
        default=90.0, description="Approach pitch-down angle in degrees (90 = top-down)"
    )
    jaw_yaw_deg: float = Field(default=0.0, description="Desired jaw yaw angle in degrees")
    q_seed: list[float] | None = Field(
        default=None, description="Optional 5-joint seed in encoder radians"
    )
    apply_to_waypoint: str | None = Field(
        default=None, description="Optional waypoint key to update with IK solution"
    )
    update_virtual_arm: bool = Field(
        default=True, description="Update virtual follower arm to solved configuration"
    )


class TeachWaypointRequest(BaseModel):
    waypoint: str = Field(
        ..., description="Waypoint key (e.g. 'source_pick', 'plate_camera_spin', 'good_bowl_drop')"
    )
    source_arm: str = Field(
        default="leader", description="'leader' (maps to follower via CAD) or 'follower'"
    )
    q_enc_rad: list[float] | None = Field(
        default=None, description="Optional explicit 5-joint encoder radians"
    )
    ticks: list[float] | None = Field(default=None, description="Optional explicit 5-joint ticks")
    normalized: list[float] | None = Field(
        default=None,
        description="Optional explicit 5-joint Physical AI Studio normalized [-100, 100] values",
    )
    gripper_pct: float | None = Field(
        default=None, description="Optional gripper opening percentage [0, 100]"
    )


class AlignArmRequest(BaseModel):
    arm: str = Field(default="follower", description="'follower' or 'leader'")
    measured_ticks: list[float] = Field(
        ..., description="5 homed STS3215 ticks recorded at reference pose"
    )
    cad_target_deg: list[float] = Field(
        default=[0.0, 93.1, -89.5, 1.5, 90.0],
        description="Reference CAD joint angles in degrees",
    )
    cad_signs: list[int] = Field(
        default=[1, 1, 1, 1, 1], description="Per-joint direction signs (+1 or -1)"
    )


class BowlsUpdateRequest(BaseModel):
    source: list[float] | None = Field(
        default=None, description="Source bowl center [x, y, z] in World meters"
    )
    good: list[float] | None = Field(
        default=None, description="Good bowl center [x, y, z] in World meters"
    )
    defective: list[float] | None = Field(
        default=None, description="Defective bowl center [x, y, z] in World meters"
    )


class SyncProjectRequest(BaseModel):
    project_id: str | None = Field(
        default=None, description="Physical AI Studio project UUID to sync from"
    )
    custom_calibration: dict[str, Any] | None = Field(
        default=None,
        description="Optional custom calibration dictionary to apply directly",
    )
    arm: str = Field(default="follower", description="Arm to apply custom_calibration to")


@router.get("/inspection")
async def get_inspection_evidence() -> dict[str, Any]:
    return inspection_evidence(get_so101_engine()).snapshot()


@router.post("/inspection")
async def publish_inspection_evidence(request: Request) -> dict[str, Any]:
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_INSPECTION_BODY:
            raise HTTPException(status_code=413, detail="Inspection request too large")
    try:
        import json

        return inspection_evidence(get_so101_engine()).publish(json.loads(raw))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/state")
async def get_so101_state(
    project_id: str | None = Query(
        default=None, description="Optional Physical AI Studio project ID"
    ),
) -> dict[str, Any]:
    """Return full SO-101 Architecture Workbench snapshot (calibration, arms, FK, Jacobians, dynamics, IK, cameras)."""
    engine = get_so101_engine()
    return engine.full_studio_snapshot(project_id=project_id)


@router.post("/sync-project")
async def sync_so101_project(req: SyncProjectRequest) -> dict[str, Any]:
    """Sync SO-101 Leader & Follower calibrations and cameras from Physical AI Studio's database or custom payload."""
    engine = get_so101_engine()
    if req.custom_calibration is not None:
        try:
            engine.apply_custom_calibration(req.arm, req.custom_calibration)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return engine.full_studio_snapshot(project_id=req.project_id)


@router.post("/kinematics")
async def compute_so101_kinematics(req: KinematicsRequest) -> dict[str, Any]:
    """Compute live FK, 6x5 Geometric Jacobians (rad/tick/deg), manipulability, and published CAD dynamics."""
    engine = get_so101_engine()
    cal = engine.follower_cal if req.arm == "follower" else engine.leader_cal

    try:
        if req.q_enc_rad is not None:
            q_rad = cal.validate_limits(req.q_enc_rad)
        elif req.ticks is not None:
            q_rad = cal.ticks_to_radians(req.ticks)
        elif req.lerobot_deg is not None:
            q_rad = cal.lerobot_degrees_to_radians(req.lerobot_deg)
        elif req.normalized is not None:
            q_rad = engine.normalized_to_radians(req.normalized, arm=req.arm)
        else:
            q_rad = engine.virtual_follower_q if req.arm == "follower" else engine.virtual_leader_q
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if req.arm == "follower":
        engine.virtual_follower_q = q_rad.copy()
        if req.gripper_pct is not None:
            engine.virtual_follower_gripper_pct = float(req.gripper_pct)
    else:
        engine.virtual_leader_q = q_rad.copy()
        if req.gripper_pct is not None:
            engine.virtual_leader_gripper_pct = float(req.gripper_pct)

    if req.payload_mass_kg is not None:
        engine.payload_mass_kg = max(0.0, float(req.payload_mass_kg))

    kin = engine.compute_kinematics_and_jacobian(q_rad, arm=req.arm, tcp_offset=req.tcp_offset)
    kin["normalized"] = [
        round(float(v), 2) for v in engine.radians_to_normalized(q_rad, arm=req.arm)
    ]
    dyn = engine.compute_dynamics(
        q_rad,
        arm=req.arm,
        payload_mass_kg=engine.payload_mass_kg,
        tcp_offset=req.tcp_offset,
    )
    arms = engine.poll_arms_state()
    return {
        "arm": req.arm,
        "kinematics": kin,
        "dynamics": dyn,
        "arms": arms,
    }


@router.post("/ik")
async def solve_so101_ik(req: IKRequest) -> dict[str, Any]:
    """Solve 5-DOF collision-checked Grasp/Placement IK in pure NumPy."""
    engine = get_so101_engine()
    try:
        sol = engine.solve_5dof_ik(
            target_xyz_world=req.target_xyz_world,
            pitch_down_deg=req.pitch_down_deg,
            jaw_yaw_deg=req.jaw_yaw_deg,
            q_seed=req.q_seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    q_rad = sol["q_enc_rad"]
    sol["normalized"] = [
        round(float(v), 2) for v in engine.radians_to_normalized(q_rad, arm="follower")
    ]
    if req.update_virtual_arm:
        import numpy as np

        engine.virtual_follower_q = np.asarray(q_rad, dtype=np.float64)
    if req.apply_to_waypoint and req.apply_to_waypoint in engine.waypoints:
        engine.waypoints[req.apply_to_waypoint]["q_enc_rad"] = [round(float(v), 5) for v in q_rad]
        sol["updated_waypoint"] = req.apply_to_waypoint
        sol["trajectory"] = engine.build_trajectory_preview(steps_per_segment=10)

    sol["dynamics"] = engine.compute_dynamics(q_rad, arm="follower")
    return sol


@router.post("/teach")
async def teach_so101_waypoint(req: TeachWaypointRequest) -> dict[str, Any]:
    """Capture Leader or Follower arm pose and store it into a Pick-Inspect-Place waypoint."""
    engine = get_so101_engine()
    if req.waypoint not in engine.waypoints:
        raise HTTPException(status_code=400, detail=f"Unknown waypoint: {req.waypoint!r}")

    import numpy as np

    if req.source_arm == "leader":
        if req.normalized is not None:
            q_lead = engine.normalized_to_radians(req.normalized, arm="leader")
            mapped = engine.map_leader_to_follower(
                leader_q_enc_rad=q_lead, leader_gripper_pct=req.gripper_pct
            )
        elif req.ticks is not None:
            mapped = engine.map_leader_to_follower(
                leader_ticks=req.ticks, leader_gripper_pct=req.gripper_pct
            )
        elif req.q_enc_rad is not None:
            mapped = engine.map_leader_to_follower(
                leader_q_enc_rad=req.q_enc_rad, leader_gripper_pct=req.gripper_pct
            )
        else:
            arms = engine.poll_arms_state()
            mapped = arms["leader_to_follower_mapping"]
        q_fol = mapped["follower_q_enc_rad"]
        grip_pct = (
            mapped["follower_gripper_pct"] if req.gripper_pct is None else float(req.gripper_pct)
        )
    else:
        if req.normalized is not None:
            q_fol = [
                round(float(v), 5)
                for v in engine.normalized_to_radians(req.normalized, arm="follower")
            ]
        elif req.ticks is not None:
            q_fol = [round(float(v), 5) for v in engine.follower_cal.ticks_to_radians(req.ticks)]
        elif req.q_enc_rad is not None:
            q_fol = [round(float(v), 5) for v in req.q_enc_rad]
        else:
            q_fol = [round(float(v), 5) for v in engine.virtual_follower_q]
        grip_pct = float(
            engine.virtual_follower_gripper_pct if req.gripper_pct is None else req.gripper_pct
        )

    engine.waypoints[req.waypoint]["q_enc_rad"] = q_fol
    engine.waypoints[req.waypoint]["gripper_pct"] = round(grip_pct, 2)
    engine.virtual_follower_q = np.asarray(q_fol, dtype=np.float64)
    engine.virtual_follower_gripper_pct = grip_pct

    return {
        "updated_waypoint": req.waypoint,
        "waypoint_data": engine.waypoints[req.waypoint],
        "waypoints": engine.waypoints,
        "kinematics": engine.compute_kinematics_and_jacobian(q_fol, arm="follower"),
        "dynamics": engine.compute_dynamics(q_fol, arm="follower"),
        "trajectory": engine.build_trajectory_preview(steps_per_segment=10),
    }


@router.post("/align")
async def align_so101_arm(req: AlignArmRequest) -> dict[str, Any]:
    """Derive per-arm CAD zero alignment (`cad_at_zero_deg` and `cad_sign`) from upright reference ticks."""
    engine = get_so101_engine()
    try:
        return engine.derive_alignment_from_reference_ticks(
            arm=req.arm,
            measured_ticks=req.measured_ticks,
            cad_target_deg=req.cad_target_deg,
            cad_signs=req.cad_signs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/bowls")
async def update_so101_bowls(req: BowlsUpdateRequest) -> dict[str, Any]:
    """Update workspace KALAS bowl positions and automatically re-solve analytical 5-DOF IK waypoints."""
    engine = get_so101_engine()
    for key, val in (("source", req.source), ("good", req.good), ("defective", req.defective)):
        if val is not None:
            if len(val) != 3:
                raise HTTPException(status_code=400, detail=f"{key} must be [x, y, z] in meters")
            engine.bowls[key]["pos_world_m"] = [round(float(v), 4) for v in val]
    engine._refresh_ik_waypoints()
    return {
        "bowls": engine.bowls,
        "waypoints": engine.waypoints,
        "trajectory": engine.build_trajectory_preview(steps_per_segment=10),
    }


class CompileGraphRequest(BaseModel):
    graph: dict[str, Any] | None = Field(
        default=None,
        description="Simulink-style SO-101 architecture graph (nodes + typed edges + custom_skills)",
    )


@router.post("/compile-graph")
async def compile_so101_graph(req: CompileGraphRequest) -> dict[str, Any]:
    """Validate port compatibility on a custom Simulink-style block diagram and compile to a standalone Python policy."""
    engine = get_so101_engine()
    return engine.compile_architecture_graph(graph=req.graph)


@router.get("/export")
async def export_so101_architecture() -> dict[str, Any]:
    """Export standalone No-VLA SO-101 Pick-Inspect-Place Python controller, calibrated URDF, and waypoints."""
    engine = get_so101_engine()
    return engine.export_architecture_bundle()


@router.get("/camera/{slot}.jpg")
async def get_so101_camera_jpeg(slot: str) -> Response:
    """Return live camera JPEG (`top`, `plate`/`side`, `wrist`) or annotated 3D synthetic preview."""
    engine = get_so101_engine()
    jpeg_bytes, source_tag = engine.render_camera_jpeg(slot)
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Camera-Source": source_tag,
        },
    )


@router.get("/ui-extension.js")
async def get_so101_ui_extension() -> Response:
    """Serve the Physical AI Studio SO-101 Architecture Workbench UI script."""
    if not _UI_EXT_PATH.is_file():
        return Response(
            content="// ui_extension.js not found\n", media_type="application/javascript"
        )
    return Response(
        content=_UI_EXT_PATH.read_text(encoding="utf-8"),
        media_type="application/javascript",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
