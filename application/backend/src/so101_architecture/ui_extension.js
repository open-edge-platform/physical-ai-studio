/**
 * Physical AI Studio — SO-101 Model-Based Pick-and-Place Architecture Workbench Extension
 *
 * Seamlessly integrates into Intel's Physical AI Studio UI (both via React route and
 * runtime SPA tab injection) to provide:
 *   1. All-Camera Feeds (Top D415, Side/Plate D415, Wrist Sonix UVC) via Studio WS or HTTP
 *   2. Leader & Follower SO-101 Arm Calibration Sync, Live Observation WS & Waypoint Teaching
 *   3. Exact Forward Kinematics (4x4 SE(3) in Base & World frames)
 *   4. 6x5 Geometric Jacobians (rad, STS3215 ticks, LeRobot degrees) + Manipulability w5(q)
 *   5. Published CAD Dynamics (5x5 Mass Matrix M(q), Gravity Torque g(q), COM, Stall Load %)
 *   6. Pure-NumPy 5-DOF Grasp/Placement Inverse Kinematics Solver
 *   7. 6-Stage Deterministic Pick -> Plate-Camera Spin -> Sort Trajectory Synthesis & Export
 */
(function () {
  "use strict";

  if (window.__SO101_ARCH_EXTENSION_LOADED__) return;
  window.__SO101_ARCH_EXTENSION_LOADED__ = true;

  const MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"];
  const WAYPOINT_ORDER = [
    "observation_home",
    "source_pregrasp",
    "source_pick",
    "plate_camera_spin",
    "good_bowl_drop",
    "defective_bowl_drop",
  ];

  const state = {
    open: false,
    snap: null,
    activeArm: "follower",
    unitMode: "q_enc_rad", // "q_enc_rad" | "ticks" | "lerobot_deg" | "normalized"
    jacMode: "J_world_rad", // "J_world_rad" | "J_base_rad" | "J_world_tick" | "J_world_lerobot_deg"
    fkFrame: "world", // "world" | "base"
    camStreamActive: true,
    camStreamTimer: null,
    leaderWs: null,
    leaderWsConnected: false,
    playingTraj: false,
    trajFrameIdx: 0,
    trajTimer: null,
    orbitAz: -0.65,
    orbitEl: 0.42,
    orbitDist: 0.85,
    dragging3d: false,
    lastMouse: { x: 0, y: 0 },
    lastIkResult: null,
    lastAlignResult: null,
    activePresetId: "full_bilevel_spin_sort",
    activeGraph: null,
    compiledBundle: null,
    selectedNodeId: null,
    wiringFrom: null, // { nodeId, portId, portType }
    draggingNodeId: null,
    dragNodeOffset: { x: 0, y: 0 },
    journeyStep: "architecture", // "architecture" | "skills" | "live" | "math" | "deploy"
    mathTab: "fk_ik", // "fk_ik" | "jacobian" | "dynamics" | "calibration"
  };

  const JOURNEY_STEPS = [
    { id: "architecture", label: "Architecture", summary: "Compose blocks & signal wires" },
    { id: "skills", label: "Skills & Waypoints", summary: "Define custom skills & poses" },
    { id: "live", label: "Live Twin & Control", summary: "Cameras, 3D arm & teleop" },
    { id: "math", label: "Kinematics & Calibration", summary: "FK, IK, Jacobian, dynamics & zeroing" },
    { id: "deploy", label: "Deploy", summary: "Compiled policy & hardware exports" },
  ];

  function injectStyles() {
    if (document.getElementById("so101-arch-styles")) return;
    const style = document.createElement("style");
    style.id = "so101-arch-styles";
    style.textContent = `
      .so101-nav-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #0054ae 0%, #0071c5 100%);
        color: #ffffff !important;
        border: 1px solid #38bdf8;
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0, 113, 197, 0.35);
        transition: all 0.15s ease;
        z-index: 9999;
        white-space: nowrap;
      }
      .so101-nav-btn.floating-dock {
        position: fixed;
        bottom: 6px;
        right: 16px;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 11.5px;
      }
      .so101-nav-btn:hover {
        background: linear-gradient(135deg, #0066cc 0%, #0ea5e9 100%);
        transform: translateY(-1px);
      }
      .so101-nav-btn.active {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%);
        border-color: #fdba74;
      }
      .so101-badge {
        background: rgba(255,255,255,0.18);
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 10px;
        letter-spacing: 0.4px;
        text-transform: uppercase;
      }
      #so101-workbench-overlay {
        position: fixed;
        top: 64px;
        left: 0;
        right: 0;
        bottom: 32px;
        background: #111318;
        color: #e2e8f0;
        z-index: 9500;
        overflow-y: auto;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        padding: 18px 24px 48px 24px;
        box-sizing: border-box;
      }
      .so101-header-bar {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        background: #1a1d24;
        border: 1px solid #2d3340;
        border-left: 4px solid #0071c5;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
      }
      .so101-title-block h2 {
        margin: 0 0 4px 0;
        font-size: 18px;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .so101-title-block p {
        margin: 0;
        font-size: 12.5px;
        color: #94a3b8;
      }
      .so101-btn-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
      }
      .so101-btn {
        background: #262b36;
        color: #e2e8f0;
        border: 1px solid #3b4252;
        border-radius: 6px;
        padding: 7px 12px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.12s ease;
      }
      .so101-btn:hover {
        background: #323947;
        border-color: #64748b;
      }
      .so101-btn.primary {
        background: #0071c5;
        border-color: #38bdf8;
        color: #fff;
      }
      .so101-btn.primary:hover {
        background: #0284c7;
      }
      .so101-btn.accent {
        background: #ea580c;
        border-color: #fb923c;
        color: #fff;
      }
      .so101-btn.accent:hover {
        background: #f97316;
      }
      .so101-btn.success {
        background: #059669;
        border-color: #34d399;
        color: #fff;
      }
      .so101-grid-2 {
        display: grid;
        grid-template-columns: 1.15fr 1fr;
        gap: 16px;
        margin-bottom: 16px;
      }
      .so101-grid-3 {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-bottom: 16px;
      }
      @media (max-width: 1280px) {
        .so101-grid-2, .so101-grid-3 {
          grid-template-columns: 1fr;
        }
      }
      .so101-card {
        background: #181b22;
        border: 1px solid #2a303c;
        border-radius: 8px;
        padding: 14px 16px;
      }
      .so101-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #262b36;
      }
      .so101-card-header h3 {
        margin: 0;
        font-size: 14px;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: 0.2px;
      }
      .so101-pill {
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 999px;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
      }
      .so101-pill.warn {
        color: #fbbf24;
        border-color: #78350f;
        background: #271a0c;
      }
      .so101-pill.ok {
        color: #34d399;
        border-color: #065f46;
        background: #06241c;
      }
      .so101-cam-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
      }
      .so101-cam-box {
        background: #0f1217;
        border: 1px solid #262b36;
        border-radius: 6px;
        overflow: hidden;
      }
      .so101-cam-box img {
        width: 100%;
        height: 175px;
        object-fit: cover;
        display: block;
        background: #090b0e;
      }
      .so101-cam-meta {
        padding: 6px 8px;
        font-size: 11px;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      .so101-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }
      .so101-table th, .so101-table td {
        padding: 6px 8px;
        text-align: left;
        border-bottom: 1px solid #222731;
        font-variant-numeric: tabular-nums;
      }
      .so101-table th {
        color: #94a3b8;
        font-weight: 600;
        background: #13161c;
      }
      .so101-mono {
        font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
        font-size: 11.5px;
      }
      .so101-slider-row {
        display: grid;
        grid-template-columns: 125px 1fr 78px 95px;
        gap: 8px;
        align-items: center;
        margin-bottom: 7px;
        font-size: 12px;
      }
      .so101-input {
        background: #0f1217;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 4px 7px;
        font-size: 12px;
        font-family: inherit;
        width: 100%;
        box-sizing: border-box;
      }
      .so101-wp-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
      }
      .so101-wp-card {
        background: #12151b;
        border: 1px solid #282e3b;
        border-radius: 6px;
        padding: 8px 10px;
      }
      .so101-wp-card h4 {
        margin: 0 0 4px 0;
        font-size: 12px;
        color: #38bdf8;
      }
      .so101-matrix {
        background: #0d1015;
        border: 1px solid #242936;
        border-radius: 6px;
        padding: 8px;
        overflow-x: auto;
      }
      /* Simulink-Style Block Composer Canvas */
      .so101-composer-wrap {
        background: linear-gradient(180deg, #151922 0%, #11141c 100%);
        border: 1px solid #2e3646;
        border-top: 3px solid #38bdf8;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
      }
      .so101-toolbar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        justify-content: space-between;
        background: #0e1117;
        border: 1px solid #262c3a;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
      }
      .so101-canvas-layout {
        display: grid;
        grid-template-columns: 1fr 340px;
        gap: 14px;
      }
      @media (max-width: 1280px) {
        .so101-canvas-layout {
          grid-template-columns: 1fr;
        }
      }
      .so101-simulink-viewport {
        position: relative;
        height: 450px;
        background-color: #090c10;
        background-image:
          radial-gradient(#222938 1px, transparent 1px),
          linear-gradient(to right, rgba(30,41,59,0.22) 1px, transparent 1px),
          linear-gradient(to bottom, rgba(30,41,59,0.22) 1px, transparent 1px);
        background-size: 20px 20px, 100px 100px, 100px 100px;
        border: 1px solid #283040;
        border-radius: 8px;
        overflow: auto;
        user-select: none;
      }
      .so101-simulink-stage {
        position: relative;
        width: 2550px;
        height: 520px;
      }
      .so101-simulink-wires {
        position: absolute;
        top: 0;
        left: 0;
        width: 2550px;
        height: 520px;
        pointer-events: none;
        z-index: 1;
      }
      .so101-node {
        position: absolute;
        width: 224px;
        background: #161b24;
        border: 1.5px solid #334155;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.45);
        z-index: 2;
        cursor: grab;
        transition: border-color 0.12s ease, box-shadow 0.12s ease;
      }
      .so101-node:hover {
        border-color: #64748b;
      }
      .so101-node.selected {
        border-color: #38bdf8;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.35), 0 6px 18px rgba(0,0,0,0.6);
      }
      .so101-node-hdr {
        padding: 6px 10px;
        border-radius: 6px 6px 0 0;
        font-size: 11.5px;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      .so101-node-sub {
        padding: 4px 10px;
        font-size: 10px;
        color: #94a3b8;
        background: #11151d;
        border-bottom: 1px solid #222834;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .so101-node-ports {
        display: flex;
        justify-content: space-between;
        padding: 7px 6px;
        gap: 6px;
        background: #141820;
      }
      .so101-port-col {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      .so101-port-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 10px;
        font-family: "JetBrains Mono", Consolas, monospace;
        color: #cbd5e1;
        padding: 2px 5px;
        border-radius: 4px;
        background: #0d1016;
        border: 1px solid #262e3d;
        cursor: pointer;
      }
      .so101-port-pill:hover {
        border-color: #38bdf8;
        background: #162235;
      }
      .so101-port-pill.wiring-active {
        border-color: #fbbf24;
        background: #3b2a08;
        color: #fde68a;
      }
      .so101-port-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        display: inline-block;
      }
      .so101-node-footer {
        padding: 4px 8px;
        font-size: 10px;
        color: #93c5fd;
        background: #0d1016;
        border-top: 1px solid #1e2430;
        border-radius: 0 0 6px 6px;
        font-family: "JetBrains Mono", Consolas, monospace;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .so101-inspector-box {
        background: #11141c;
        border: 1px solid #283040;
        border-radius: 8px;
        padding: 12px 14px;
        height: 450px;
        overflow-y: auto;
        box-sizing: border-box;
      }
      .so101-journey-bar {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-bottom: 18px;
      }
      @media (max-width: 1100px) {
        .so101-journey-bar {
          grid-template-columns: 1fr;
        }
      }
      .so101-journey-step {
        background: #161920;
        border: 1px solid #262c3a;
        border-radius: 8px;
        padding: 10px 14px;
        cursor: pointer;
        text-align: left;
        transition: all 0.14s ease;
        display: flex;
        flex-direction: column;
        gap: 3px;
      }
      .so101-journey-step:hover {
        background: #1d222c;
        border-color: #3b4459;
      }
      .so101-journey-step.active {
        background: linear-gradient(180deg, #17253b 0%, #131c2e 100%);
        border-color: #38bdf8;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.16);
      }
      .so101-journey-title {
        font-size: 13px;
        font-weight: 700;
        color: #f8fafc;
      }
      .so101-journey-step.active .so101-journey-title {
        color: #38bdf8;
      }
      .so101-journey-sub {
        font-size: 11px;
        color: #94a3b8;
      }
      .so101-journey-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 18px;
        padding-top: 14px;
        border-top: 1px solid #232936;
      }
      .so101-segmented {
        display: inline-flex;
        background: #0e1117;
        border: 1px solid #262c3a;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        flex-wrap: wrap;
      }
      .so101-seg-btn {
        background: transparent;
        color: #94a3b8;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.12s ease;
      }
      .so101-seg-btn:hover {
        color: #e2e8f0;
      }
      .so101-seg-btn.active {
        background: #1e293b;
        color: #38bdf8;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3);
      }
      details.so101-details {
        background: #11141b;
        border: 1px solid #242a38;
        border-radius: 6px;
        padding: 8px 12px;
        margin-top: 10px;
      }
      details.so101-details > summary {
        cursor: pointer;
        font-size: 12px;
        font-weight: 600;
        color: #cbd5e1;
        user-select: none;
      }
    `;
    document.head.appendChild(style);
  }

  async function apiGet(path) {
    const res = await fetch(path, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error(`GET ${path} failed (${res.status})`);
    return res.json();
  }

  async function apiPost(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`POST ${path} failed: ${txt}`);
    }
    return res.json();
  }

  function downloadText(filename, text, mime) {
    const blob = new Blob([text], { type: mime || "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 150);
  }

  function ensureHeaderButtons() {
    injectStyles();
    if (!document.getElementById("so101-arch-toggle-btn")) {
      const btn = document.createElement("button");
      btn.id = "so101-arch-toggle-btn";
      btn.className = "so101-nav-btn floating-dock";
      btn.innerHTML = `<span>SO-101 Manipulation Studio</span>`;
      btn.onclick = () => toggleWorkbench();
      document.body.appendChild(btn);
    }
  }

  async function toggleWorkbench(forceOpen) {
    state.open = typeof forceOpen === "boolean" ? forceOpen : !state.open;
    const btn = document.getElementById("so101-arch-toggle-btn");
    if (btn) btn.classList.toggle("active", state.open);

    let overlay = document.getElementById("so101-workbench-overlay");
    if (!state.open) {
      if (overlay) overlay.style.display = "none";
      stopCamRefresh();
      stopTrajectoryPlayback();
      return;
    }

    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = "so101-workbench-overlay";
      document.body.appendChild(overlay);
    }
    overlay.style.display = "block";
    overlay.innerHTML = `<div style="padding:32px;text-align:center;color:#94a3b8;">Loading SO-101 Model-Based Architecture Workbench from Physical AI Studio backend...</div>`;

    try {
      const m = window.location.pathname.match(/\/projects\/([0-9a-fA-F-]{36})/);
      const pid = m ? m[1] : "";
      state.snap = await apiGet(`/api/so101/state${pid ? `?project_id=${pid}` : ""}`);
    } catch (err) {
      overlay.innerHTML = `<div class="so101-card" style="border-color:#ef4444;color:#fca5a5;">
        <h3>Failed to load /api/so101/state</h3>
        <pre>${String(err)}</pre>
      </div>`;
      return;
    }

    try {
      renderWorkbench();
      startCamRefresh();
    } catch (renderErr) {
      overlay.innerHTML = `<div class="so101-card" style="border-color:#ef4444;color:#fca5a5;">
        <h3>UI Render Error in SO-101 Architecture Workbench</h3>
        <pre>${String(renderErr && renderErr.stack ? renderErr.stack : renderErr)}</pre>
      </div>`;
    }
  }

  function startCamRefresh() {
    stopCamRefresh();
    state.camStreamTimer = setInterval(() => {
      if (!state.open || !state.camStreamActive) return;
      refreshInspectionEvidence();
      const ts = Date.now();
      ["top"].forEach((slot) => {
        const img = document.getElementById(`so101-cam-img-${slot}`);
        if (img) img.src = `/api/so101/camera/${slot}.jpg?t=${ts}`;
      });
    }, 1200);
  }

  let inspectionRequestActive = false;
  async function refreshInspectionEvidence() {
    if (inspectionRequestActive) return;
    inspectionRequestActive = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch("/api/so101/inspection", {cache: "no-store", signal: controller.signal});
      if (!response.ok) throw new Error("Inspection connection unavailable");
      const evidence = await response.json();
      for (const camera of ["plate", "wrist"]) {
        const result = evidence.cameras[camera];
        const image = document.getElementById(`so101-cam-img-${camera}`);
        const status = document.getElementById(`so101-inspection-${camera}`);
        if (!image || !status) continue;
        // The overlay and source must come from the same inspection packet.
        // Never put an old heatmap over a newer live-camera image.
        if (result && result.fresh && result.image_jpeg) {
          image.src = "data:image/jpeg;base64," + (result.overlay_jpeg || result.image_jpeg);
          image.alt = `${camera}: ${result.decision} — ${result.reason}`;
          status.textContent = `${result.decision} · ${result.reason} · Frame ${result.frame_id}`;
          status.style.color = result.decision === "DEFECTIVE" ? "#fca5a5" : "#fcd34d";
        } else {
          image.src = `/api/so101/camera/${camera}.jpg?t=${Date.now()}`;
          status.textContent = result ? `${result.decision} · ${result.reason}` : "Detector not connected";
          status.style.color = "#94a3b8";
        }
      }
    } catch (_error) {
      for (const camera of ["plate", "wrist"]) {
        const status = document.getElementById(`so101-inspection-${camera}`);
        const image = document.getElementById(`so101-cam-img-${camera}`);
        if (status) status.textContent = "Inspection unavailable · showing camera preview";
        if (image) image.src = `/api/so101/camera/${camera}.jpg?t=${Date.now()}`;
      }
    } finally {
      clearTimeout(timeout);
      inspectionRequestActive = false;
    }
  }

  function stopCamRefresh() {
    if (state.camStreamTimer) {
      clearInterval(state.camStreamTimer);
      state.camStreamTimer = null;
    }
  }

  function connectStudioLeaderWs() {
    if (state.leaderWs) {
      state.leaderWs.close();
      state.leaderWs = null;
      state.leaderWsConnected = false;
      renderWorkbench();
      return;
    }
    const db = (state.snap && state.snap.studio_db) || {};
    const proj = db.active_project;
    const robots = db.robots || [];
    const leader = robots.find((r) => r.type === "SO101_Leader" || r.name === "leader");
    if (!proj || !leader) {
      alert("No SO101_Leader robot found in the active Physical AI Studio project.");
      return;
    }
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/api/projects/${proj.id}/robots/${leader.id}/observations/ws?fps=15`;
    const ws = new WebSocket(wsUrl);
    state.leaderWs = ws;
    ws.onopen = () => {
      state.leaderWsConnected = true;
      updateWsStatusBadge();
    };
    ws.onmessage = async (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.event === "observation" && msg.data) {
          const norm5 = MOTOR_NAMES.map((m) => Number(msg.data[`${m}.pos`] ?? 0));
          const gripPct = Number(msg.data["gripper.pos"] ?? 22);
          const res = await apiPost("/api/so101/kinematics", {
            arm: "leader",
            normalized: norm5,
            gripper_pct: gripPct,
          });
          state.snap.arms = res.arms;
          if (state.activeArm === "leader") {
            state.snap.kinematics = res.kinematics;
            state.snap.dynamics = res.dynamics;
          }
          updateLiveReadoutsOnly();
        }
      } catch (_e) {
        // Ignore transient parse error
      }
    };
    ws.onclose = () => {
      state.leaderWsConnected = false;
      state.leaderWs = null;
      updateWsStatusBadge();
    };
  }

  function updateWsStatusBadge() {
    const el = document.getElementById("so101-ws-status-pill");
    if (!el) return;
    el.className = `so101-pill ${state.leaderWsConnected ? "ok" : "warn"}`;
    el.textContent = state.leaderWsConnected
      ? "STUDIO LEADER WS LIVE (/observations/ws)"
      : "STUDIO LEADER WS STANDBY";
  }

  function ensureComposerState() {
    if (!state.snap || !state.snap.composer) return;
    const comp = state.snap.composer;
    if (!state.activeGraph) {
      state.activeGraph = JSON.parse(JSON.stringify(comp.active_graph));
    }
    if (!state.compiledBundle) {
      state.compiledBundle = comp.compiled;
    }
    if (
      !state.selectedNodeId &&
      state.activeGraph &&
      state.activeGraph.nodes &&
      state.activeGraph.nodes.length > 0
    ) {
      state.selectedNodeId = state.activeGraph.nodes[0].id;
    }
  }

  function getBlockDef(typeName) {
    const lib = (state.snap && state.snap.composer && state.snap.composer.block_library) || [];
    return lib.find((b) => b.type === typeName) || null;
  }

  function renderComposerSectionHTML() {
    ensureComposerState();
    const comp = (state.snap && state.snap.composer) || {};
    const lib = comp.block_library || [];
    const presets = comp.presets || {};
    const portTypes = comp.port_types || {};
    const graph = state.activeGraph || { name: "Custom Architecture", nodes: [], edges: [] };
    const compiled = state.compiledBundle || {};
    const val = compiled.validation || { valid: true, errors: [], warnings: [], camera_count: 0 };

    // Group library by category for dropdown
    const byCat = {};
    lib.forEach((b) => {
      if (!byCat[b.category]) byCat[b.category] = [];
      byCat[b.category].push(b);
    });

    const nodesHTML = (graph.nodes || [])
      .map((node) => {
        const def = getBlockDef(node.type);
        if (!def) return "";
        const isSel = state.selectedNodeId === node.id;
        const paramSummary = Object.entries(node.params || {})
          .slice(0, 2)
          .map(([k, v]) => `${k}: ${Array.isArray(v) ? `[${v.join(",")}]` : v}`)
          .join(" · ");

        return `
          <div class="so101-node ${isSel ? "selected" : ""}" id="sim-node-${node.id}" data-node-id="${node.id}" style="left:${node.x || 40}px;top:${node.y || 60}px;">
            <div class="so101-node-hdr" style="background:${def.color || "#0284c7"};">
              <span>${def.icon || "⚙️"} ${def.title}</span>
              <span style="font-size:10px;opacity:0.85;cursor:pointer;" data-del-node="${node.id}" title="Delete Block">✕</span>
            </div>
            <div class="so101-node-sub" title="${def.subtitle}">${node.id} · ${def.subtitle}</div>
            <div class="so101-node-ports">
              <div class="so101-port-col">
                ${(def.inputs || [])
                  .map((p) => {
                    const pColor = (portTypes[p.type] && portTypes[p.type].color) || "#94a3b8";
                    return `
                      <div class="so101-port-pill" id="port-in-${node.id}-${p.id}" data-port-dir="in" data-node-id="${node.id}" data-port-id="${p.id}" data-port-type="${p.type}" title="Input (${p.type}): Click to connect wire">
                        <span class="so101-port-dot" style="background:${pColor};"></span>
                        <span>${p.label}</span>
                      </div>
                    `;
                  })
                  .join("")}
              </div>
              <div class="so101-port-col" style="align-items:flex-end;">
                ${(def.outputs || [])
                  .map((p) => {
                    const pColor = (portTypes[p.type] && portTypes[p.type].color) || "#38bdf8";
                    const isWiring =
                      state.wiringFrom &&
                      state.wiringFrom.nodeId === node.id &&
                      state.wiringFrom.portId === p.id;
                    return `
                      <div class="so101-port-pill ${isWiring ? "wiring-active" : ""}" id="port-out-${node.id}-${p.id}" data-port-dir="out" data-node-id="${node.id}" data-port-id="${p.id}" data-port-type="${p.type}" title="Output (${p.type}): Click to start wire">
                        <span>${p.label}</span>
                        <span class="so101-port-dot" style="background:${pColor};"></span>
                      </div>
                    `;
                  })
                  .join("")}
              </div>
            </div>
            <div class="so101-node-footer" title="${paramSummary}">${paramSummary || def.code_ref}</div>
          </div>
        `;
      })
      .join("");

    return `
      <div class="so101-composer-wrap">
        <div class="so101-card-header" style="border-bottom-color:#263042;margin-bottom:10px;">
          <div>
            <h3 style="font-size:15px;color:#f8fafc;display:flex;align-items:center;gap:10px;">
              <span>Architecture Block Composer</span>
              <span class="so101-pill ${val.valid ? "ok" : "warn"}">
                ${val.valid ? `Valid Graph (${val.node_count} blocks · ${val.edge_count} wires)` : `Validation Issue`}
              </span>
            </h3>
            <div style="font-size:12px;color:#94a3b8;margin-top:2px;">
              Wire vision, planning, and control blocks into a task-agnostic manipulation pipeline.
            </div>
          </div>
          <div style="display:flex;gap:8px;align-items:center;">
            <button class="so101-btn accent" id="btn-so101-compile-graph">Validate &amp; Compile</button>
          </div>
        </div>

        <div class="so101-toolbar">
          <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
            <label style="font-size:11.5px;color:#94a3b8;font-weight:600;">Preset:</label>
            <select id="sel-so101-preset" class="so101-input" style="width:280px;">
              ${Object.values(presets)
                .map(
                  (p) =>
                    `<option value="${p.id}" ${state.activePresetId === p.id ? "selected" : ""}>${p.name}</option>`
                )
                .join("")}
            </select>

            <label style="font-size:11.5px;color:#94a3b8;font-weight:600;margin-left:6px;">Block Library:</label>
            <select id="sel-so101-add-block" class="so101-input" style="width:250px;">
              ${Object.entries(byCat)
                .map(
                  ([cat, items]) =>
                    `<optgroup label="${cat}">${items
                      .map((it) => `<option value="${it.type}">${it.icon} ${it.title}</option>`)
                      .join("")}</optgroup>`
                )
                .join("")}
            </select>
            <button class="so101-btn primary" id="btn-so101-add-block">+ Add Block</button>
          </div>

          <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
            ${
              state.wiringFrom
                ? `<span class="so101-pill warn">Connecting ${state.wiringFrom.nodeId}.${state.wiringFrom.portId} (${state.wiringFrom.portType}) — Select matching input or <u id="btn-cancel-wire" style="cursor:pointer;">Cancel</u></span>`
                : `<span style="font-size:11px;color:#94a3b8;">Drag blocks to arrange · Click an output port then a matching input port to wire</span>`
            }
          </div>
        </div>

        ${
          val.errors && val.errors.length > 0
            ? `<div style="background:#351318;border:1px solid #ef4444;color:#fecaca;padding:8px 12px;border-radius:6px;margin-bottom:10px;font-size:12px;">
                <strong>Graph Validation:</strong> ${val.errors.join(" | ")}
              </div>`
            : ""
        }

        <div class="so101-canvas-layout">
          <div class="so101-simulink-viewport" id="so101-simulink-viewport">
            <div class="so101-simulink-stage" id="so101-simulink-stage">
              <svg class="so101-simulink-wires" id="so101-simulink-wires"></svg>
              ${nodesHTML}
            </div>
          </div>
          <div class="so101-inspector-box" id="so101-simulink-inspector">
            ${renderSimulinkInspectorHTML()}
          </div>
        </div>

        <details class="so101-details">
          <summary>Signal Port Types (Matching colors connect)</summary>
          <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;font-size:11px;align-items:center;">
            ${Object.entries(portTypes)
              .map(
                ([_k, pt]) => `
                <span style="display:inline-flex;align-items:center;gap:5px;color:#cbd5e1;" title="${pt.desc}">
                  <span class="so101-port-dot" style="background:${pt.color};"></span>
                  <span>${pt.label}</span>
                </span>
              `
              )
              .join("")}
          </div>
        </details>
      </div>
    `;
  }

  function renderSimulinkInspectorHTML() {
    const graph = state.activeGraph || { nodes: [], edges: [] };
    const node = (graph.nodes || []).find((n) => n.id === state.selectedNodeId);
    if (!node) {
      return `<div style="color:#94a3b8;font-size:12px;">Select any block on the canvas to inspect its parameters and connected signal wires.</div>`;
    }
    const def = getBlockDef(node.type);
    if (!def) return "";

    const connectedEdges = (graph.edges || []).filter(
      (e) => e.from_node === node.id || e.to_node === node.id
    );

    const paramFieldsHTML = Object.entries(def.params || {})
      .map(([pKey, schema]) => {
        const curVal =
          node.params && node.params[pKey] !== undefined ? node.params[pKey] : schema.default;
        let inputEl = "";
        if (schema.type === "select") {
          inputEl = `<select class="so101-input" data-node-param="${pKey}">
            ${(schema.options || [])
              .map((opt) => `<option value="${opt}" ${String(curVal) === String(opt) ? "selected" : ""}>${opt}</option>`)
              .join("")}
          </select>`;
        } else if (schema.type === "bool") {
          inputEl = `<select class="so101-input" data-node-param="${pKey}" data-param-type="bool">
            <option value="true" ${curVal ? "selected" : ""}>true</option>
            <option value="false" ${!curVal ? "selected" : ""}>false</option>
          </select>`;
        } else if (schema.type === "number") {
          inputEl = `<input type="number" step="any" class="so101-input" data-node-param="${pKey}" data-param-type="number" value="${curVal}" />`;
        } else {
          const strVal = Array.isArray(curVal) ? JSON.stringify(curVal) : String(curVal);
          inputEl = `<input type="text" class="so101-input" data-node-param="${pKey}" data-param-type="${schema.type}" value='${strVal}' />`;
        }
        return `
          <div style="margin-bottom:8px;">
            <label style="font-size:11px;color:#94a3b8;display:block;margin-bottom:2px;">${schema.label || pKey}</label>
            ${inputEl}
          </div>
        `;
      })
      .join("");

    return `
      <div style="border-bottom:1px solid #262e3d;padding-bottom:8px;margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span class="so101-pill" style="background:${def.color};color:#fff;border:none;">${def.category}</span>
          <code class="so101-mono" style="font-size:11px;color:#94a3b8;">${node.id}</code>
        </div>
        <h4 style="margin:6px 0 3px 0;font-size:14px;color:#f8fafc;">${def.icon} ${def.title}</h4>
        <div style="font-size:11.5px;color:#94a3b8;line-height:1.4;">${def.description}</div>
      </div>

      <div>
        <strong style="font-size:12px;color:#f1f5f9;display:block;margin-bottom:6px;">Block Parameters</strong>
        ${paramFieldsHTML}
        <button class="so101-btn primary" id="btn-so101-save-node-params" style="width:100%;margin-top:4px;">Apply Parameters</button>
      </div>

      <div style="margin-top:12px;padding-top:10px;border-top:1px solid #262e3d;">
        <strong style="font-size:11.5px;color:#94a3b8;display:block;margin-bottom:6px;">Connected Wires</strong>
        ${
          connectedEdges.length === 0
            ? `<div style="font-size:11px;color:#64748b;">No wires attached yet.</div>`
            : connectedEdges
                .map(
                  (e) => `
                <div style="display:flex;justify-content:space-between;align-items:center;background:#0b0e14;padding:4px 8px;border-radius:4px;margin-bottom:4px;font-size:11px;" class="so101-mono">
                  <span>${e.from_node}.${e.from_port} → ${e.to_node}.${e.to_port}</span>
                  <span style="color:#f87171;cursor:pointer;" data-del-edge="${e.from_node}|${e.from_port}|${e.to_node}|${e.to_port}" title="Remove Wire">✕</span>
                </div>
              `
                )
                .join("")
        }
      </div>
    `;
  }

  function ensureCustomSkillsOnGraph() {
    if (!state.activeGraph) return [];
    if (!Array.isArray(state.activeGraph.custom_skills) || state.activeGraph.custom_skills.length === 0) {
      const compiled = state.compiledBundle || {};
      const trace = compiled.symbolic_plan_trace || [
        {
          step: 1,
          operator: "LocalizeObject(obj_0)",
          preconditions: ["HandEmpty()"],
          expected_atoms: ["ObjectLocalized(obj_0)"],
          waypoint: "observation_home",
          gripper_pct: 22.0,
        },
        {
          step: 2,
          operator: "ApproachAndGrasp(obj_0)",
          preconditions: ["HandEmpty()", "ObjectLocalized(obj_0)"],
          expected_atoms: ["ObjectHeld(obj_0)"],
          waypoint: "source_pick",
          gripper_pct: 3.5,
        },
        {
          step: 3,
          operator: "PlaceAtGoal(obj_0)",
          preconditions: ["ObjectHeld(obj_0)"],
          expected_atoms: ["ObjectPlaced(obj_0)", "HandEmpty()"],
          waypoint: "good_bowl_drop",
          gripper_pct: 24.0,
        },
        {
          step: 4,
          operator: "ParkAtRest(obj_0)",
          preconditions: ["HandEmpty()"],
          expected_atoms: ["ArmAtRest()"],
          waypoint: "observation_home",
          gripper_pct: 22.0,
        },
      ];
      state.activeGraph.custom_skills = trace.map((st) => ({
        operator: st.operator,
        preconditions: Array.isArray(st.preconditions) ? st.preconditions.join(", ") : String(st.preconditions || ""),
        expected_atoms: Array.isArray(st.expected_atoms) ? st.expected_atoms.join(", ") : String(st.expected_atoms || ""),
        waypoint: st.waypoint || "observation_home",
        gripper_pct: st.gripper_pct !== undefined ? Number(st.gripper_pct) : 22.0,
      }));
    }
    return state.activeGraph.custom_skills;
  }

  function renderSymSkillTabHTML() {
    const skills = ensureCustomSkillsOnGraph();
    const compiled = state.compiledBundle || {};
    const spec = compiled.architecture_spec || {};
    return `
      <div class="so101-card">
        <div class="so101-card-header">
          <div>
            <h3>Symbolic Skill Builder</h3>
            <div style="font-size:11.5px;color:#94a3b8;margin-top:2px;">Name, add, reorder, and customize your symbolic operators, preconditions, and waypoints.</div>
          </div>
          <div style="display:flex;gap:6px;">
            <button class="so101-btn primary" id="btn-so101-add-skill">+ Add Skill</button>
            <button class="so101-btn accent" id="btn-so101-save-skills">Save &amp; Compile</button>
          </div>
        </div>
        <table class="so101-table so101-mono">
          <thead>
            <tr>
              <th>Skill / Operator</th>
              <th>Preconditions</th>
              <th>Expected Effects</th>
              <th style="width:155px;">Target Waypoint</th>
              <th style="width:72px;">Grip %</th>
              <th style="width:86px;">Order</th>
            </tr>
          </thead>
          <tbody>
            ${skills
              .map(
                (sk, idx) => `
              <tr data-skill-row="${idx}">
                <td>
                  <input type="text" class="so101-input" data-sk-field="operator" data-sk-idx="${idx}" value="${String(sk.operator || "").replace(/"/g, "&quot;")}" placeholder="e.g. GraspObject(obj)" style="color:#fb923c;font-weight:700;" />
                </td>
                <td>
                  <input type="text" class="so101-input" data-sk-field="preconditions" data-sk-idx="${idx}" value="${String(sk.preconditions || "").replace(/"/g, "&quot;")}" placeholder="HandEmpty(), Localized(obj)" />
                </td>
                <td>
                  <input type="text" class="so101-input" data-sk-field="expected_atoms" data-sk-idx="${idx}" value="${String(sk.expected_atoms || "").replace(/"/g, "&quot;")}" placeholder="ObjectHeld(obj)" style="color:#34d399;" />
                </td>
                <td>
                  <select class="so101-input" data-sk-field="waypoint" data-sk-idx="${idx}">
                    ${WAYPOINT_ORDER.map(
                      (wk) => `<option value="${wk}" ${sk.waypoint === wk ? "selected" : ""}>${wk}</option>`
                    ).join("")}
                    <option value="good_bowl_drop / defective_bowl_drop" ${sk.waypoint.includes("/") ? "selected" : ""}>branch: target_a / target_b</option>
                  </select>
                </td>
                <td>
                  <input type="number" step="0.5" min="0" max="100" class="so101-input" data-sk-field="gripper_pct" data-sk-idx="${idx}" value="${Number(sk.gripper_pct ?? 22)}" />
                </td>
                <td>
                  <div style="display:flex;gap:4px;">
                    <button class="so101-btn" data-sk-up="${idx}" style="padding:2px 6px;font-size:10px;" title="Move Up">↑</button>
                    <button class="so101-btn" data-sk-down="${idx}" style="padding:2px 6px;font-size:10px;" title="Move Down">↓</button>
                    <button class="so101-btn" data-sk-del="${idx}" style="padding:2px 6px;font-size:10px;color:#f87171;" title="Delete Skill">✕</button>
                  </div>
                </td>
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
        <details class="so101-details">
          <summary>Compiled Architecture &amp; Skill Specification (JSON)</summary>
          <pre class="so101-matrix so101-mono" style="max-height:240px;font-size:11px;color:#cbd5e1;margin-top:8px;">${JSON.stringify(
            {
              name: spec.name,
              cameras_configured: (spec.cameras || []).length,
              computer_vision_blocks: (spec.computer_vision || []).map((c) => c.type),
              control_architecture: (spec.control_architecture || []).map((c) => c.type),
              compiled_skill_plan: compiled.symbolic_plan_trace || [],
            },
            null,
            2
          )}</pre>
        </details>
      </div>
    `;
  }

  function renderCompiledCodeTabHTML() {
    const compiled = state.compiledBundle || {};
    const code = compiled.compiled_python_policy || "# Click 'Validate & Compile' to generate standalone Python script";
    return `
      <div class="so101-card">
        <div class="so101-card-header">
          <div>
            <h3>Compiled Standalone Policy (<code>so101_compiled_policy.py</code>)</h3>
            <div style="font-size:11.5px;color:#94a3b8;margin-top:2px;">Self-contained Python policy generated from your block graph and custom skills.</div>
          </div>
          <div style="display:flex;gap:8px;">
            <button class="so101-btn" id="btn-so101-copy-compiled">Copy Code</button>
            <button class="so101-btn success" id="btn-so101-dl-compiled-py">Download Policy (.py)</button>
          </div>
        </div>
        <pre class="so101-matrix so101-mono" style="max-height:420px;font-size:11.5px;color:#e2e8f0;margin:0;line-height:1.45;">${code
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")}</pre>
      </div>
    `;
  }

  function drawSimulinkWires() {
    const svg = document.getElementById("so101-simulink-wires");
    const stage = document.getElementById("so101-simulink-stage");
    if (!svg || !stage || !state.activeGraph) return;

    const stageRect = stage.getBoundingClientRect();
    const portTypes = (state.snap && state.snap.composer && state.snap.composer.port_types) || {};
    const paths = [];

    (state.activeGraph.edges || []).forEach((e) => {
      const outEl = document.getElementById(`port-out-${e.from_node}-${e.from_port}`);
      const inEl = document.getElementById(`port-in-${e.to_node}-${e.to_port}`);
      if (!outEl || !inEl) return;

      const r1 = outEl.getBoundingClientRect();
      const r2 = inEl.getBoundingClientRect();
      const x1 = r1.right - stageRect.left;
      const y1 = r1.top + r1.height / 2 - stageRect.top;
      const x2 = r2.left - stageRect.left;
      const y2 = r2.top + r2.height / 2 - stageRect.top;

      const dx = Math.max(Math.abs(x2 - x1) * 0.48, 48);
      const pType = outEl.getAttribute("data-port-type") || "IMAGE_FRAME";
      const stroke = (portTypes[pType] && portTypes[pType].color) || "#38bdf8";

      paths.push(
        `<path d="M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}" fill="none" stroke="${stroke}" stroke-width="2.6" opacity="0.92" />` +
          `<circle cx="${x1}" cy="${y1}" r="3.5" fill="${stroke}" />` +
          `<circle cx="${x2}" cy="${y2}" r="3.5" fill="${stroke}" />`
      );
    });

    svg.innerHTML = paths.join("");
  }

  function renderWaypointsCardHTML() {
    const s = state.snap;
    return `
      <div class="so101-card">
        <div class="so101-card-header">
          <div>
            <h3>Task Waypoints &amp; Target Coordinates</h3>
            <div style="font-size:11.5px;color:#94a3b8;margin-top:2px;">Preview joint poses, teach directly from the Leader arm, or solve from XYZ coordinates.</div>
          </div>
          <span class="so101-pill ok">Limits Verified: ${s.trajectory.all_within_joint_limits ? "Yes" : "Check"}</span>
        </div>
        <div class="so101-wp-grid">
          ${WAYPOINT_ORDER.map((wpKey) => {
            const wp = s.waypoints[wpKey];
            return `
              <div class="so101-wp-card">
                <h4>${wp.label.replace(/^\d+\.\s*/, "")}</h4>
                <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">${wp.description}</div>
                <div class="so101-mono" style="font-size:11px;color:#e2e8f0;margin-bottom:6px;">
                  rad: [${wp.q_enc_rad.map((v) => Number(v).toFixed(2)).join(", ")}] · Grip: ${wp.gripper_pct}%
                </div>
                <div style="display:flex;gap:6px;">
                  <button class="so101-btn" data-wp-load="${wpKey}" style="padding:3px 8px;font-size:11px;">Preview Pose</button>
                  <button class="so101-btn primary" data-wp-teach="${wpKey}" style="padding:3px 8px;font-size:11px;">Teach from Leader</button>
                </div>
              </div>
            `;
          }).join("")}
        </div>

        <details class="so101-details">
          <summary>Solve Waypoints from World XYZ Target Coordinates (meters)</summary>
          <div style="margin-top:10px;">
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:11.5px;margin-bottom:8px;">
              ${[
                { key: "source", label: "Pick Target (XYZ)" },
                { key: "good", label: "Place Target A (XYZ)" },
                { key: "defective", label: "Place Target B (XYZ)" },
              ]
                .map(({ key: bKey, label }) => {
                  const b = s.bowls[bKey];
                  return `
                  <div style="background:#0d1016;padding:6px 8px;border-radius:4px;border:1px solid #242936;">
                    <strong style="color:#f8fafc;">${label}</strong>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:4px;">
                      <input class="so101-input" id="bowl-${bKey}-x" type="number" step="0.005" value="${b.pos_world_m[0]}" title="X (m)" />
                      <input class="so101-input" id="bowl-${bKey}-y" type="number" step="0.005" value="${b.pos_world_m[1]}" title="Y (m)" />
                      <input class="so101-input" id="bowl-${bKey}-z" type="number" step="0.005" value="${b.pos_world_m[2]}" title="Z (m)" />
                    </div>
                  </div>
                `;
                })
                .join("")}
            </div>
            <button class="so101-btn primary" id="btn-so101-solve-bowls">Solve Target Waypoints via Inverse Kinematics</button>
          </div>
        </details>
      </div>
    `;
  }

  function renderLiveTwinStageHTML() {
    const s = state.snap;
    const db = s.studio_db || {};
    const studioCams = db.cameras || [];
    const kin = s.kinematics;
    const cal = s.calibration;

    const activeCamNodes = ((state.activeGraph && state.activeGraph.nodes) || []).filter(
      (n) => n.type === "camera_source"
    );
    const camRoleToSlot = (role) => {
      if (role === "eye_in_hand_wrist") return "wrist";
      if (role === "side_plate") return "plate";
      return "top";
    };

    return `
      <div class="so101-grid-2">
        <div class="so101-card">
          <div class="so101-card-header">
            <h3>3D Kinematic Arm Skeleton</h3>
            <div style="display:flex;gap:8px;align-items:center;">
              <span class="so101-pill" id="so101-3d-phase-pill">TCP: [${kin.tcp_world_xyz_mm.join(", ")}] mm</span>
              <button class="so101-btn accent" id="btn-so101-play-traj" style="padding:4px 10px;font-size:11.5px;">${state.playingTraj ? "Stop Trajectory" : "Play Skill Trajectory"}</button>
            </div>
          </div>
          <canvas id="so101-3d-canvas" width="620" height="275" style="width:100%;height:275px;background:#0b0e13;border-radius:6px;border:1px solid #242936;cursor:grab;"></canvas>
          <div style="margin-top:8px;font-size:11px;color:#94a3b8;display:flex;justify-content:space-between;">
            <span>Drag to orbit · Scroll to zoom</span>
            <span>World Frame: +X forward, +Y left, +Z up</span>
          </div>
        </div>

        <div class="so101-card">
          <div class="so101-card-header">
            <h3>Leader &amp; Follower Arm Control</h3>
            <div style="display:flex;gap:6px;align-items:center;">
              <span id="so101-ws-status-pill" class="so101-pill ${state.leaderWsConnected ? "ok" : "warn"}">
                ${state.leaderWsConnected ? "Leader Stream Live" : "Leader Standby"}
              </span>
              <button class="so101-btn" id="btn-so101-leader-ws" style="padding:4px 10px;font-size:11.5px;">
                ${state.leaderWsConnected ? "Disconnect Leader" : "Stream Leader Arm"}
              </button>
            </div>
          </div>

          <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;align-items:center;">
            <select id="sel-so101-arm" class="so101-input" style="width:165px;">
              <option value="follower" ${state.activeArm === "follower" ? "selected" : ""}>Follower Arm</option>
              <option value="leader" ${state.activeArm === "leader" ? "selected" : ""}>Leader Arm</option>
            </select>
            <select id="sel-so101-unit" class="so101-input" style="width:195px;">
              <option value="q_enc_rad" ${state.unitMode === "q_enc_rad" ? "selected" : ""}>Encoder Radians</option>
              <option value="ticks" ${state.unitMode === "ticks" ? "selected" : ""}>Servo Ticks (0..4095)</option>
              <option value="lerobot_deg" ${state.unitMode === "lerobot_deg" ? "selected" : ""}>LeRobot Degrees</option>
              <option value="normalized" ${state.unitMode === "normalized" ? "selected" : ""}>Studio Normalized</option>
            </select>
            <button class="so101-btn primary" id="btn-so101-map-l2f" style="padding:5px 10px;font-size:11.5px;">Map Leader → Follower</button>
          </div>

          <div id="so101-joint-sliders-container">
            ${renderJointSlidersHTML()}
          </div>

          <div style="margin-top:8px;padding-top:8px;border-top:1px solid #242936;font-size:11px;color:#94a3b8;">
            <span class="so101-mono" id="so101-l2f-readout">
              Mapped Follower Ticks: [${s.arms.leader_to_follower_mapping.follower_ticks.join(", ")}] · Gripper: ${s.arms.leader_to_follower_mapping.follower_gripper_tick} (min ${cal.safe_gripper_min_raw})
            </span>
          </div>
        </div>
      </div>

      <div class="so101-card">
        <div class="so101-card-header">
          <h3>Live Camera Feeds</h3>
          <div style="display:flex;gap:8px;align-items:center;">
            <span class="so101-pill">${studioCams.length} Hardware Camera${studioCams.length === 1 ? "" : "s"} Detected</span>
            <button class="so101-btn" id="btn-so101-cam-toggle" style="padding:4px 10px;font-size:11.5px;">${state.camStreamActive ? "Pause Feeds" : "Resume Feeds"}</button>
          </div>
        </div>
        ${
          activeCamNodes.length === 0
            ? `<div style="padding:24px;text-align:center;background:#0f1217;border:1px dashed #2d3546;border-radius:6px;color:#94a3b8;font-size:12.5px;">
                No camera blocks in the active architecture graph. Add a <em>Camera Stream Source</em> block in the Architecture view to enable live feeds.
              </div>`
            : `<div class="so101-cam-grid" style="grid-template-columns:repeat(${Math.min(activeCamNodes.length, 3)}, 1fr);">
                ${activeCamNodes
                  .map((cn) => {
                    const p = cn.params || {};
                    const slot = camRoleToSlot(p.camera_role);
                    return `
                      <div class="so101-cam-box">
                        <img class="so101-live-cam-img" data-cam-slot="${slot}" id="so101-cam-img-${slot}" src="/api/so101/camera/${slot}.jpg?t=${Date.now()}" alt="${cn.id}" />
                        <div class="so101-cam-meta">
                          <div>
                            <strong style="color:#f1f5f9;">${cn.id} (${p.camera_role || "camera"})</strong><br/>
                            <span>${p.driver || "uvc"} · ${p.serial_or_dev || "auto"} · ${p.resolution || "640x480"}</span>
                            ${slot !== "top" ? `<div id="so101-inspection-${slot}" style="color:#94a3b8">Detector not connected</div>` : ""}
                          </div>
                        </div>
                      </div>
                    `;
                  })
                  .join("")}
              </div>`
        }
      </div>
    `;
  }

  function renderMathAndCalibrationStageHTML() {
    const s = state.snap;
    const kin = s.kinematics;
    const dyn = s.dynamics;
    const cal = s.calibration;

    return `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:10px;">
        <div class="so101-segmented">
          <button class="so101-seg-btn ${state.mathTab === "fk_ik" ? "active" : ""}" data-math-tab="fk_ik">Forward &amp; Inverse Kinematics</button>
          <button class="so101-seg-btn ${state.mathTab === "jacobian" ? "active" : ""}" data-math-tab="jacobian">Geometric Jacobian</button>
          <button class="so101-seg-btn ${state.mathTab === "dynamics" ? "active" : ""}" data-math-tab="dynamics">CAD Dynamics &amp; Gravity</button>
          <button class="so101-seg-btn ${state.mathTab === "calibration" ? "active" : ""}" data-math-tab="calibration">Arm Calibration &amp; Zeroing</button>
        </div>
      </div>

      ${
        state.mathTab === "fk_ik"
          ? `
        <div class="so101-grid-2">
          <div class="so101-card">
            <div class="so101-card-header">
              <h3>Forward Kinematics</h3>
              <select id="sel-so101-fk-frame" class="so101-input" style="width:150px;">
                <option value="world" ${state.fkFrame === "world" ? "selected" : ""}>World Frame</option>
                <option value="base" ${state.fkFrame === "base" ? "selected" : ""}>Base Frame</option>
              </select>
            </div>
            <div id="so101-fk-panel">
              ${renderFKPanelHTML()}
            </div>
          </div>

          <div class="so101-card">
            <div class="so101-card-header">
              <h3>Grasp &amp; Placement Inverse Kinematics Solver</h3>
              <span id="so101-ik-status" class="so101-pill ${!state.lastIkResult || state.lastIkResult.feasible ? "ok" : "warn"}">
                ${state.lastIkResult ? `Error: ${state.lastIkResult.position_error_mm} mm (${state.lastIkResult.feasible ? "Feasible" : "Limit"})` : "Ready"}
              </span>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:6px;">
              <div>
                <label style="font-size:11px;color:#94a3b8;">Target X (m)</label>
                <input class="so101-input" id="ik-tgt-x" type="number" step="0.005" value="${kin.tcp_world_xyz_m[0]}" />
              </div>
              <div>
                <label style="font-size:11px;color:#94a3b8;">Target Y (m)</label>
                <input class="so101-input" id="ik-tgt-y" type="number" step="0.005" value="${kin.tcp_world_xyz_m[1]}" />
              </div>
              <div>
                <label style="font-size:11px;color:#94a3b8;">Target Z (m)</label>
                <input class="so101-input" id="ik-tgt-z" type="number" step="0.005" value="${kin.tcp_world_xyz_m[2]}" />
              </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1.2fr;gap:8px;margin-top:10px;">
              <div>
                <label style="font-size:11px;color:#94a3b8;">Pitch Down (°)</label>
                <input class="so101-input" id="ik-pitch-deg" type="number" step="5" value="90" />
              </div>
              <div>
                <label style="font-size:11px;color:#94a3b8;">Jaw Yaw (°)</label>
                <input class="so101-input" id="ik-yaw-deg" type="number" step="5" value="0" />
              </div>
              <div>
                <label style="font-size:11px;color:#94a3b8;">Assign to Waypoint</label>
                <select class="so101-input" id="ik-save-wp">
                  <option value="">(Preview Only)</option>
                  ${WAYPOINT_ORDER.map((k) => `<option value="${k}">${k}</option>`).join("")}
                </select>
              </div>
            </div>
            <div style="margin-top:14px;">
              <button class="so101-btn primary" id="btn-so101-solve-ik">Solve Inverse Kinematics &amp; Apply</button>
            </div>
          </div>
        </div>
      `
          : state.mathTab === "jacobian"
            ? `
        <div class="so101-card">
          <div class="so101-card-header">
            <h3>Geometric Jacobian &amp; Manipulability</h3>
            <select id="sel-so101-jac-mode" class="so101-input" style="width:190px;">
              <option value="J_world_rad" ${state.jacMode === "J_world_rad" ? "selected" : ""}>World Frame (m/rad)</option>
              <option value="J_base_rad" ${state.jacMode === "J_base_rad" ? "selected" : ""}>Base Frame (m/rad)</option>
              <option value="J_world_tick" ${state.jacMode === "J_world_tick" ? "selected" : ""}>World Frame (m/tick)</option>
              <option value="J_world_lerobot_deg" ${state.jacMode === "J_world_lerobot_deg" ? "selected" : ""}>World Frame (LeRobot deg)</option>
            </select>
          </div>
          <div id="so101-jacobian-panel">
            ${renderJacobianPanelHTML()}
          </div>
        </div>
      `
            : state.mathTab === "dynamics"
              ? `
        <div class="so101-card">
          <div class="so101-card-header">
            <h3>CAD Mass Matrix, Center of Mass &amp; Gravity Compensation</h3>
            <span class="so101-pill">${dyn.total_mass_g} g Total Mass (Arm 632g + Payload ${dyn.payload_mass_g}g)</span>
          </div>
          <div id="so101-dynamics-panel">
            ${renderDynamicsPanelHTML()}
          </div>
        </div>
      `
              : `
        <div class="so101-card">
          <div class="so101-card-header">
            <h3>Arm Calibration &amp; Upright Reference Pose Alignment</h3>
            <span class="so101-pill">Upright Reference Pose: [0°, +93.1°, -89.5°, +1.5°, +90°] CAD</span>
          </div>
          <div class="so101-grid-2" style="margin-bottom:0;">
            <div>
              <table class="so101-table so101-mono">
                <thead>
                  <tr>
                    <th>Joint</th>
                    <th>CAD Joint</th>
                    <th>Range Ticks</th>
                    <th>Zero Tick</th>
                    <th>Homing Off</th>
                    <th>Limits (rad)</th>
                    <th>CAD@Zero (°)</th>
                    <th>Sign</th>
                  </tr>
                </thead>
                <tbody>
                  ${cal.arms[state.activeArm].joints
                    .map(
                      (j) => `
                    <tr>
                      <td>${j.motor_name}</td>
                      <td>${j.cad_joint}</td>
                      <td>[${j.range_min_tick}, ${j.range_max_tick}]</td>
                      <td>${j.zero_ticks}</td>
                      <td>${j.homing_offset}</td>
                      <td>[${j.lower_enc_rad.toFixed(2)}, ${j.upper_enc_rad.toFixed(2)}]</td>
                      <td>${j.cad_at_zero_deg.toFixed(2)}°</td>
                      <td>${j.cad_sign > 0 ? "+1" : "-1"}</td>
                    </tr>
                  `
                    )
                    .join("")}
                </tbody>
              </table>
            </div>
            <div>
              <div style="font-size:12px;color:#cbd5e1;margin-bottom:8px;">
                Place the arm in the upright visual reference pose (<code>[0, 93.1, -89.5, 1.5, 90] deg</code>), enter its homed STS3215 ticks below, and derive its CAD zero alignment:
              </div>
              <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:8px;">
                ${kin.ticks
                  .map(
                    (t, idx) => `
                  <div>
                    <label style="font-size:10.5px;color:#94a3b8;">${MOTOR_NAMES[idx]}</label>
                    <input class="so101-input" id="align-tick-${idx}" type="number" value="${t}" />
                  </div>
                `
                  )
                  .join("")}
              </div>
              <div style="display:flex;gap:8px;align-items:center;">
                <button class="so101-btn primary" id="btn-so101-run-align">Derive Zero Alignment</button>
                <button class="so101-btn" id="btn-so101-use-cur-ticks">Use Current Arm Ticks</button>
              </div>
              <pre id="so101-align-output" class="so101-matrix so101-mono" style="margin-top:8px;max-height:110px;font-size:11px;color:#93c5fd;">${
                state.lastAlignResult
                  ? JSON.stringify(state.lastAlignResult.alignment_snippet, null, 2)
                  : "// Click 'Derive Zero Alignment' to compute calibration parameters"
              }</pre>
            </div>
          </div>
        </div>
      `
      }
    `;
  }

  function renderDeployStageHTML() {
    return `
      <div class="so101-card" style="margin-bottom:16px;">
        <div class="so101-card-header">
          <div>
            <h3>Export Artifacts &amp; Hardware Bundles</h3>
            <div style="font-size:11.5px;color:#94a3b8;margin-top:2px;">Download standalone Python policies, calibrated URDF models, or the complete architecture bundle.</div>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;">
            <button class="so101-btn primary" id="btn-so101-dl-py">Download Controller (.py)</button>
            <button class="so101-btn" id="btn-so101-dl-urdf">Download Calibrated URDF (.urdf)</button>
            <button class="so101-btn" id="btn-so101-dl-json">Download Full Bundle (.json)</button>
          </div>
        </div>
      </div>
      ${renderCompiledCodeTabHTML()}
    `;
  }

  function renderJourneyFooterHTML() {
    const idx = JOURNEY_STEPS.findIndex((s) => s.id === state.journeyStep);
    const prev = idx > 0 ? JOURNEY_STEPS[idx - 1] : null;
    const next = idx >= 0 && idx < JOURNEY_STEPS.length - 1 ? JOURNEY_STEPS[idx + 1] : null;
    return `
      <div class="so101-journey-footer">
        <div>
          ${
            prev
              ? `<button class="so101-btn" data-journey-step="${prev.id}">← ${prev.label}</button>`
              : `<span></span>`
          }
        </div>
        <div style="font-size:12px;color:#94a3b8;">
          ${JOURNEY_STEPS[idx] ? `${JOURNEY_STEPS[idx].label} — ${JOURNEY_STEPS[idx].summary}` : ""}
        </div>
        <div>
          ${
            next
              ? `<button class="so101-btn primary" data-journey-step="${next.id}">Continue to ${next.label} →</button>`
              : `<span></span>`
          }
        </div>
      </div>
    `;
  }

  function renderWorkbench() {
    const overlay = document.getElementById("so101-workbench-overlay");
    if (!overlay || !state.snap) return;

    const s = state.snap;
    const db = s.studio_db || {};
    const proj = db.active_project;

    let stageHTML = "";
    if (state.journeyStep === "architecture") {
      stageHTML = renderComposerSectionHTML();
    } else if (state.journeyStep === "skills") {
      stageHTML = `
        <div class="so101-grid-2">
          ${renderSymSkillTabHTML()}
          ${renderWaypointsCardHTML()}
        </div>
      `;
    } else if (state.journeyStep === "live") {
      stageHTML = renderLiveTwinStageHTML();
    } else if (state.journeyStep === "math") {
      stageHTML = renderMathAndCalibrationStageHTML();
    } else {
      stageHTML = renderDeployStageHTML();
    }

    overlay.innerHTML = `
      <div class="so101-header-bar">
        <div class="so101-title-block">
          <h2>
            <span>SO-101 Manipulation Studio</span>
            <span class="so101-pill ok">Model-Based</span>
            <span class="so101-pill">${proj ? proj.name : "Default Workspace"}</span>
          </h2>
          <p>Compose vision, symbolic skills, kinematics, and control into a standalone robot policy.</p>
        </div>
        <div class="so101-btn-row">
          <button class="so101-btn" id="btn-so101-sync">Sync Hardware DB</button>
          <button class="so101-btn" id="btn-so101-close">Return to Studio</button>
        </div>
      </div>

      <div class="so101-journey-bar" role="tablist">
        ${JOURNEY_STEPS.map(
          (st) => `
          <button class="so101-journey-step ${state.journeyStep === st.id ? "active" : ""}" data-journey-step="${st.id}" role="tab">
            <span class="so101-journey-title">${st.label}</span>
            <span class="so101-journey-sub">${st.summary}</span>
          </button>
        `
        ).join("")}
      </div>

      ${stageHTML}
      ${renderJourneyFooterHTML()}
    `;

    bindWorkbenchEvents();
    bindComposerEvents();
    draw3DCanvas();
    setTimeout(drawSimulinkWires, 30);
  }

  function renderJointSlidersHTML() {
    const s = state.snap;
    const calArm = s.calibration.arms[state.activeArm];
    const kin = s.kinematics;
    const normArr = kin.normalized || [0, 0, 0, 0, 0];

    const rows = calArm.joints.map((j, idx) => {
      let minVal, maxVal, stepVal, curVal, unitSuffix;
      if (state.unitMode === "ticks") {
        minVal = j.range_min_tick;
        maxVal = j.range_max_tick;
        stepVal = 1;
        curVal = kin.ticks[idx];
        unitSuffix = "ticks";
      } else if (state.unitMode === "lerobot_deg") {
        minVal = j.lower_lerobot_deg;
        maxVal = j.upper_lerobot_deg;
        stepVal = 0.5;
        curVal = kin.lerobot_deg[idx];
        unitSuffix = "deg";
      } else if (state.unitMode === "normalized") {
        minVal = -100;
        maxVal = 100;
        stepVal = 0.5;
        curVal = normArr[idx] ?? 0;
        unitSuffix = "norm";
      } else {
        minVal = j.lower_enc_rad;
        maxVal = j.upper_enc_rad;
        stepVal = 0.01;
        curVal = kin.q_enc_rad[idx];
        unitSuffix = "rad";
      }

      return `
        <div class="so101-slider-row">
          <span class="so101-mono" style="color:#e2e8f0;">${j.motor_name}</span>
          <input type="range" class="so101-joint-slider" data-jidx="${idx}" min="${minVal}" max="${maxVal}" step="${stepVal}" value="${curVal}" />
          <input type="number" class="so101-input so101-joint-num" data-jidx="${idx}" min="${minVal}" max="${maxVal}" step="${stepVal}" value="${Number(curVal).toFixed(state.unitMode === "ticks" ? 0 : 2)}" />
          <span class="so101-mono" style="color:#94a3b8;font-size:11px;">${kin.q_cad_deg[idx].toFixed(1)}° (${unitSuffix})</span>
        </div>
      `;
    });

    const gripPct = s.arms.arms[state.activeArm].gripper_pct;
    const gripTick = s.arms.arms[state.activeArm].gripper_tick;
    rows.push(`
      <div class="so101-slider-row" style="margin-top:6px;padding-top:6px;border-top:1px dashed #282e3b;">
        <span class="so101-mono" style="color:#fbbf24;">gripper</span>
        <input type="range" id="so101-grip-slider" min="0" max="100" step="0.5" value="${gripPct}" />
        <input type="number" class="so101-input" id="so101-grip-num" min="0" max="100" step="1" value="${gripPct}" />
        <span class="so101-mono" style="color:#fbbf24;font-size:11px;">Tick: ${gripTick}</span>
      </div>
    `);
    return rows.join("");
  }

  function renderFKPanelHTML() {
    const kin = state.snap.kinematics;
    const T = state.fkFrame === "world" ? kin.T_world_tcp : kin.T_base_tcp;
    const xyz = state.fkFrame === "world" ? kin.tcp_world_xyz_mm : kin.tcp_base_xyz_mm;
    const rpy = state.fkFrame === "world" ? kin.tcp_world_rpy_deg : kin.tcp_base_rpy_deg;

    return `
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;">
        <span><strong>TCP Position:</strong> <code class="so101-mono">[${xyz.join(", ")}] mm</code></span>
        <span><strong>RPY:</strong> <code class="so101-mono">[${rpy.join("°, ")}°]</code></span>
      </div>
      <div class="so101-matrix so101-mono">
        <table class="so101-table">
          <tbody>
            ${T.map(
              (row) =>
                `<tr>${row.map((v) => `<td>${Number(v).toFixed(4)}</td>`).join("")}</tr>`
            ).join("")}
          </tbody>
        </table>
      </div>
      <details class="so101-details">
        <summary>Link Frame Origins (World mm)</summary>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:8px;font-size:11px;" class="so101-mono">
          ${kin.link_chain
            .map(
              (l) => `
            <div style="background:#0d1016;padding:3px 6px;border-radius:4px;">
              <span style="color:#38bdf8;">${l.link}:</span> [${l.pos_world_mm.map((v) => Math.round(v)).join(",")}]
            </div>
          `
            )
            .join("")}
        </div>
      </details>
    `;
  }

  function renderJacobianPanelHTML() {
    const kin = state.snap.kinematics;
    const J = kin[state.jacMode] || kin.J_world_rad;
    const rowLabels = ["vx", "vy", "vz", "ωx", "ωy", "ωz"];
    const decimals = state.jacMode === "J_world_tick" ? 6 : 4;

    return `
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;">
        <span><strong>Manipulability:</strong> <code class="so101-mono" style="color:#34d399;">${kin.manipulability_5d.toFixed(5)}</code></span>
        <span><strong>Condition Number κ(J):</strong> <code class="so101-mono" style="color:#fbbf24;">${kin.condition_number.toFixed(2)}</code></span>
      </div>
      <div class="so101-matrix so101-mono">
        <table class="so101-table">
          <thead>
            <tr>
              <th>Axis</th>
              <th>shoulder_pan</th>
              <th>shoulder_lift</th>
              <th>elbow_flex</th>
              <th>wrist_flex</th>
              <th>wrist_roll</th>
            </tr>
          </thead>
          <tbody>
            ${J.map(
              (row, rIdx) => `
              <tr>
                <td style="color:#38bdf8;font-weight:700;">${rowLabels[rIdx]}</td>
                ${row.map((v) => `<td>${Number(v).toFixed(decimals)}</td>`).join("")}
              </tr>
            `
            ).join("")}
          </tbody>
        </table>
      </div>
      <div style="margin-top:8px;font-size:11.5px;color:#94a3b8;" class="so101-mono">
        <strong>Singular Values σ:</strong> [${kin.singular_values.map((v) => Number(v).toFixed(4)).join(", ")}]
      </div>
    `;
  }

  function renderDynamicsPanelHTML() {
    const dyn = state.snap.dynamics;
    const M = dyn.mass_matrix_5x5_kg_m2;

    return `
      <div style="font-size:12px;margin-bottom:6px;display:flex;justify-content:space-between;">
        <span><strong>Whole-Arm Center of Mass (World):</strong> <code class="so101-mono">[${dyn.whole_arm_com_world_m.map((v) => (v * 1000).toFixed(1)).join(", ")}] mm</code></span>
      </div>
      <table class="so101-table so101-mono">
        <thead>
          <tr>
            <th>Joint</th>
            <th>Gravity Torque (N·m)</th>
            <th>Stall Load %</th>
            <th>Feedforward Offset (ticks)</th>
          </tr>
        </thead>
        <tbody>
          ${MOTOR_NAMES.map(
            (m, i) => `
            <tr>
              <td>${m}</td>
              <td>${dyn.gravity_torque_nm[i].toFixed(4)}</td>
              <td style="color:${dyn.servo_load_pct_of_stall[i] > 35 ? "#fbbf24" : "#34d399"};">${dyn.servo_load_pct_of_stall[i].toFixed(1)}%</td>
              <td>${dyn.feedforward_gravity_ticks[i] >= 0 ? "+" : ""}${dyn.feedforward_gravity_ticks[i]}</td>
            </tr>
          `
          ).join("")}
        </tbody>
      </table>
      <details class="so101-details">
        <summary>Joint-Space Mass Matrix M(q) (kg·m²)</summary>
        <div class="so101-matrix so101-mono" style="margin-top:6px;">
          <table class="so101-table">
            <tbody>
              ${M.map(
                (row) => `<tr>${row.map((v) => `<td>${Number(v).toFixed(5)}</td>`).join("")}</tr>`
              ).join("")}
            </tbody>
          </table>
        </div>
      </details>
    `;
  }

  function updateLiveReadoutsOnly() {
    const fkEl = document.getElementById("so101-fk-panel");
    const jacEl = document.getElementById("so101-jacobian-panel");
    const dynEl = document.getElementById("so101-dynamics-panel");
    const pillEl = document.getElementById("so101-3d-phase-pill");
    const l2fEl = document.getElementById("so101-l2f-readout");
    if (fkEl) fkEl.innerHTML = renderFKPanelHTML();
    if (jacEl) jacEl.innerHTML = renderJacobianPanelHTML();
    if (dynEl) dynEl.innerHTML = renderDynamicsPanelHTML();
    if (pillEl && state.snap) {
      const kin = state.snap.kinematics;
      pillEl.textContent = `TCP: [${kin.tcp_world_xyz_mm.join(", ")}] mm`;
    }
    if (l2fEl && state.snap) {
      const m = state.snap.arms.leader_to_follower_mapping;
      l2fEl.textContent = `Mapped Follower Ticks: [${m.follower_ticks.join(", ")}] · Gripper: ${m.follower_gripper_tick} (min ${state.snap.calibration.safe_gripper_min_raw})`;
    }
    draw3DCanvas();
  }

  async function pushJointUpdateFromUI() {
    const sliders = Array.from(document.querySelectorAll(".so101-joint-slider"));
    const vals = sliders.map((s) => Number(s.value));
    const gripEl = document.getElementById("so101-grip-slider");
    const gripPct = gripEl ? Number(gripEl.value) : 22;

    const payload = { arm: state.activeArm, gripper_pct: gripPct };
    payload[state.unitMode] = vals;

    try {
      const res = await apiPost("/api/so101/kinematics", payload);
      state.snap.kinematics = res.kinematics;
      state.snap.dynamics = res.dynamics;
      state.snap.arms = res.arms;
      updateLiveReadoutsOnly();
      if (state.camStreamActive) {
        ["top", "plate", "wrist"].forEach((slot) => {
          const img = document.getElementById(`so101-cam-img-${slot}`);
          if (img) img.src = `/api/so101/camera/${slot}.jpg?t=${Date.now()}`;
        });
      }
    } catch (_e) {
      // Ignore out-of-range transient drag
    }
  }

  async function recompileActiveGraph() {
    if (!state.activeGraph) return;
    try {
      const res = await apiPost("/api/so101/compile-graph", { graph: state.activeGraph });
      state.compiledBundle = res;
      renderWorkbench();
    } catch (err) {
      alert(`Compile failed: ${String(err)}`);
    }
  }

  function bindComposerEvents() {
    document.querySelectorAll("[data-journey-step]").forEach((btn) => {
      btn.onclick = () => {
        state.journeyStep = btn.getAttribute("data-journey-step");
        renderWorkbench();
      };
    });

    document.querySelectorAll("[data-math-tab]").forEach((btn) => {
      btn.onclick = () => {
        state.mathTab = btn.getAttribute("data-math-tab");
        renderWorkbench();
      };
    });

    const presetSel = document.getElementById("sel-so101-preset");
    if (presetSel) {
      presetSel.onchange = async (e) => {
        const pid = e.target.value;
        state.activePresetId = pid;
        const presets = (state.snap && state.snap.composer && state.snap.composer.presets) || {};
        if (presets[pid]) {
          state.activeGraph = JSON.parse(JSON.stringify(presets[pid]));
          state.selectedNodeId =
            state.activeGraph.nodes && state.activeGraph.nodes[0]
              ? state.activeGraph.nodes[0].id
              : null;
          state.wiringFrom = null;
          await recompileActiveGraph();
        }
      };
    }

    const addBlockBtn = document.getElementById("btn-so101-add-block");
    if (addBlockBtn) {
      addBlockBtn.onclick = async () => {
        const sel = document.getElementById("sel-so101-add-block");
        if (!sel || !state.activeGraph) return;
        const typeName = sel.value;
        const def = getBlockDef(typeName);
        if (!def) return;
        const defaultParams = {};
        Object.entries(def.params || {}).forEach(([k, sch]) => {
          defaultParams[k] = sch.default;
        });
        const newId = `${typeName.split("_")[0]}_${Math.floor(100 + Math.random() * 899)}`;
        const count = (state.activeGraph.nodes || []).length;
        state.activeGraph.nodes.push({
          id: newId,
          type: typeName,
          x: 80 + (count % 6) * 260,
          y: 70 + Math.floor(count / 6) * 160,
          params: defaultParams,
        });
        state.selectedNodeId = newId;
        await recompileActiveGraph();
      };
    }

    const compileBtn = document.getElementById("btn-so101-compile-graph");
    if (compileBtn) {
      compileBtn.onclick = () => recompileActiveGraph();
    }

    const dlCompBtn = document.getElementById("btn-so101-dl-compiled-py");
    if (dlCompBtn) {
      dlCompBtn.onclick = async () => {
        if (!state.compiledBundle) await recompileActiveGraph();
        const code =
          (state.compiledBundle && state.compiledBundle.compiled_python_policy) || "";
        downloadText("so101_compiled_policy.py", code, "text/x-python");
      };
    }

    const copyBtn = document.getElementById("btn-so101-copy-compiled");
    if (copyBtn) {
      copyBtn.onclick = () => {
        const code =
          (state.compiledBundle && state.compiledBundle.compiled_python_policy) || "";
        if (navigator.clipboard) {
          navigator.clipboard.writeText(code);
          copyBtn.textContent = "✓ Copied!";
        }
      };
    }

    // Custom SymSkill / NSRT Skill Builder events
    const syncSkillInputsFromDOM = () => {
      if (!state.activeGraph) return;
      const skills = ensureCustomSkillsOnGraph();
      document.querySelectorAll("[data-sk-field]").forEach((inp) => {
        const idx = Number(inp.getAttribute("data-sk-idx"));
        const field = inp.getAttribute("data-sk-field");
        if (skills[idx] && field) {
          skills[idx][field] = field === "gripper_pct" ? Number(inp.value) : inp.value;
        }
      });
    };

    document.querySelectorAll("[data-sk-field]").forEach((inp) => {
      inp.onchange = () => syncSkillInputsFromDOM();
    });

    const addSkillBtn = document.getElementById("btn-so101-add-skill");
    if (addSkillBtn) {
      addSkillBtn.onclick = async () => {
        syncSkillInputsFromDOM();
        const skills = ensureCustomSkillsOnGraph();
        const nextNum = skills.length + 1;
        skills.push({
          operator: `CustomSkill_${nextNum}(obj_0)`,
          preconditions: "HandEmpty()",
          expected_atoms: `CustomSkill_${nextNum}Done(obj_0)`,
          waypoint: "observation_home",
          gripper_pct: 22.0,
        });
        await recompileActiveGraph();
      };
    }

    const saveSkillsBtn = document.getElementById("btn-so101-save-skills");
    if (saveSkillsBtn) {
      saveSkillsBtn.onclick = async () => {
        syncSkillInputsFromDOM();
        await recompileActiveGraph();
      };
    }

    document.querySelectorAll("[data-sk-up]").forEach((btn) => {
      btn.onclick = async () => {
        syncSkillInputsFromDOM();
        const idx = Number(btn.getAttribute("data-sk-up"));
        const skills = ensureCustomSkillsOnGraph();
        if (idx > 0 && idx < skills.length) {
          const tmp = skills[idx - 1];
          skills[idx - 1] = skills[idx];
          skills[idx] = tmp;
          await recompileActiveGraph();
        }
      };
    });

    document.querySelectorAll("[data-sk-down]").forEach((btn) => {
      btn.onclick = async () => {
        syncSkillInputsFromDOM();
        const idx = Number(btn.getAttribute("data-sk-down"));
        const skills = ensureCustomSkillsOnGraph();
        if (idx >= 0 && idx < skills.length - 1) {
          const tmp = skills[idx + 1];
          skills[idx + 1] = skills[idx];
          skills[idx] = tmp;
          await recompileActiveGraph();
        }
      };
    });

    document.querySelectorAll("[data-sk-del]").forEach((btn) => {
      btn.onclick = async () => {
        syncSkillInputsFromDOM();
        const idx = Number(btn.getAttribute("data-sk-del"));
        const skills = ensureCustomSkillsOnGraph();
        if (skills.length > 1) {
          skills.splice(idx, 1);
          await recompileActiveGraph();
        }
      };
    });

    const cancelWire = document.getElementById("btn-cancel-wire");
    if (cancelWire) {
      cancelWire.onclick = () => {
        state.wiringFrom = null;
        renderWorkbench();
      };
    }

    // Delete block button
    document.querySelectorAll("[data-del-node]").forEach((btn) => {
      btn.onclick = async (ev) => {
        ev.stopPropagation();
        const nid = btn.getAttribute("data-del-node");
        if (!state.activeGraph) return;
        state.activeGraph.nodes = (state.activeGraph.nodes || []).filter((n) => n.id !== nid);
        state.activeGraph.edges = (state.activeGraph.edges || []).filter(
          (e) => e.from_node !== nid && e.to_node !== nid
        );
        if (state.selectedNodeId === nid) {
          state.selectedNodeId =
            state.activeGraph.nodes[0] ? state.activeGraph.nodes[0].id : null;
        }
        await recompileActiveGraph();
      };
    });

    // Delete wire button
    document.querySelectorAll("[data-del-edge]").forEach((btn) => {
      btn.onclick = async () => {
        const raw = btn.getAttribute("data-del-edge") || "";
        const [fn, fp, tn, tp] = raw.split("|");
        if (!state.activeGraph) return;
        state.activeGraph.edges = (state.activeGraph.edges || []).filter(
          (e) => !(e.from_node === fn && e.from_port === fp && e.to_node === tn && e.to_port === tp)
        );
        await recompileActiveGraph();
      };
    });

    // Save node params in inspector
    const saveParamsBtn = document.getElementById("btn-so101-save-node-params");
    if (saveParamsBtn) {
      saveParamsBtn.onclick = async () => {
        const node = (state.activeGraph.nodes || []).find((n) => n.id === state.selectedNodeId);
        if (!node) return;
        if (!node.params) node.params = {};
        document.querySelectorAll("[data-node-param]").forEach((el) => {
          const pKey = el.getAttribute("data-node-param");
          const pType = el.getAttribute("data-param-type") || "string";
          const rawVal = el.value;
          if (pType === "number") {
            node.params[pKey] = Number(rawVal);
          } else if (pType === "bool") {
            node.params[pKey] = rawVal === "true";
          } else if (pType === "vec3" || pType === "vec2") {
            try {
              node.params[pKey] = JSON.parse(rawVal);
            } catch (_e) {
              node.params[pKey] = rawVal;
            }
          } else {
            node.params[pKey] = rawVal;
          }
        });
        await recompileActiveGraph();
      };
    }

    // Port clicking for Simulink wiring
    document.querySelectorAll(".so101-port-pill").forEach((pill) => {
      pill.onclick = async (ev) => {
        ev.stopPropagation();
        const dir = pill.getAttribute("data-port-dir");
        const nodeId = pill.getAttribute("data-node-id");
        const portId = pill.getAttribute("data-port-id");
        const portType = pill.getAttribute("data-port-type");

        if (dir === "out") {
          state.wiringFrom = { nodeId, portId, portType };
          renderWorkbench();
          return;
        }
        if (dir === "in" && state.wiringFrom) {
          if (state.wiringFrom.nodeId === nodeId) return;
          if (state.wiringFrom.portType !== portType) {
            alert(
              `Port Type Mismatch!\nCannot connect ${state.wiringFrom.nodeId}.${state.wiringFrom.portId} (${state.wiringFrom.portType}) to ${nodeId}.${portId} (${portType}).\nOnly matching signal types can connect.`
            );
            return;
          }
          // Replace existing incoming wire on that input port and add new edge
          state.activeGraph.edges = (state.activeGraph.edges || []).filter(
            (e) => !(e.to_node === nodeId && e.to_port === portId)
          );
          state.activeGraph.edges.push({
            from_node: state.wiringFrom.nodeId,
            from_port: state.wiringFrom.portId,
            to_node: nodeId,
            to_port: portId,
          });
          state.wiringFrom = null;
          await recompileActiveGraph();
        }
      };
    });

    // Dragging nodes on the Simulink stage
    const stage = document.getElementById("so101-simulink-stage");
    document.querySelectorAll(".so101-node").forEach((nodeEl) => {
      nodeEl.onmousedown = (ev) => {
        if (
          ev.target.closest(".so101-port-pill") ||
          ev.target.hasAttribute("data-del-node")
        ) {
          return;
        }
        const nid = nodeEl.getAttribute("data-node-id");
        state.selectedNodeId = nid;
        const nodeObj = (state.activeGraph.nodes || []).find((n) => n.id === nid);
        if (!nodeObj || !stage) return;

        const stageRect = stage.getBoundingClientRect();
        state.draggingNodeId = nid;
        state.dragNodeOffset = {
          x: ev.clientX - stageRect.left - (nodeObj.x || 0),
          y: ev.clientY - stageRect.top - (nodeObj.y || 0),
        };

        document.querySelectorAll(".so101-node").forEach((el) => {
          el.classList.toggle("selected", el.getAttribute("data-node-id") === nid);
        });
        const insp = document.getElementById("so101-simulink-inspector");
        if (insp) {
          insp.innerHTML = renderSimulinkInspectorHTML();
          bindComposerEvents();
        }
      };
    });

    if (stage) {
      stage.onmousemove = (ev) => {
        if (!state.draggingNodeId || !state.activeGraph) return;
        const nodeObj = (state.activeGraph.nodes || []).find(
          (n) => n.id === state.draggingNodeId
        );
        const nodeEl = document.getElementById(`sim-node-${state.draggingNodeId}`);
        if (!nodeObj || !nodeEl) return;
        const stageRect = stage.getBoundingClientRect();
        const nx = Math.max(10, Math.min(2300, Math.round(ev.clientX - stageRect.left - state.dragNodeOffset.x)));
        const ny = Math.max(10, Math.min(420, Math.round(ev.clientY - stageRect.top - state.dragNodeOffset.y)));
        nodeObj.x = nx;
        nodeObj.y = ny;
        nodeEl.style.left = `${nx}px`;
        nodeEl.style.top = `${ny}px`;
        drawSimulinkWires();
      };
      stage.onmouseup = () => {
        state.draggingNodeId = null;
      };
    }
  }

  function bindWorkbenchEvents() {
    const closeBtn = document.getElementById("btn-so101-close");
    if (closeBtn) closeBtn.onclick = () => toggleWorkbench(false);

    const syncBtn = document.getElementById("btn-so101-sync");
    if (syncBtn) {
      syncBtn.onclick = async () => {
        const m = window.location.pathname.match(/\/projects\/([0-9a-fA-F-]{36})/);
        state.snap = await apiPost("/api/so101/sync-project", { project_id: m ? m[1] : null });
        renderWorkbench();
      };
    }

    const playBtn = document.getElementById("btn-so101-play-traj");
    if (playBtn) {
      playBtn.onclick = () => {
        if (state.playingTraj) {
          stopTrajectoryPlayback();
          renderWorkbench();
        } else {
          startTrajectoryPlayback();
        }
      };
    }

    const dlPy = document.getElementById("btn-so101-dl-py");
    if (dlPy) {
      dlPy.onclick = async () => {
        const bundle = await apiGet("/api/so101/export");
        downloadText("so101_pick_place_controller.py", bundle.controller_python_script, "text/x-python");
      };
    }

    const dlUrdf = document.getElementById("btn-so101-dl-urdf");
    if (dlUrdf) {
      dlUrdf.onclick = async () => {
        const bundle = await apiGet("/api/so101/export");
        downloadText("so101-follower-calibrated.urdf", bundle.calibrated_follower_urdf, "application/xml");
      };
    }

    const dlJson = document.getElementById("btn-so101-dl-json");
    if (dlJson) {
      dlJson.onclick = async () => {
        const bundle = await apiGet("/api/so101/export");
        downloadText("so101-architecture-bundle.json", JSON.stringify(bundle, null, 2), "application/json");
      };
    }

    const camToggle = document.getElementById("btn-so101-cam-toggle");
    if (camToggle) {
      camToggle.onclick = () => {
        state.camStreamActive = !state.camStreamActive;
        camToggle.textContent = state.camStreamActive ? "⏸ Pause Feeds" : "▶ Resume Feeds";
      };
    }

    const wsBtn = document.getElementById("btn-so101-leader-ws");
    if (wsBtn) wsBtn.onclick = () => connectStudioLeaderWs();

    const armSel = document.getElementById("sel-so101-arm");
    if (armSel) {
      armSel.onchange = async (e) => {
        state.activeArm = e.target.value;
        const res = await apiPost("/api/so101/kinematics", { arm: state.activeArm });
        state.snap.kinematics = res.kinematics;
        state.snap.dynamics = res.dynamics;
        state.snap.arms = res.arms;
        renderWorkbench();
      };
    }

    const unitSel = document.getElementById("sel-so101-unit");
    if (unitSel) {
      unitSel.onchange = (e) => {
        state.unitMode = e.target.value;
        renderWorkbench();
      };
    }

    const l2fBtn = document.getElementById("btn-so101-map-l2f");
    if (l2fBtn) {
      l2fBtn.onclick = async () => {
        const mapped = state.snap.arms.leader_to_follower_mapping;
        state.activeArm = "follower";
        const res = await apiPost("/api/so101/kinematics", {
          arm: "follower",
          q_enc_rad: mapped.follower_q_enc_rad,
          gripper_pct: mapped.follower_gripper_pct,
        });
        state.snap.kinematics = res.kinematics;
        state.snap.dynamics = res.dynamics;
        state.snap.arms = res.arms;
        renderWorkbench();
      };
    }

    document.querySelectorAll(".so101-joint-slider").forEach((sl) => {
      sl.oninput = (e) => {
        const idx = e.target.getAttribute("data-jidx");
        const num = document.querySelector(`.so101-joint-num[data-jidx="${idx}"]`);
        if (num) num.value = e.target.value;
        pushJointUpdateFromUI();
      };
    });

    document.querySelectorAll(".so101-joint-num").forEach((num) => {
      num.onchange = (e) => {
        const idx = e.target.getAttribute("data-jidx");
        const sl = document.querySelector(`.so101-joint-slider[data-jidx="${idx}"]`);
        if (sl) sl.value = e.target.value;
        pushJointUpdateFromUI();
      };
    });

    const gSl = document.getElementById("so101-grip-slider");
    const gNum = document.getElementById("so101-grip-num");
    if (gSl && gNum) {
      gSl.oninput = () => {
        gNum.value = gSl.value;
        pushJointUpdateFromUI();
      };
      gNum.onchange = () => {
        gSl.value = gNum.value;
        pushJointUpdateFromUI();
      };
    }

    document.querySelectorAll("[data-wp-load]").forEach((btn) => {
      btn.onclick = async () => {
        const wpKey = btn.getAttribute("data-wp-load");
        const wp = state.snap.waypoints[wpKey];
        if (!wp) return;
        state.activeArm = "follower";
        const res = await apiPost("/api/so101/kinematics", {
          arm: "follower",
          q_enc_rad: wp.q_enc_rad,
          gripper_pct: wp.gripper_pct,
        });
        state.snap.kinematics = res.kinematics;
        state.snap.dynamics = res.dynamics;
        state.snap.arms = res.arms;
        renderWorkbench();
      };
    });

    document.querySelectorAll("[data-wp-teach]").forEach((btn) => {
      btn.onclick = async () => {
        const wpKey = btn.getAttribute("data-wp-teach");
        const res = await apiPost("/api/so101/teach", {
          waypoint: wpKey,
          source_arm: state.activeArm,
        });
        state.snap.waypoints = res.waypoints;
        state.snap.kinematics = res.kinematics;
        state.snap.dynamics = res.dynamics;
        state.snap.trajectory = res.trajectory;
        renderWorkbench();
      };
    });

    const solveBowlsBtn = document.getElementById("btn-so101-solve-bowls");
    if (solveBowlsBtn) {
      solveBowlsBtn.onclick = async () => {
        const body = {};
        ["source", "good", "defective"].forEach((k) => {
          body[k] = [
            Number(document.getElementById(`bowl-${k}-x`).value),
            Number(document.getElementById(`bowl-${k}-y`).value),
            Number(document.getElementById(`bowl-${k}-z`).value),
          ];
        });
        const res = await apiPost("/api/so101/bowls", body);
        state.snap.bowls = res.bowls;
        state.snap.waypoints = res.waypoints;
        state.snap.trajectory = res.trajectory;
        renderWorkbench();
      };
    }

    const fkSel = document.getElementById("sel-so101-fk-frame");
    if (fkSel) {
      fkSel.onchange = (e) => {
        state.fkFrame = e.target.value;
        updateLiveReadoutsOnly();
      };
    }

    const jacSel = document.getElementById("sel-so101-jac-mode");
    if (jacSel) {
      jacSel.onchange = (e) => {
        state.jacMode = e.target.value;
        updateLiveReadoutsOnly();
      };
    }

    const solveIkBtn = document.getElementById("btn-so101-solve-ik");
    if (solveIkBtn) {
      solveIkBtn.onclick = async () => {
        const tx = Number(document.getElementById("ik-tgt-x").value);
        const ty = Number(document.getElementById("ik-tgt-y").value);
        const tz = Number(document.getElementById("ik-tgt-z").value);
        const pitch = Number(document.getElementById("ik-pitch-deg").value);
        const yaw = Number(document.getElementById("ik-yaw-deg").value);
        const wp = document.getElementById("ik-save-wp").value || null;
        const res = await apiPost("/api/so101/ik", {
          target_xyz_world: [tx, ty, tz],
          pitch_down_deg: pitch,
          jaw_yaw_deg: yaw,
          apply_to_waypoint: wp,
          update_virtual_arm: true,
        });
        state.lastIkResult = res;
        state.snap.kinematics = res.kinematics;
        state.snap.dynamics = res.dynamics;
        if (res.trajectory) state.snap.trajectory = res.trajectory;
        renderWorkbench();
      };
    }

    const useTicksBtn = document.getElementById("btn-so101-use-cur-ticks");
    if (useTicksBtn) {
      useTicksBtn.onclick = () => {
        state.snap.kinematics.ticks.forEach((t, i) => {
          const inp = document.getElementById(`align-tick-${i}`);
          if (inp) inp.value = t;
        });
      };
    }

    const runAlignBtn = document.getElementById("btn-so101-run-align");
    if (runAlignBtn) {
      runAlignBtn.onclick = async () => {
        const ticks = [0, 1, 2, 3, 4].map((i) => Number(document.getElementById(`align-tick-${i}`).value));
        const res = await apiPost("/api/so101/align", {
          arm: state.activeArm,
          measured_ticks: ticks,
        });
        state.lastAlignResult = res;
        const pre = document.getElementById("so101-align-output");
        if (pre) pre.textContent = JSON.stringify(res.alignment_snippet, null, 2);
      };
    }

    // 3D Canvas orbit controls
    const canvas = document.getElementById("so101-3d-canvas");
    if (canvas) {
      canvas.onmousedown = (e) => {
        state.dragging3d = true;
        state.lastMouse = { x: e.clientX, y: e.clientY };
      };
      window.addEventListener("mouseup", () => {
        state.dragging3d = false;
      });
      canvas.onmousemove = (e) => {
        if (!state.dragging3d) return;
        const dx = e.clientX - state.lastMouse.x;
        const dy = e.clientY - state.lastMouse.y;
        state.lastMouse = { x: e.clientX, y: e.clientY };
        state.orbitAz -= dx * 0.008;
        state.orbitEl = Math.max(0.08, Math.min(1.45, state.orbitEl + dy * 0.008));
        draw3DCanvas();
      };
      canvas.onwheel = (e) => {
        e.preventDefault();
        state.orbitDist = Math.max(0.4, Math.min(1.6, state.orbitDist + e.deltaY * 0.0008));
        draw3DCanvas();
      };
    }
  }

  function startTrajectoryPlayback() {
    stopTrajectoryPlayback();
    state.playingTraj = true;
    state.trajFrameIdx = 0;
    const playBtn = document.getElementById("btn-so101-play-traj");
    if (playBtn) playBtn.textContent = "⏹ Stop 3D Trajectory";

    state.trajTimer = setInterval(() => {
      if (!state.open || !state.playingTraj || !state.snap) {
        stopTrajectoryPlayback();
        return;
      }
      const frames = state.snap.trajectory.frames || [];
      if (!frames.length) return;
      const f = frames[state.trajFrameIdx % frames.length];
      state.trajFrameIdx = (state.trajFrameIdx + 1) % frames.length;
      const pill = document.getElementById("so101-3d-phase-pill");
      if (pill) {
        pill.textContent = `${f.phase} | w5 = ${f.manipulability_5d.toFixed(4)}`;
      }
      draw3DCanvas(f);
    }, 95);
  }

  function stopTrajectoryPlayback() {
    state.playingTraj = false;
    if (state.trajTimer) {
      clearInterval(state.trajTimer);
      state.trajTimer = null;
    }
  }

  function project3D(pt, w, h) {
    const target = [0.22, 0.0, 0.08];
    const ex = target[0] + state.orbitDist * Math.cos(state.orbitEl) * Math.cos(state.orbitAz);
    const ey = target[1] + state.orbitDist * Math.cos(state.orbitEl) * Math.sin(state.orbitAz);
    const ez = target[2] + state.orbitDist * Math.sin(state.orbitEl);

    let fx = target[0] - ex,
      fy = target[1] - ey,
      fz = target[2] - ez;
    const fn = Math.hypot(fx, fy, fz) || 1;
    fx /= fn;
    fy /= fn;
    fz /= fn;

    // Right = forward x up(0,0,1)
    let rx = fy,
      ry = -fx,
      rz = 0;
    const rn = Math.hypot(rx, ry, rz) || 1;
    rx /= rn;
    ry /= rn;

    // True up = right x forward
    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;

    const dx = pt[0] - ex,
      dy = pt[1] - ey,
      dz = pt[2] - ez;
    const cx = dx * rx + dy * ry + dz * rz;
    const cy = dx * ux + dy * uy + dz * uz;
    const cz = dx * fx + dy * fy + dz * fz;
    if (cz <= 0.05) return null;
    const f = 430;
    return [w * 0.5 + (cx / cz) * f, h * 0.54 - (cy / cz) * f, cz];
  }

  function draw3DCanvas(overrideFrame) {
    const canvas = document.getElementById("so101-3d-canvas");
    if (!canvas || !state.snap) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // 1. Floor grid (z = 0)
    ctx.strokeStyle = "#1e2633";
    ctx.lineWidth = 1;
    for (let gx = 0.0; gx <= 0.5; gx += 0.05) {
      const p0 = project3D([gx, -0.3, 0], w, h);
      const p1 = project3D([gx, 0.3, 0], w, h);
      if (p0 && p1) {
        ctx.beginPath();
        ctx.moveTo(p0[0], p0[1]);
        ctx.lineTo(p1[0], p1[1]);
        ctx.stroke();
      }
    }
    for (let gy = -0.3; gy <= 0.301; gy += 0.05) {
      const p0 = project3D([0.0, gy, 0], w, h);
      const p1 = project3D([0.5, gy, 0], w, h);
      if (p0 && p1) {
        ctx.beginPath();
        ctx.moveTo(p0[0], p0[1]);
        ctx.lineTo(p1[0], p1[1]);
        ctx.stroke();
      }
    }

    // 2. World Coordinate Triad (+X red, +Y green, +Z blue)
    const o0 = project3D([0, 0, 0], w, h);
    const ox = project3D([0.08, 0, 0], w, h);
    const oy = project3D([0, 0.08, 0], w, h);
    const oz = project3D([0, 0, 0.08], w, h);
    if (o0 && ox && oy && oz) {
      [
        [ox, "#ef4444", "+X"],
        [oy, "#22c55e", "+Y"],
        [oz, "#3b82f6", "+Z"],
      ].forEach(([pt, col, lbl]) => {
        ctx.strokeStyle = col;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(o0[0], o0[1]);
        ctx.lineTo(pt[0], pt[1]);
        ctx.stroke();
        ctx.fillStyle = col;
        ctx.font = "10px monospace";
        ctx.fillText(lbl, pt[0] + 3, pt[1] - 2);
      });
    }

    // 3. Draw ONLY the SO-101 Kinematic Arm Skeleton (clean 3D environment)
    const chainPts = overrideFrame
      ? [...overrideFrame.link_chain_world, overrideFrame.tcp_world_xyz_m]
      : [...state.snap.kinematics.link_chain.map((r) => r.pos_world_m), state.snap.kinematics.tcp_world_xyz_m];

    ctx.strokeStyle = "#f97316";
    ctx.lineWidth = 4;
    ctx.beginPath();
    const projChain = chainPts.map((pt) => project3D(pt, w, h));
    projChain.forEach((p, idx) => {
      if (!p) return;
      if (idx === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.stroke();

    projChain.forEach((p, idx) => {
      if (!p) return;
      ctx.fillStyle = idx === projChain.length - 1 ? "#38bdf8" : "#fde047";
      ctx.beginPath();
      ctx.arc(p[0], p[1], idx === projChain.length - 1 ? 5.5 : 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // Observe DOM changes in Physical AI Studio SPA to keep button mounted
  const obs = new MutationObserver(() => ensureHeaderButtons());
  if (document.body) {
    ensureHeaderButtons();
    obs.observe(document.body, { childList: true, subtree: true });
  } else {
    window.addEventListener("DOMContentLoaded", () => {
      ensureHeaderButtons();
      obs.observe(document.body, { childList: true, subtree: true });
    });
  }

  if (window.location.hash === "#so101-architecture") {
    setTimeout(() => toggleWorkbench(true), 300);
  }
})();
