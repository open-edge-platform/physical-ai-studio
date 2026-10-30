"""Interactive pi05 language demo in the MuJoCo bucket_sort scene.

Type any instruction, run the policy closed loop, and see where every object ends up. The same
layout (seed) can be replayed with different wording to test what the policy understands.

The loop matches the sim evaluation: 30 Hz control, actions eased over one tick, both cameras
JPEG-encoded at quality 85 before the policy sees them, inference in lockstep with the physics.

Serves on 127.0.0.1 only; reach it through an SSH tunnel.
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2
import mujoco
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from physicalai.data import Observation
from physicalai.policies import get_physicalai_policy_class
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from pydantic import BaseModel

HZ = 30
CAMERAS = ("overview", "wrist")
PAGE = (Path(__file__).parent / "demo_page.html").read_text()
torch.set_num_threads(4)
cv2.setNumThreads(2)


def jpeg_roundtrip(rgb: np.ndarray, quality: int = 85) -> tuple[np.ndarray, bytes]:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), buf.tobytes()


GRASPABLE = ("resistor", "battery", "cpu")


def pinched_bodies(model, data) -> set[int]:
    """Objects that both Franka fingers touch (grasped objects); the table and buckets never count."""
    fingers = {model.body(n).id for n in ("left_finger", "right_finger")}
    objects = {model.body(n).id for n in GRASPABLE}
    touched: dict[int, set[int]] = {}
    for i in range(data.ncon):
        b1, b2 = model.geom_bodyid[data.contact[i].geom1], model.geom_bodyid[data.contact[i].geom2]
        for f, o in ((b1, b2), (b2, b1)):
            if f in fingers and o in objects:
                touched.setdefault(int(o), set()).add(int(f))
    return {o for o, v in touched.items() if len(v) == 2}


class Demo:
    """Owns the simulation and the policy; every simulation call happens on the control thread."""

    def __init__(self, ckpt: str, policy_name: str, max_s: float, viser_port: int) -> None:
        self.ckpt = ckpt
        self.max_s = max_s
        if policy_name == "lerobot":  # a lerobot-train pi05 pretrained_model dir
            from eval_bucket_sort import LeRobotPolicy

            self.policy = LeRobotPolicy(ckpt, "cuda")
        else:
            self.policy = get_physicalai_policy_class(policy_name).load_from_checkpoint(ckpt, map_location="cpu")
            from eval_bucket_sort import training_camera_order

            self.cameras = training_camera_order(self.policy)
            self.policy.to("cuda").eval()
        self.robot = MuJoCoRobot(
            "franka_fr3", scene="bucket_sort", cameras=[], http_port=0, rate_hz=HZ,
            enable_viewer=viser_port > 0, viser_host="127.0.0.1", viser_port=viser_port,
        )
        self.robot.connect()
        if self.robot._episode_auto_reset is not None:
            self.robot._episode_auto_reset.set_active(False)  # no automatic respawn after a success
        self.renderer = None  # created on the control thread: its EGL context is bound to that thread
        self.commands: queue.Queue = queue.Queue()
        self.frames = {cam: b"" for cam in CAMERAS}
        self.frame_id = 0
        self.lock = threading.Lock()
        self.seed: int | None = None
        self.status = "idle"
        self.instruction = ""
        self.started = 0.0
        self.elapsed = 0.0
        self.infer_ms: list[float] = []
        self.history: list[dict] = []
        self.grip_assist = False
        self.run_assist = False
        self.placed_ticks = 0
        self.start_buckets: dict = {}
        """Command the gripper fully shut while it pinches a body and the policy is not opening it."""
        self.last_start_state: dict | None = None
        self._reset(int(np.random.default_rng().integers(0, 1_000_000)))

    # -- control thread -------------------------------------------------

    def _reset(self, seed: int) -> None:
        self.robot._set_seed(seed)
        self.robot._reset()
        self.robot._set_seed(None)
        self.policy.reset()
        self.seed = seed
        self.status = "idle"
        self.elapsed = 0.0

    def _render(self, publish: bool) -> dict[str, np.ndarray]:
        images = {}
        for cam in getattr(self, "cameras", CAMERAS):
            self.renderer.update_scene(self.robot._sim.data, camera=cam)
            rgb, jpg = jpeg_roundtrip(self.renderer.render())
            images[cam] = rgb
            if publish:
                self.frames[cam] = jpg
        if publish:
            self.frame_id += 1
        return images

    def scene_state(self) -> dict:
        task = self.robot._sim.on_reset
        state = task.evaluate(self.robot._sim.model, self.robot._sim.data)
        desc = task.describe()
        return {
            "in_bucket": {k: v for k, v in state.in_bucket.items()},
            "buckets_left_to_right": [s["color"] for s in desc["slots"]],
            "scene_instruction": task.instruction.wording if task.instruction is not None else None,
        }

    def _finish(self, outcome: str) -> None:
        end = self.scene_state()["in_bucket"]
        self.history.insert(0, {
            "seed": self.seed, "instruction": self.instruction, "outcome": outcome, "assist": self.run_assist,
            "seconds": round(self.elapsed, 1), "placed": {k: v for k, v in end.items() if v},
            "infer_ms": round(float(np.mean(self.infer_ms)), 0) if self.infer_ms else None,
        })
        del self.history[30:]
        self.status = "idle"

    def loop(self) -> None:
        self.renderer = mujoco.Renderer(self.robot._sim.model, 480, 640)
        tick = 1.0 / HZ
        next_t = time.perf_counter()
        while True:
            try:
                while True:
                    cmd, arg = self.commands.get_nowait()
                    with self.lock:
                        if cmd == "reset":
                            if self.status == "running":
                                self._finish("stopped")
                            self._reset(arg)
                        elif cmd == "run":
                            if self.status == "running":
                                self._finish("stopped")
                            if arg["restart"]:
                                self._reset(self.seed)
                            self.policy.reset()
                            self.instruction = arg["instruction"]
                            self.status = "running"
                            self.started = time.perf_counter()
                            self.elapsed = 0.0
                            self.infer_ms = []
                            self.run_assist = self.grip_assist
                            self.placed_ticks = 0
                            self.start_buckets = self.scene_state()["in_bucket"]
                        elif cmd == "stop" and self.status == "running":
                            self._finish("stopped")
            except queue.Empty:
                pass

            obs = self.robot.get_observation()
            running = self.status == "running"
            images = self._render(publish=True)
            if running:
                batch = Observation(
                    state=torch.from_numpy(obs.joint_positions).float().unsqueeze(0).cuda(),
                    images={c: torch.from_numpy(images[c]).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
                            for c in CAMERAS},
                    task=[self.instruction],
                )
                queued = len(self.policy._action_queue)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    action = self.policy.select_action(batch)
                if queued == 0:
                    torch.cuda.synchronize()
                    self.infer_ms.append(1000 * (time.perf_counter() - t0))
                act = action.float().reshape(-1).cpu().numpy()
                if (self.grip_assist and act[7] < obs.joint_positions[7] + 0.003
                        and pinched_bodies(self.robot._sim.model, self.robot._sim.data)):
                    act[7] = 0.0
                self.robot.send_action(act, goal_time=tick)
                self.elapsed += tick
                # End the run once something has rested in a bucket for a second: the policy is done.
                now = self.scene_state()["in_bucket"]
                placed = any(b and b != self.start_buckets.get(o) for o, b in now.items())
                self.placed_ticks = self.placed_ticks + 1 if placed else 0
                if self.placed_ticks >= HZ:
                    with self.lock:
                        self._finish("placed")
                elif self.elapsed >= self.max_s:
                    with self.lock:
                        self._finish("timed out")
            # Real-time pacing when idle; lockstep (no catch-up) while the policy runs.
            next_t += tick
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()

    # -- HTTP side --------------------------------------------------------

    def state(self) -> dict:
        with self.lock:
            scene = self.scene_state()
            return {
                "status": self.status, "instruction": self.instruction, "seed": self.seed,
                "elapsed": round(self.elapsed, 1), "max_s": self.max_s,
                "infer_ms": round(float(np.mean(self.infer_ms)), 0) if self.infer_ms else None,
                "checkpoint": Path(self.ckpt).parent.name + "/" + Path(self.ckpt).name,
                "grip_assist": self.grip_assist,
                "history": self.history, **scene,
            }


class RunBody(BaseModel):
    instruction: str
    restart: bool = True


class AssistBody(BaseModel):
    grip_assist: bool


class ResetBody(BaseModel):
    seed: int | None = None


def make_app(demo: Demo) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(demo.state())

    @app.post("/api/run")
    def run(body: RunBody) -> dict:
        text = body.instruction.strip()[:200]
        if not text:
            return {"ok": False, "error": "Type an instruction first."}
        demo.commands.put(("run", {"instruction": text, "restart": body.restart}))
        return {"ok": True}

    @app.post("/api/assist")
    def assist(body: AssistBody) -> dict:
        demo.grip_assist = body.grip_assist
        return {"ok": True}

    @app.post("/api/stop")
    def stop() -> dict:
        demo.commands.put(("stop", None))
        return {"ok": True}

    @app.post("/api/reset")
    def reset(body: ResetBody) -> dict:
        seed = body.seed if body.seed is not None else int(np.random.default_rng().integers(0, 1_000_000))
        demo.commands.put(("reset", seed))
        return {"ok": True, "seed": seed}

    @app.get("/stream/{cam}")
    def stream(cam: str) -> StreamingResponse:
        def gen():
            last = -1
            while True:
                if demo.frame_id != last and demo.frames.get(cam):
                    last = demo.frame_id
                    jpg = demo.frames[cam]
                    yield b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                time.sleep(1 / 30)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--policy", default="pi05")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--viser-port", type=int, default=9090, help="3D viewer port; 0 disables it.")
    p.add_argument("--max-s", type=float, default=30.0)
    args = p.parse_args()
    demo = Demo(args.ckpt, args.policy, args.max_s, args.viser_port)
    threading.Thread(target=demo.loop, daemon=True).start()
    uvicorn.run(make_app(demo), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
