"""Closed-loop evaluation of a trained policy in the MuJoCo bucket_sort scene.

Mirrors how Studio drives the sim: 30 Hz control, actions eased over one tick, the two 640x480
cameras JPEG-encoded at quality 85 (the MJPEG stream) before the policy sees them. Each episode
uses a fixed seed (held out from recording, which used random seeds) so every policy sees the
same layouts and instructions.

Outcomes per episode: success (instructed object at rest in the instructed bucket), wrong_bucket
(the object ended in another bucket), or fail.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2
import mujoco
import numpy as np
import torch
from physicalai.data import Observation
from physicalai.policies import get_physicalai_policy_class
from physicalai_mujoco_plugin.robot import MuJoCoRobot

HZ = 30
torch.set_num_threads(4)
cv2.setNumThreads(2)
CAMERAS = ("wrist", "overview")


def training_camera_order(policy) -> tuple[str, ...]:
    """Camera names in the order the policy was trained on, read from its saved dataset stats.

    physicalai's pi05 fills its image slots in the order the batch dict lists the cameras, not by
    name, so a batch must list them in training order (the runtime fix, physicalai#411, does this
    for exported models only). The generated datasets are overview-first, Studio's are wrist-first.
    """
    stats = getattr(policy, "_dataset_stats", None) or {}
    names = tuple(
        key.removeprefix("observation.").removeprefix("images.")
        for key, stat in stats.items()
        if isinstance(stat, dict) and str(stat.get("type", "")).upper().endswith("VISUAL")
    )
    return names if set(names) == set(CAMERAS) else CAMERAS


def jpeg(rgb: np.ndarray, quality: int = 85) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class RemoteChunkPolicy:
    """Upstream LingBot inference server (deploy/lingbot_vla_v2_policy.py) behind the select_action API.

    The server applies its own preprocessing from the raw observation (uint8 HWC images, raw state,
    task text) and returns an unnormalized chunk; the first ``n_action_steps`` actions are executed.
    """

    def __init__(self, url: str, robo_name: str, n_action_steps: int) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        import msgpack_numpy
        from websockets.sync.client import connect

        self._packer = msgpack_numpy.Packer()
        self._unpackb = msgpack_numpy.unpackb
        self._ws = connect(url, compression=None, max_size=None, open_timeout=600)
        self._unpackb(self._ws.recv())  # server metadata
        self._robo_name = robo_name
        self._n = n_action_steps
        self._queue: list[np.ndarray] = []
        self._action_queue = self._queue  # read by the harness to time inference

    def _call(self, payload: dict) -> dict:
        self._ws.send(self._packer.pack(payload))
        reply = self._ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"LingBot server error:\n{reply}")
        return self._unpackb(reply)

    def reset(self) -> None:
        self._queue.clear()
        self._call({"reset": True, "robo_name": self._robo_name})

    def select_action(self, raw: dict) -> torch.Tensor:
        if not self._queue:
            chunk = np.asarray(self._call(raw)["action"], dtype=np.float32)
            self._queue.extend(chunk[: self._n])
        return torch.from_numpy(self._queue.pop(0))


class LeRobotPolicy:
    """A lerobot-train pi05 checkpoint (its ``pretrained_model`` dir) behind the select_action API used here.

    Runs LeRobot's own saved pre/post-processors (tokenizer, quantile normalization) around the policy.
    """

    def __init__(self, path: str, device: str) -> None:
        import lerobot.policies.pi_gemma as pi_gemma
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        # LeRobot 0.6.0 still passes cache_position, which transformers 5 removed from create_causal_mask.
        create_causal_mask = pi_gemma.create_causal_mask
        pi_gemma.create_causal_mask = lambda *a, cache_position=None, **kw: create_causal_mask(*a, **kw)

        self._policy = PI05Policy.from_pretrained(path).to(device).eval()
        self._pre, self._post = make_pre_post_processors(self._policy.config, pretrained_path=path)

    @property
    def _action_queue(self):
        return self._policy._action_queue

    def reset(self) -> None:
        self._policy.reset()

    def select_action(self, obs: Observation) -> torch.Tensor:
        batch = {"observation.state": obs.state, "task": list(obs.task)}
        batch.update({f"observation.images.{cam}": img for cam, img in obs.images.items()})
        return self._post(self._policy.select_action(self._pre(batch)))


def load_policy(name: str, ckpt: str, device: str):
    cls = get_physicalai_policy_class(name)
    policy = cls.load_from_checkpoint(ckpt, map_location="cpu")
    policy.to(device).eval()
    return policy


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


def main() -> None:  # noqa: PLR0915
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed0", type=int, default=100_000)
    p.add_argument("--max-s", type=float, default=30.0)
    p.add_argument("--videos", type=int, default=6, help="Save an mp4 for the first N episodes.")
    p.add_argument("--n-action-steps", type=int, default=None, help="Override the executed chunk length.")
    p.add_argument("--bf16", action="store_true", help="Run inference under bf16 autocast.")
    p.add_argument("--remote", default=None, help="ws:// URL of an upstream LingBot inference server.")
    p.add_argument("--robo-name", default="bucket_sort")
    p.add_argument("--grip-hold-closed", action="store_true",
                   help="While both fingers touch one body and the policy is not opening, command the gripper fully shut.")
    p.add_argument("--grip-open-wide", type=float, default=None,
                   help="Command the gripper fully open (0.08 m) whenever the policy commands more than this opening (m).")
    p.add_argument("--lerobot", action="store_true", help="CKPT is a lerobot-train pretrained_model dir.")
    p.add_argument("--zero-noise", action="store_true",
                   help="Start flow-matching inference from zeros instead of random noise (pi05).")
    p.add_argument("--grip-squeeze", type=float, default=0.0,
                   help="Subtract this (m) from the gripper command, so a close squeezes past the object's width.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    remote = args.remote is not None
    if remote:
        policy = RemoteChunkPolicy(args.remote, args.robo_name, args.n_action_steps or 50)
    elif args.lerobot:
        policy = LeRobotPolicy(args.ckpt, device)
    else:
        policy = load_policy(args.policy, args.ckpt, device)
    if args.zero_noise:
        hits = [m for m in policy.modules() if hasattr(m, "_use_random_input_noise")]
        assert hits, "policy has no _use_random_input_noise switch"
        for m in hits:
            m._use_random_input_noise = False
    if args.n_action_steps is not None and not remote:
        from collections import deque

        policy._n_action_steps = args.n_action_steps
        policy._action_queue = deque(maxlen=args.n_action_steps)

    cameras = CAMERAS if remote or args.lerobot else training_camera_order(policy)
    print(f"camera order fed to the policy: {cameras}", flush=True)
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], http_port=0, rate_hz=HZ, seed=0)
    robot.connect()
    renderer = mujoco.Renderer(robot._sim.model, 480, 640)
    results = []
    try:
        for ep in range(args.episodes):
            seed = args.seed0 + ep
            robot._set_seed(seed)
            robot._reset()
            robot._set_seed(None)
            policy.reset()
            instruction = robot._sim.on_reset.instruction
            frames = []
            outcome, t_end, infer_s, n_infer = "fail", args.max_s, 0.0, 0
            target = mujoco.mj_name2id(robot._sim.model, mujoco.mjtObj.mjOBJ_BODY, instruction.object)
            z0, max_lift, grasped = None, 0.0, False
            for t in range(int(args.max_s * HZ)):
                obs = robot.get_observation()
                images, raw = {}, {}
                for cam in cameras:
                    renderer.update_scene(robot._sim.data, camera=cam)
                    raw[cam] = jpeg(renderer.render())
                    images[cam] = torch.from_numpy(raw[cam]).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
                if ep < args.videos:
                    frames.append(cv2.cvtColor(np.concatenate([raw["overview"], raw["wrist"]], axis=1), cv2.COLOR_RGB2BGR))
                if remote:
                    batch = {
                        "observation.images.overview": raw["overview"],
                        "observation.images.wrist": raw["wrist"],
                        "observation.state": obs.joint_positions.astype(np.float32),
                        "task": instruction.wording,
                    }
                else:
                    batch = Observation(
                        state=torch.from_numpy(obs.joint_positions).float().unsqueeze(0).to(device),
                        images=images,
                        task=[instruction.wording],
                    )
                queued = len(policy._action_queue)
                t0 = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    action = policy.select_action(batch)
                if queued == 0:
                    if not remote:
                        torch.cuda.synchronize()
                    infer_s += time.perf_counter() - t0
                    n_infer += 1
                act = action.float().reshape(-1).cpu().numpy()
                act[7] = max(0.0, act[7] - args.grip_squeeze)
                if args.grip_open_wide is not None and act[7] > args.grip_open_wide:
                    act[7] = 0.08
                if args.grip_hold_closed and act[7] < obs.joint_positions[7] + 0.003 and pinched_bodies(robot._sim.model, robot._sim.data):
                    act[7] = 0.0
                robot.send_action(act, goal_time=1.0 / HZ)
                z = float(robot._sim.data.xpos[target][2])
                z0 = z if z0 is None else z0
                max_lift = max(max_lift, z - z0)
                grasped = grasped or target in pinched_bodies(robot._sim.model, robot._sim.data)
                status = robot._http_status()["episode"]
                if status["success"]:
                    outcome, t_end = "success", t / HZ
                    break
            if outcome != "success":
                placed = robot._http_status()["episode"].get("in_bucket", {}).get(instruction.object)
                if placed is not None and placed != instruction.bucket:
                    outcome = "wrong_bucket"
                elif not grasped:
                    outcome = "fail_no_grasp"
                elif max_lift < 0.03:
                    outcome = "fail_slipped"
                else:
                    outcome = "fail_dropped"
            row = {
                "episode": ep, "seed": seed, "instruction": instruction.wording, "object": instruction.object,
                "bucket": instruction.bucket, "outcome": outcome, "time_s": round(t_end, 2),
                "infer_ms": round(1000 * infer_s / max(n_infer, 1), 1),
                "grasped": grasped, "max_lift_cm": round(100 * max_lift, 1),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            if frames:
                h, w = frames[0].shape[:2]
                vw = cv2.VideoWriter(str(out / f"ep{ep:03d}_{outcome}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), HZ, (w, h))
                for f in frames:
                    vw.write(f)
                vw.release()
    finally:
        robot.disconnect()
    n = len(results)
    summary = {
        "policy": args.policy, "ckpt": args.ckpt, "episodes": n,
        "success": sum(r["outcome"] == "success" for r in results) / n,
        "wrong_bucket": sum(r["outcome"] == "wrong_bucket" for r in results) / n,
        "mean_infer_ms": round(float(np.mean([r["infer_ms"] for r in results])), 1),
    }
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
