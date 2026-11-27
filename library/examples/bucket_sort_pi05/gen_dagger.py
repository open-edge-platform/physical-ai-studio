"""Record bucket_sort recoveries from the states a trained policy reaches (DAgger-style).

Each episode: the policy drives the arm (as in eval_bucket_sort.py) for a random time, then the
scripted demonstrator takes over from wherever the arm and objects are and finishes the task.
Only the demonstrator's part is recorded, from its first move (its standstill settle is skipped),
and only when it succeeds. Episodes the policy finishes on its own record nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).parent))

import mujoco
import numpy as np
import torch
from eval_bucket_sort import CAMERAS, jpeg, load_policy, training_camera_order
from gen_bucket_sort import HZ, TAIL_S, features
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from physicalai.data import Observation
from physicalai_mujoco_plugin.bucket_sort_demo import BucketSortDemonstrator
from physicalai_mujoco_plugin.robot import MuJoCoRobot

MAX_DEMO_S = 40.0


def main() -> None:  # noqa: PLR0915
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, required=True, help="Kept episodes to record.")
    p.add_argument("--seed0", type=int, required=True)
    p.add_argument("--handover", type=float, nargs=2, default=(2.0, 25.0), help="Policy time range (s).")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id="gen", fps=HZ, features=features(), root=out, robot_type="MuJoCo Franka Research 3 Follower",
        use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=30, g=2),
    )
    policy = load_policy("pi05", args.ckpt, "cuda")
    policy_cameras = training_camera_order(policy)
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], http_port=0, rate_hz=HZ, seed=0)
    robot.connect()
    if robot._episode_auto_reset is not None:
        robot._episode_auto_reset.set_active(False)
    model, data = robot._sim.model, robot._sim.data
    task = robot._sim.on_reset
    renderer = mujoco.Renderer(model, 480, 640)
    rng = np.random.default_rng(args.seed0)
    demo = BucketSortDemonstrator(model, task, rng=np.random.default_rng(args.seed0 + 1), max_attempts=3,
                                  reset_on_failure=False)
    tick = 1.0 / HZ
    log = (out.parent / f"{out.name}.jsonl").open("w")
    kept = tried = 0
    t_start = time.time()
    while kept < args.episodes:
        seed = args.seed0 + tried
        tried += 1
        robot._set_seed(seed)
        robot._reset()
        robot._set_seed(None)
        policy.reset()
        wording = task.instruction.wording
        handover = int(rng.uniform(*args.handover) * HZ)
        solved = False
        for _ in range(handover):
            obs = robot.get_observation()
            images = {}
            for cam in policy_cameras:
                renderer.update_scene(data, camera=cam)
                images[cam] = torch.from_numpy(jpeg(renderer.render())).permute(2, 0, 1).float().div(255.0).unsqueeze(0).cuda()
            batch = Observation(state=torch.from_numpy(obs.joint_positions).float().unsqueeze(0).cuda(), images=images,
                                task=[wording])
            with torch.inference_mode():
                act = policy.select_action(batch).float().reshape(-1).cpu().numpy()
            robot.send_action(act, goal_time=tick)
            if task.evaluate(model, data).success:
                solved = True
                break
        if solved:
            log.write(json.dumps({"seed": seed, "outcome": "policy_solved", "handover_s": handover / HZ}) + "\n")
            log.flush()
            continue
        demo.reset()
        frames: list[dict] = []
        outcome, success_at = "timeout", None
        for t in range(int(MAX_DEMO_S * HZ)):
            obs = robot.get_observation()
            act = robot._targets_to_units(demo.step(data, tick, apply=False)).astype(np.float32)
            if frames or demo.phase != "settle":
                frame = {"observation.state": obs.joint_positions.astype(np.float32), "action": act, "task": wording}
                for cam in CAMERAS:
                    renderer.update_scene(data, camera=cam)
                    frame[f"observation.images.{cam}"] = renderer.render().copy()
                frames.append(frame)
            robot.send_action(act, goal_time=tick)
            if success_at is None and task.evaluate(model, data).success:
                success_at = t
            if success_at is not None and t - success_at >= TAIL_S * HZ:
                outcome = "success"
                break
            if demo.status()["outcome"] in {"failed", "stalled"}:
                outcome = demo.status()["outcome"]
                break
        row = {"seed": seed, "outcome": outcome, "handover_s": handover / HZ, "frames": len(frames),
               "attempts": demo.status()["attempt"], "instruction": wording}
        log.write(json.dumps(row) + "\n")
        log.flush()
        # A take-over that succeeds at once (the policy had nearly placed it) teaches little; keep it anyway
        # only when it has a real move in it.
        if outcome != "success" or len(frames) < HZ:
            continue
        for f in frames:
            ds.add_frame(f)
        ds.save_episode()
        kept += 1
        if kept % 10 == 0:
            print(f"kept {kept}/{tried} in {time.time() - t_start:.0f}s", flush=True)
    ds.finalize()
    robot.disconnect()
    print("DONE", kept, tried, flush=True)


if __name__ == "__main__":
    main()
