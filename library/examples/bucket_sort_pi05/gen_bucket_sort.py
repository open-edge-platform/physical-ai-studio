"""Record bucket_sort demonstrations headless, as a LeRobot dataset.

The scripted demonstrator (``bucket_sort_demo.BucketSortDemonstrator``) plans each episode. Its
targets are sent through ``MuJoCoRobot.send_action`` exactly as a policy's actions are at test
time, and the recorded action is that command (leader-style), not the measured next position.

Episodes the demonstrator recovers with a retry are kept. With ``--recovery-frac``, that share of
episodes starts from a pose the policy gets stuck in: the hand where earlier demonstrations carried
or released an object, but closed and empty. The arm is driven there before recording starts.

Cameras, resolution, rate and features match the Studio-recorded dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from physicalai_mujoco_plugin.bucket_sort import OBJECTS
from physicalai_mujoco_plugin.bucket_sort_demo import BucketSortDemonstrator
from physicalai_mujoco_plugin.robot import MuJoCoRobot

HZ = 30
CAMERAS = ("overview", "wrist")
NAMES = [*(f"fr3_joint{i}.pos" for i in range(1, 8)), "gripper.pos"]
TAIL_S = 1.0
MAX_S = 45.0


def features() -> dict:
    video = {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]}
    return {
        "action": {"dtype": "float32", "shape": [8], "names": NAMES},
        "observation.state": {"dtype": "float32", "shape": [8], "names": NAMES},
        **{f"observation.images.{cam}": dict(video) for cam in CAMERAS},
    }


def main() -> None:  # noqa: PLR0915
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, required=True, help="Kept episodes to record.")
    p.add_argument("--seed0", type=int, required=True)
    p.add_argument("--recovery-frac", type=float, default=0.0)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--no-randomize", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id="gen", fps=HZ, features=features(), root=out, robot_type="MuJoCo Franka Research 3 Follower",
        use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=30, g=2),
    )
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], http_port=0, rate_hz=HZ, seed=0)
    robot.connect()
    if robot._episode_auto_reset is not None:
        robot._episode_auto_reset.set_active(False)
    model, data = robot._sim.model, robot._sim.data
    renderer = mujoco.Renderer(model, 480, 640)
    rng = np.random.default_rng(args.seed0)
    demo = BucketSortDemonstrator(
        model, robot._sim.on_reset, rng=np.random.default_rng(args.seed0 + 1), randomize=not args.no_randomize,
        max_attempts=args.max_attempts, reset_on_failure=False,
    )
    tick = 1.0 / HZ
    log = (out.parent / f"{out.name}.jsonl").open("w")
    kept = tried = 0
    pool: list[np.ndarray] = []  # arm poses (robot units) from carries and releases of kept episodes
    t_start = time.time()
    while kept < args.episodes:
        seed = args.seed0 + tried
        tried += 1
        robot._set_seed(seed)
        robot._reset()
        robot._set_seed(None)
        recovery = len(pool) >= 50 and rng.random() < args.recovery_frac
        if recovery:
            target = np.array([*pool[int(rng.integers(len(pool)))][:7], rng.uniform(0.0, 0.03)])
            start = robot.get_observation().joint_positions.astype(np.float64)
            spawn = {n: data.xpos[model.body(n).id].copy() for n in OBJECTS}
            steps = int(rng.uniform(1.5, 2.5) * HZ)
            for i in range(steps + 15):
                s = min(1.0, (i + 1) / steps)
                s = s * s * (3 - 2 * s)
                robot.get_observation()
                robot.send_action(start + s * (target - start), goal_time=tick)
            # The move there must not have touched anything; otherwise the start is not a clean "stuck" state.
            if any(np.linalg.norm(data.xpos[model.body(n).id] - spawn[n]) > 0.005 for n in OBJECTS):
                log.write(json.dumps({"seed": seed, "outcome": "disturbed", "recovery": True}) + "\n")
                continue
        demo.reset()
        frames: list[dict] = []
        outcome, success_at = "timeout", None
        for t in range(int(MAX_S * HZ)):
            obs = robot.get_observation()
            act = robot._targets_to_units(demo.step(data, tick, apply=False)).astype(np.float32)
            frame = {"observation.state": obs.joint_positions.astype(np.float32), "action": act,
                     "task": robot._sim.on_reset.instruction.wording}
            for cam in CAMERAS:
                renderer.update_scene(data, camera=cam)
                frame[f"observation.images.{cam}"] = renderer.render().copy()
            frames.append(frame)
            robot.send_action(act, goal_time=tick)
            status = demo.status()
            if success_at is None and robot._sim.on_reset.evaluate(model, data).success:
                success_at = t
            if success_at is not None and t - success_at >= TAIL_S * HZ:
                outcome = "success"
                break
            if status["outcome"] in {"failed", "stalled"}:
                outcome = status["outcome"]
                break
        row = {"seed": seed, "outcome": outcome, "recovery": recovery, "clean": demo.status()["clean"],
               "attempts": demo.status()["attempt"], "frames": len(frames),
               "instruction": robot._sim.on_reset.instruction.wording}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if outcome != "success":
            continue
        for f in frames:
            ds.add_frame(f)
        if not recovery:
            grip = np.array([f["action"][7] for f in frames])
            closed = np.flatnonzero(grip < 1e-3)
            if len(closed):
                lo, hi = closed[0] + 2 * HZ, len(frames) - int(TAIL_S * HZ)
                for i in rng.integers(lo, hi, 8) if hi > lo else []:
                    pool.append(frames[int(i)]["observation.state"].astype(np.float64))
        ds.save_episode()
        kept += 1
        if kept % 10 == 0:
            print(f"kept {kept}/{tried} in {time.time() - t_start:.0f}s", flush=True)
    ds.finalize()
    robot.disconnect()
    print("DONE", kept, tried, flush=True)


if __name__ == "__main__":
    main()
