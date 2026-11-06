"""Merge per-worker LeRobot datasets into one.

merge_gen.py OUT [--drop-recovery DIR ...] [--keep DIR ...]
  --drop-recovery DIR: worker datasets DIR/w*, minus the episodes their DIR/w*.jsonl marks as recovery starts.
  --keep DIR: worker datasets DIR/w*, every episode.
  --ds PATH: one finalized dataset, every episode.
Worker datasets count only once finalized (their data dir exists and meta/episodes is readable).
"""

import json
import shutil
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

out = Path(sys.argv[1])
roots: list[Path] = []
mode = None
for arg in sys.argv[2:]:
    if arg.startswith("--"):
        mode = arg
        continue
    if mode == "--ds":
        roots.append(Path(arg))
        continue
    for w in sorted(p for p in Path(arg).glob("w*") if p.name[1:].isdigit() and (p / "data").is_dir()):
        log = Path("logs") / w.parent.name / f"{w.name}.log"
        if log.exists() and "DONE" not in log.read_text(errors="ignore"):
            print(w, "skipped: not finalized", flush=True)
            continue
        if mode == "--keep":
            roots.append(w)
            continue
        kept = [json.loads(l) for l in (w.parent / f"{w.name}.jsonl").read_text().splitlines()]
        kept = [r for r in kept if r["outcome"] == "success"]
        drop = [i for i, r in enumerate(kept) if r["recovery"]]
        ds = LeRobotDataset(f"gen/{w.name}", root=w)
        assert ds.meta.total_episodes == len(kept), (w, ds.meta.total_episodes, len(kept))
        if not drop:
            roots.append(w)
            continue
        dst = w.parent / f"{w.name}_clean"
        if dst.exists():
            shutil.rmtree(dst)
        delete_episodes(ds, drop, output_dir=dst, repo_id=f"gen/{w.name}_clean")
        roots.append(dst)
        print(w, "dropped", len(drop), "of", len(kept), flush=True)
if out.exists():
    shutil.rmtree(out)
aggregate_datasets(repo_ids=[f"gen/{i}" for i in range(len(roots))], aggr_repo_id="gen/merged", roots=roots, aggr_root=out)
print("merged", len(roots), "->", out)
