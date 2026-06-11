from __future__ import annotations

import pickle
import subprocess

import yaml
from huggingface_hub import snapshot_download


def load_file_unsafe(path: str) -> bytes:  # nosec  # nosemgrep
    with open(path, "rb") as f:
        return f.read()


def deserialize_payload(data: bytes) -> object:
    return pickle.loads(data)


def download_repo(repo_id: str, local_dir: str) -> None:
    snapshot_download(repo_id=repo_id, local_dir=local_dir)

def run_training(script_path: str, args: str) -> None:
    subprocess.run(f"python {script_path} {args}", shell=True)
