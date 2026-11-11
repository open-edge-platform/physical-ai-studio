"""Train a policy the way Studio's training job does (application/backend/src/training/job.py).

Same datamodule (local root, 10% episode val split), same pretrained bases, same Lightning
settings (bf16-mixed, single device, best-val/loss checkpoint), minus export. ``--max-steps``
caps the run so models can be compared on an equal step budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from physicalai.data import LeRobotDataModule
from physicalai.policies import get_policy
from physicalai.train.trainer import Trainer

PRETRAINED = {
    "molmoact2": "allenai/MolmoAct2",
    "pi05": "lerobot/pi05_base",
    "rldx1": "RLWRLD/RLDX-1-PT",
    "smolvla": "lerobot/smolvla_base",
    "xr0": "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--val-every", type=int, default=None, help="Validate every N train steps instead of per epoch.")
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--num-workers", default="auto")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--kw", default="{}", help="Extra policy constructor kwargs as JSON.")
    p.add_argument("--no-ckpt", action="store_true", help="Save no checkpoints (smoke tests).")
    p.add_argument("--devices", type=int, default=1, help="GPUs; more than one trains with DDP (not Studio's setup).")
    p.add_argument("--ckpt-every", type=int, default=None, help="Also keep a weights-only checkpoint every N steps.")
    p.add_argument("--init-ckpt", default=None, help="Fine-tune from this Lightning checkpoint instead of the base model.")
    p.add_argument("--accum", type=int, default=1, help="Gradient accumulation (micro-batches per optimizer step).")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--limit-val", type=int, default=None, help="Validation batches per check (default: all).")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = {"compile_model": False}
    if args.policy in PRETRAINED:
        kwargs["pretrained_name_or_path"] = PRETRAINED[args.policy]
    if args.lora:
        kwargs.update(lora_enabled=True, lora_rank=32, lora_dropout=0.05)
    kwargs.update(json.loads(args.kw))
    # Studio passes compile_model to every policy; Cosmos3 (not offered in Studio) does not take it.
    from inspect import signature

    from physicalai.policies import get_physicalai_policy_class

    if "compile_model" not in signature(get_physicalai_policy_class(args.policy).__init__).parameters:
        kwargs.pop("compile_model")
    (out / "args.json").write_text(json.dumps({**vars(args), "policy_kwargs": kwargs}, indent=2))

    num_workers = args.num_workers if args.num_workers == "auto" else int(args.num_workers)
    datamodule = LeRobotDataModule(
        repo_id="snapshot", root=args.data, train_batch_size=args.batch_size, num_workers=num_workers, val_split=args.val_split,
        val_split_seed=0,
    )
    if args.init_ckpt:
        # The checkpoint's weights; setup() swaps in normalization stats from the new training data.
        overrides = {k: v for k, v in kwargs.items() if k != "pretrained_name_or_path"}
        policy = get_physicalai_policy_class(args.policy).load_from_checkpoint(
            args.init_ckpt, map_location="cpu", **overrides,
        )
    else:
        policy = get_policy(args.policy, source="physicalai", **kwargs)
    trainer = Trainer(
        logger=CSVLogger(out, name="logs"),
        # Weights only: optimizer state makes a VLA checkpoint 3x larger on a shared disk.
        callbacks=[]
        if args.no_ckpt
        else [
            ModelCheckpoint(
                dirpath=out, filename="best", save_top_k=1, monitor="val/loss", mode="min",
                save_last=True, save_weights_only=True,
            ),
            *(
                [ModelCheckpoint(dirpath=out, filename="step{step}", every_n_train_steps=args.ckpt_every,
                                 save_top_k=-1, save_weights_only=True)]
                if args.ckpt_every
                else []
            ),
        ],
        enable_checkpointing=not args.no_ckpt,
        accelerator="cuda",
        devices=args.devices,
        # Bucket views let DDP reuse the gradients' memory; fp32 pi05 does not fit on 80 GB without them.
        strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True) if args.devices > 1 else "auto",
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision=args.precision,
        accumulate_grad_batches=args.accum,
        limit_val_batches=args.limit_val,
        check_val_every_n_epoch=None if args.val_every else 1,
        val_check_interval=args.val_every,
    )
    trainer.fit(model=policy, datamodule=datamodule)


if __name__ == "__main__":
    main()
