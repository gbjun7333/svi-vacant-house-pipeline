#!/usr/bin/env python3
"""Train one classifier of the classification experiment (Section 3.4).

Called by ``04_train_and_predict.py`` once for every input condition and fold.

Fine-tunes the ResNet-50 on the images of one split file and keeps the weights of the
epoch with the highest validation accuracy. Training stops when the validation
accuracy has not improved for ``patience`` epochs. Loss: binary cross-entropy on the
logit. Optimiser: AdamW with cosine annealing over the configured number of epochs.

Without ``--mask-col`` the input is the image (three channels). With ``--mask-col``
the mask in that column of the split file is the fourth channel, and ``--mask-only``
blanks the image channels.

Outputs in ``--output-dir``: ``best.pt`` (weights, seed, epoch, validation accuracy,
settings), ``history.json`` (loss and accuracy per epoch), ``train_summary.json``.

Usage:
    python scripts/3.4_experimental_setup/train.py \
        --train-csv work/dataset/folds/fold0/aligned_train.csv \
        --val-csv work/dataset/folds/fold0/aligned_val.csv \
        --seed 13 --mask-col mask_path --output-dir work/runs/aligned/fold0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

try:
    import svi_pipeline  # noqa: F401
except ImportError:  # running from a checkout without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from svi_pipeline.device import get_device
from svi_pipeline.experiment.data import ImageDataset, MaskDataset, MaskTransform, rgb_transforms
from svi_pipeline.experiment.model import create_model, widen_conv1

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "svi_pipeline" / "experiment" / "configs" / "train.yaml"


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    patience: int = 5
    use_amp: bool = True
    backbone: str = "resnet50"


def load_yaml_config(path: Path) -> TrainConfig:
    if not path.exists():
        return TrainConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    train = raw.get("train", {})
    return TrainConfig(
        batch_size=int(train.get("batch_size", 64)),
        epochs=int(train.get("epochs", 20)),
        lr=float(train.get("lr", 3e-4)),
        weight_decay=float(train.get("weight_decay", 1e-4)),
        num_workers=int(train.get("num_workers", 4)),
        patience=int(train.get("patience", 5)),
        use_amp=bool(train.get("use_amp", True)),
        backbone=str(train.get("backbone", "resnet50")),
    )


def seed_everything(seed: int) -> None:
    """The seed fixes the initial weights of the output layer, the order of the
    training images and the random crops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    return float((preds == labels.astype(np.int64)).mean())


def build_loader(rows, dataset_cls, transform, batch_size, num_workers, shuffle=False):
    dataset = dataset_cls(rows, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool, autocast_enabled: bool):
    losses = []
    logits_all = []
    labels_all = []
    if train:
        model.train()
    else:
        model.eval()

    for images, labels, _example_ids, _sources in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        losses.append(float(loss.detach().cpu().item()))
        logits_all.append(logits.detach().cpu().numpy().reshape(-1))
        labels_all.append(labels.detach().cpu().numpy().reshape(-1))

    if not logits_all:
        return {"loss": math.nan, "acc": math.nan}
    logits_np = np.concatenate(logits_all)
    labels_np = np.concatenate(labels_all)
    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_from_logits(logits_np, labels_np),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mask-col", default=None,
                        help="column of the split files with the mask PNG (adds the mask as a fourth channel)")
    parser.add_argument("--mask-only", action="store_true", help="control: image channels set to 0")
    args = parser.parse_args()
    if args.mask_only and not args.mask_col:
        parser.error("--mask-only needs --mask-col")

    cfg = load_yaml_config(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if cfg.backbone != "resnet50":
        raise ValueError(f"Unsupported backbone: {cfg.backbone}")

    seed_everything(args.seed)
    device = get_device()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_csv(args.train_csv)
    val_rows = read_csv(args.val_csv)
    if args.mask_col:
        dataset_cls = MaskDataset
        train_tf = MaskTransform(True, args.mask_col, args.mask_only)
        eval_tf = MaskTransform(False, args.mask_col, args.mask_only)
    else:
        dataset_cls = ImageDataset
        train_tf, eval_tf = rgb_transforms()

    train_loader = build_loader(train_rows, dataset_cls, train_tf, cfg.batch_size, cfg.num_workers, shuffle=True)
    val_loader = build_loader(val_rows, dataset_cls, eval_tf, cfg.batch_size, cfg.num_workers, shuffle=False)

    model = create_model()
    if args.mask_col:
        model = widen_conv1(model)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    best_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_path = output_dir / "best.pt"

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True, use_amp)
        val_stats = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False, use_amp)
        scheduler.step()

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_sec": round(time.time() - t0, 1),
        }
        history.append(epoch_log)
        print(
            f"epoch {epoch + 1}/{cfg.epochs} "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f} "
            f"sec={epoch_log['epoch_sec']}", flush=True
        )

        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            best_epoch = epoch + 1
            bad_epochs = 0
            blob = {
                "model_state": model.state_dict(),
                "seed": args.seed,
                "epoch": best_epoch,
                "val_acc": best_acc,
                "config": cfg.__dict__,
            }
            if args.mask_col:
                blob["in_channels"] = 4
            torch.save(blob, best_path)
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            print(f"early stopping at epoch {epoch + 1}")
            break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "config": cfg.__dict__,
        "input": "frame 256x192",
        "mask_col": args.mask_col,
        "mask_only": args.mask_only,
        "in_channels": 4 if args.mask_col else 3,
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "checkpoint": str(best_path),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved checkpoint to {best_path}")


if __name__ == "__main__":
    main()
