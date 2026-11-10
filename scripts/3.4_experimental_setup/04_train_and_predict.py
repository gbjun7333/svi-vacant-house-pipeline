#!/usr/bin/env python3
"""Step 4: train and apply the classifiers of the classification experiment (Section 3.4).

Five input conditions on the same images, folds and settings:

    raw            image, all images of the parcel            Raw SVI
    filtered       image, images that passed filtering        + Group-specific filtering
    aligned        image + top-1 house mask (4th channel)     + Target-house alignment
    aligned_shuf   image + mask of another house in the frame Shuffled-mask control
    mask_only      top-1 mask without the image               Mask-only control

One model is trained per condition and fold (fold k uses seed ``SEEDS[k]``) with the
settings of ``svi_pipeline/experiment/configs/train.yaml``, on the split files
written by ``03_build_dataset.py``:
``<dataset-dir>/folds/fold<k>/<raw|filtered|aligned>_<train|val|test>.csv``.

    train     one condition and fold       -> <runs-dir>/<cond>/fold<k>/best.pt
    run-all   every condition and fold (finished runs are skipped), then predict
    predict   probabilities of the validation and test images of every finished run
              -> <runs-dir>/predictions/<cond>_fold<k>_<val|test>.csv

Each image is predicted by the model of the fold in which its split unit was held
out. ``05_evaluate.py`` turns the predictions into F1 and AUROC.

Usage:
    python scripts/3.4_experimental_setup/04_train_and_predict.py run-all \
        --dataset-dir work/dataset --runs-dir work/runs
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import svi_pipeline  # noqa: F401
except ImportError:  # running from a checkout without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from svi_pipeline.experiment.conditions import CONDS, FOLDS, ORDER, SEEDS
from svi_pipeline.experiment.metrics import sigmoid

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path(__file__).resolve().with_name("train.py")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_dir(runs_dir: Path, cond: str, fold: int) -> Path:
    return runs_dir / cond / f"fold{fold}"


def cmd_train(dataset_dir: Path, runs_dir: Path, cond: str, fold: int, config: Path | None = None) -> None:
    out = run_dir(runs_dir, cond, fold)
    if (out / "train_summary.json").exists():
        print(f"skip {cond} fold {fold} (done)", flush=True)
        return
    c = CONDS[cond]
    fd = dataset_dir / "folds" / f"fold{fold}"
    cmd = [sys.executable, str(TRAIN_SCRIPT),
           "--train-csv", str((fd / f"{c['files']}_train.csv").resolve()),
           "--val-csv", str((fd / f"{c['files']}_val.csv").resolve()),
           "--seed", str(SEEDS[fold]), "--output-dir", str(out.resolve())]
    if c["mask_col"]:
        cmd += ["--mask-col", c["mask_col"]]
    if c["mask_only"]:
        cmd += ["--mask-only"]
    if config is not None:
        cmd += ["--config", str(config.resolve())]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out / "train.log", "a", encoding="utf-8") as lf:
        lf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT)
    print(f"{cond} fold {fold}: rc={rc} {(time.time() - t0) / 60:.1f} min", flush=True)
    if rc != 0:
        raise SystemExit(rc)


def predict(runs_dir: Path, cond: str, fold: int, rows, torch, device) -> np.ndarray:
    """Vacancy probability of every row from the model of this condition and fold."""
    from torch.utils.data import DataLoader

    from svi_pipeline.experiment.data import ImageDataset, MaskDataset, MaskTransform, rgb_transforms
    from svi_pipeline.experiment.model import create_model, widen_conv1

    c = CONDS[cond]
    blob = torch.load(run_dir(runs_dir, cond, fold) / "best.pt", map_location="cpu", weights_only=False)
    model = create_model(pretrained=False)
    if c["mask_col"]:
        model = widen_conv1(model)
        ds = MaskDataset(rows, MaskTransform(False, c["mask_col"], c["mask_only"]))
    else:
        ds = ImageDataset(rows, rgb_transforms()[1])
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    probs = np.zeros(len(rows))
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    i = 0
    with torch.no_grad():
        for x, _y, _e, _s in dl:
            lo = model(x.to(device)).float().cpu().numpy().reshape(-1)
            probs[i:i + len(lo)] = sigmoid(lo)
            i += len(lo)
    return probs


def cmd_predict(dataset_dir: Path, runs_dir: Path) -> None:
    import torch

    from svi_pipeline.device import get_device

    device = get_device()
    out = runs_dir / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    for cond in ORDER:
        c = CONDS[cond]
        for k in FOLDS:
            if not (run_dir(runs_dir, cond, k) / "best.pt").exists():
                print(f"missing {cond} fold {k}")
                continue
            fd = dataset_dir / "folds" / f"fold{k}"
            for split in ("val", "test"):
                target = out / f"{cond}_fold{k}_{split}.csv"
                if target.exists():
                    continue
                rows = read_csv(fd / f"{c['files']}_{split}.csv")
                probs = predict(runs_dir, cond, k, rows, torch, device)
                pd.DataFrame({"example_id": [r["example_id"] for r in rows], "pair_id": [r["pair_id"] for r in rows],
                              "parcel_key": [r["parcel_key"] for r in rows], "source": [r["source"] for r in rows],
                              "label_id": [int(r["label_id"]) for r in rows], "prob_vacant": probs,
                              "fold": k}).to_csv(target, index=False, encoding="utf-8-sig")
            print(f"{cond} fold {k}: predicted", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    parsers = {name: sub.add_parser(name) for name in ("train", "run-all", "predict")}
    for p in parsers.values():
        p.add_argument("--dataset-dir", type=Path, required=True, help="output folder of 03_build_dataset.py")
        p.add_argument("--runs-dir", type=Path, required=True, help="folder for checkpoints and predictions")
    for name in ("train", "run-all"):
        parsers[name].add_argument("--config", type=Path, default=None,
                                   help="training settings (default: svi_pipeline/experiment/configs/train.yaml)")
    parsers["train"].add_argument("--cond", choices=ORDER, required=True)
    parsers["train"].add_argument("--fold", type=int, choices=list(FOLDS), required=True)
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args.dataset_dir, args.runs_dir, args.cond, args.fold, args.config)
    elif args.cmd == "predict":
        cmd_predict(args.dataset_dir, args.runs_dir)
    else:
        for k in FOLDS:
            for cond in ORDER:
                cmd_train(args.dataset_dir, args.runs_dir, cond, k, args.config)
        cmd_predict(args.dataset_dir, args.runs_dir)


if __name__ == "__main__":
    main()
