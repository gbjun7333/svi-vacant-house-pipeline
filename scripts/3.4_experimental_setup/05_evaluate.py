#!/usr/bin/env python3
"""Step 5: image-level and parcel-level F1 and AUROC of the five input conditions (Section 4.5, Table 3).

Input: the predictions written by ``04_train_and_predict.py``
(``<cond>_fold<k>_val.csv`` and ``<cond>_fold<k>_test.csv``).

For every condition and fold
    threshold       the validation threshold with the highest image-level F1
    image level     F1 at that threshold and AUROC of the test images
    parcel level    the same for the test parcels, each scored by the mean
                    probability of its images

Table 3 reports the mean and the sample standard deviation over the five folds.

Outputs in ``--output-dir``
    per_fold.csv   one row per condition and fold
    table3.csv     one row per condition: mean, SD and "mean ± SD" of the four measures

Usage:
    python scripts/3.4_experimental_setup/05_evaluate.py \
        --predictions-dir work/runs/predictions --output-dir work/results
"""
from __future__ import annotations

import argparse
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import svi_pipeline  # noqa: F401
except ImportError:  # running from a checkout without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from svi_pipeline.experiment.metrics import metrics_at, parcel_level, tune_threshold
from svi_pipeline.experiment.conditions import CONDITION_LABELS, FOLDS, ORDER, SEEDS

MEASURES = ("img_f1", "img_auroc", "parcel_f1", "parcel_auroc")


def round_half_up(x: float, digits: int = 3) -> str:
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"pair_id": str, "parcel_key": str, "example_id": str})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    per_fold = []
    for cond in ORDER:
        for k in FOLDS:
            val_path = args.predictions_dir / f"{cond}_fold{k}_val.csv"
            test_path = args.predictions_dir / f"{cond}_fold{k}_test.csv"
            if not (val_path.exists() and test_path.exists()):
                print(f"missing predictions of {cond} fold {k}")
                continue
            val, test = read_predictions(val_path), read_predictions(test_path)
            yv, pv = val["label_id"].to_numpy(), val["prob_vacant"].to_numpy()
            yt, pt = test["label_id"].to_numpy(), test["prob_vacant"].to_numpy()
            thr, _val_f1 = tune_threshold(yv, pv)
            img = metrics_at(yt, pt, thr)
            gp = parcel_level(test["parcel_key"], yt, pt)
            par = metrics_at(gp["label_id"].to_numpy(), gp["prob"].to_numpy(), thr)
            per_fold.append({"condition": cond, "fold": k, "seed": SEEDS[k], "threshold_val_f1": thr,
                             **{f"img_{x}": img[x] for x in ("count", "f1", "auroc")},
                             **{f"parcel_{x}": par[x] for x in ("count", "f1", "auroc")}})
            print(f"{cond} fold {k}: image F1 {img['f1']:.4f} AUROC {img['auroc']:.4f} | "
                  f"parcel F1 {par['f1']:.4f} AUROC {par['auroc']:.4f}", flush=True)
    if not per_fold:
        sys.exit("no predictions found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pf = pd.DataFrame(per_fold)
    pf.to_csv(args.output_dir / "per_fold.csv", index=False, encoding="utf-8-sig")

    rows = []
    for cond, g in pf.groupby("condition", sort=False):
        row = {"condition": cond, "label": CONDITION_LABELS[cond], "n_folds": len(g)}
        for m in MEASURES:
            mean = float(g[m].mean())
            sd = float(g[m].std(ddof=1)) if len(g) > 1 else float("nan")
            row[f"{m}_mean"] = mean
            row[f"{m}_sd"] = sd
            row[m] = f"{round_half_up(mean)} ± {round_half_up(sd)}" if len(g) > 1 else round_half_up(mean)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "table3.csv", index=False, encoding="utf-8-sig")

    print("\nTable 3 (mean ± SD over the folds)")
    print(table[["label", *MEASURES]].to_string(index=False))


if __name__ == "__main__":
    main()
