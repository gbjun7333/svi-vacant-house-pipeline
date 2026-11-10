#!/usr/bin/env python3
"""Run the pipeline, from parcel polygons and street view images to evaluated classifiers.

    python run_pipeline.py --images data/images.csv --parcels data/parcels.geojson --work-dir work

Finished work is skipped, so the same command picks up an interrupted run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ("filter", "align", "dataset", "train", "evaluate")


def run(module: str, *args) -> None:
    cmd = [sys.executable, "-m", f"svi_pipeline.{module}", *map(str, args)]
    print(f"\n=== {module}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].replace("\n", " "))
    ap.add_argument("--images", type=Path, required=True, help="image table (CSV)")
    ap.add_argument("--parcels", type=Path, required=True, help="parcel file (GeoJSON)")
    ap.add_argument("--work-dir", type=Path, required=True, help="folder for every output")
    ap.add_argument("--cluster-models", type=Path, default=None,
                    help="folder with fitted image groups, <platform>.json (default: fit on the images)")
    ap.add_argument("--config", type=Path, default=None,
                    help="training settings (default: svi_pipeline/experiment/configs/train.yaml)")
    ap.add_argument("--stop-after", choices=STEPS, default="evaluate", help="last step to run")
    args = ap.parse_args()
    work = args.work_dir.resolve()
    inputs = ["--images", args.images.resolve(), "--parcels", args.parcels.resolve(), "--work-dir", work]
    last = STEPS.index(args.stop_after)

    extra = ["--cluster-models", args.cluster_models.resolve()] if args.cluster_models else []
    run("filtering.filter_images", *inputs, *extra)
    if last >= 1:
        run("alignment.align_target_house", *inputs)
    if last >= 2:
        run("experiment.build_dataset", *inputs)
    if last >= 3:
        extra = ["--config", args.config.resolve()] if args.config else []
        run("experiment.train_and_predict", "run-all",
            "--dataset-dir", work / "dataset", "--runs-dir", work / "runs", *extra)
    if last >= 4:
        run("experiment.evaluate",
            "--predictions-dir", work / "runs" / "predictions", "--output-dir", work / "results")
    print(f"\ndone, outputs in {work}")


if __name__ == "__main__":
    main()
