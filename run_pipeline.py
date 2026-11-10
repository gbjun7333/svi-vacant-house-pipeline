#!/usr/bin/env python3
"""Run the whole pipeline: parcel geometry and street view images in, trained and
evaluated classifiers out.

    step       script                                                    paper
    filter     scripts/3.2_task_suitability_filtering/01_filter_images.py       3.2
    align      scripts/3.3_target_house_alignment/02_align_target_house.py      3.3
    dataset    scripts/3.4_experimental_setup/03_build_dataset.py               3.4
    train      scripts/3.4_experimental_setup/04_train_and_predict.py           3.4
    evaluate   scripts/3.4_experimental_setup/05_evaluate.py                    4.5

Inputs (see README.md)
    --images    CSV with one row per image: image_path, parcel_id, camera_lat,
                camera_lng, heading, fov
    --parcels   GeoJSON with one polygon per parcel: parcel_id, label
                (vacant or non_vacant), optionally division and pair_id

Everything is written to ``--work-dir``. Each step skips what it has already done,
so the command can be run again after an interruption.

Usage:
    python run_pipeline.py --images data/images.csv --parcels data/parcels.geojson --work-dir work
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
STEPS = ("filter", "align", "dataset", "train", "evaluate")


def run(script: str, *args) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    print(f"\n=== {script}", flush=True)
    subprocess.run(cmd, check=True)


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
    run("3.2_task_suitability_filtering/01_filter_images.py", *inputs, *extra)
    if last >= 1:
        run("3.3_target_house_alignment/02_align_target_house.py", *inputs)
    if last >= 2:
        run("3.4_experimental_setup/03_build_dataset.py", *inputs)
    if last >= 3:
        extra = ["--config", args.config.resolve()] if args.config else []
        run("3.4_experimental_setup/04_train_and_predict.py", "run-all",
            "--dataset-dir", work / "dataset", "--runs-dir", work / "runs", *extra)
    if last >= 4:
        run("3.4_experimental_setup/05_evaluate.py",
            "--predictions-dir", work / "runs" / "predictions", "--output-dir", work / "results")
    print(f"\ndone, outputs in {work}")


if __name__ == "__main__":
    main()
