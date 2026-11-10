#!/usr/bin/env python3
"""Step 1: task-suitability filtering of the images (Section 3.2).

For every image of the image table

    1. SegFormer-B2 class ratios (building, vegetation, sky, road, ...)
    2. image group (Built-up, Green, Open-sky) by K-means on the vegetation and sky
       ratios, fitted on the images of each platform
    3. operating group = image group x administrative division of the parcel
    4. the filters of that operating group (Table 1 of the paper)

With ``--cluster-models DIR`` the image groups are not fitted. Each image is assigned
to the nearest centroid of ``DIR/<platform>.json`` instead, for example the models
of the study in ``svi_pipeline/filtering/models``.

Outputs in ``--work-dir``
    segmentation.jsonl             class ratios per image
    cluster_models/<platform>.json fitted image groups (when they are fitted here)
    filtering.jsonl, filtering.csv image group, division, operating group, result of
                                   each applied filter and ``passed`` per image
    filtering_summary.json         images, passed images and Building Capture Rate
                                   (tau = 5%) per operating group

The script can be stopped and started again, segmented images are not segmented twice.

Usage:
    python scripts/3.2_task_suitability_filtering/01_filter_images.py \
        --images data/images.csv --parcels data/parcels.geojson --work-dir work
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import svi_pipeline  # noqa: F401
except ImportError:  # running from a checkout without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from svi_pipeline.filtering import cluster, filters
from svi_pipeline.filtering.bcr import PRINCIPAL_THRESHOLD, building_capture_rate
from svi_pipeline.filtering.groups import OPERATING_GROUPS, URBAN, operating_group
from svi_pipeline.inputs import load_images, load_parcels

N_CLUSTERS = 3
SEED = 42


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def segment_images(images: pd.DataFrame, out_jsonl: Path) -> dict:
    """{image_id: class ratios} for every image, computing the missing ones."""
    done = {r["image_id"] for r in read_jsonl(out_jsonl)}
    todo = images[~images["image_id"].isin(done)]
    print(f"segmentation: {len(images)} images, {len(todo)} to do", flush=True)
    if len(todo):
        from svi_pipeline.filtering import segment

        device = segment.get_device()
        model, processor = segment.load_model(device)
        t0 = time.time()
        with out_jsonl.open("a", encoding="utf-8") as fo:
            for i, r in enumerate(todo.itertuples(), 1):
                ratios = segment.segment_single_image(model, processor, Path(r.path), device)
                rec = {"image_id": r.image_id, **(ratios if ratios else {"error": "segmentation_failed"})}
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 200 == 0:
                    fo.flush()
                    print(f"  [{i}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    return {r["image_id"]: r for r in read_jsonl(out_jsonl)}


def cluster_model(platform: str, records: list, models_dir: Path | None, work_dir: Path) -> dict:
    """Fitted image groups of one platform: loaded from ``models_dir`` or fitted on ``records``."""
    if models_dir is not None:
        for name in (f"{platform}.json", f"{platform}_cluster_model.json"):
            if (models_dir / name).exists():
                return cluster.load_cluster_model(models_dir / name)
        raise SystemExit(f"no cluster model for platform {platform!r} in {models_dir}")
    if len(records) < N_CLUSTERS:
        raise SystemExit(f"platform {platform!r} has {len(records)} segmented images, "
                         f"too few to fit {N_CLUSTERS} image groups")
    model, _labels = cluster.fit_cluster_model(records, n_clusters=N_CLUSTERS, seed=SEED)
    out = work_dir / "cluster_models" / f"{platform}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"platform": platform, **model}, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", type=Path, required=True, help="image table (CSV)")
    ap.add_argument("--parcels", type=Path, required=True, help="parcel file (GeoJSON)")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--cluster-models", type=Path, default=None,
                    help="folder with fitted image groups, <platform>.json (default: fit on the images)")
    args = ap.parse_args()
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    images = load_images(args.images)
    parcels = load_parcels(args.parcels)
    unknown = sorted(set(images["parcel_id"]) - set(parcels))
    if unknown:
        print(f"note: {len(unknown)} parcel ids of the image table are not in the parcel file "
              f"(division {URBAN} is used for them), for example {unknown[0]!r}")

    ratios = segment_images(images, work / "segmentation.jsonl")

    records = []
    for platform, g in images.groupby("platform", sort=True):
        ok = [ratios[i] for i in g["image_id"] if i in ratios and "error" not in ratios[i]]
        model = cluster_model(platform, ok, args.cluster_models, work)
        for r in g.itertuples():
            rec = {"image_id": r.image_id, "path": r.path, "parcel_id": r.parcel_id, "platform": platform}
            seg = ratios.get(r.image_id)
            if seg is None or "error" in seg:
                rec.update({"passed": False, "error": "segmentation_failed"})
            else:
                cid, image_group = cluster.assign_cluster(seg["vegetation_ratio"], seg["sky_ratio"], model)
                division = parcels[r.parcel_id]["division"] if r.parcel_id in parcels else URBAN
                group = operating_group(image_group, division)
                passed, detail = filters.apply_group_filters(seg, group)
                rec.update({**{k: v for k, v in seg.items() if k != "image_id"}, "cluster": cid,
                            "image_group": image_group, "division": division, "group": group,
                            "filters": detail, "passed": bool(passed), "error": ""})
            records.append(rec)

    order = {image_id: i for i, image_id in enumerate(images["image_id"])}
    records.sort(key=lambda r: order[r["image_id"]])
    with (work / "filtering.jsonl").open("w", encoding="utf-8") as fo:
        for rec in records:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    columns = ["image_id", "path", "parcel_id", "platform", "image_group", "division", "group",
               "building_ratio", "vegetation_ratio", "sky_ratio", "passed", "error"]
    df = pd.DataFrame(records).reindex(columns=columns)
    df.to_csv(work / "filtering.csv", index=False, encoding="utf-8-sig")

    summary = {"images": len(df), "passed": int(df["passed"].sum()), "groups": {}}
    print(f"\n{'operating group':<16} {'images':>7} {'passed':>7} {'kept':>7} {'BCR before':>11} {'BCR after':>10}")
    for g in OPERATING_GROUPS:
        rows = [r for r in records if r.get("group") == g]
        kept = [r for r in rows if r["passed"]]
        if not rows:
            continue
        s = {"images": len(rows), "passed": len(kept), "retention": len(kept) / len(rows),
             "bcr_before": building_capture_rate(rows, PRINCIPAL_THRESHOLD),
             "bcr_after": building_capture_rate(kept, PRINCIPAL_THRESHOLD)}
        summary["groups"][g] = s
        print(f"{g:<16} {s['images']:>7} {s['passed']:>7} {s['retention'] * 100:>6.1f}% "
              f"{s['bcr_before'] * 100:>10.1f}% {s['bcr_after'] * 100:>9.1f}%")
    print(f"passed {summary['passed']} / {summary['images']} images")
    (work / "filtering_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
