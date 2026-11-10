"""Step 1. Task-suitability filtering (Section 3.2).

SegFormer-B2 class ratios per image, image groups by K-means on the vegetation and
sky ratios (one model per platform), then the filters of each operating group.
--cluster-models svi_pipeline/filtering/models reuses the groups fitted in the study.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ..inputs import load_images, load_parcels
from . import cluster, filters
from .bcr import PRINCIPAL_THRESHOLD, building_capture_rate
from .groups import OPERATING_GROUPS, URBAN, operating_group

N_CLUSTERS = 3
SEED = 42


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def segment_images(images: pd.DataFrame, out_jsonl: Path) -> dict:
    """Class ratios by image_id. Images already in out_jsonl are not segmented again."""
    done = {r["image_id"] for r in read_jsonl(out_jsonl)}
    todo = images[~images["image_id"].isin(done)]
    print(f"segmentation: {len(images)} images, {len(todo)} to do", flush=True)
    if len(todo):
        from ..device import get_device
        from . import segment

        device = get_device()
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
    """K-means model of one platform, loaded from models_dir or fitted on records."""
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
