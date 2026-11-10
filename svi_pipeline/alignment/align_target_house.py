"""Step 2. Target-house alignment (Section 3.3).

SAM 3 house masks for the images that passed filtering. The parcel polygon is then
projected into each image, and the mask that overlaps this parcel band most is taken
as the target house. top1_idx in alignment.csv is its index in masks/<image_id>.npz.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ..inputs import CAMERA_COLUMNS, load_images, load_parcels
from .masks import load_mask_stack, tight_bbox_from_mask
from .scoring import score_image
from .selection import band_gate, pick_top1_overlap


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def run_sam(images: pd.DataFrame, work: Path) -> None:
    """SAM 3 on the images that passed filtering."""
    from . import sam3

    filt = read_table(work / "filtering.csv")
    passed = set(filt.loc[filt["passed"] == "True", "image_id"])
    out_jsonl = work / "sam.jsonl"
    done = {r["image_id"] for r in read_jsonl(out_jsonl)}
    todo = images[images["image_id"].isin(passed) & ~images["image_id"].isin(done)]
    print(f"sam: {len(passed)} filtered images, {len(todo)} to do", flush=True)
    if len(todo):
        runner = sam3.Sam3Runner()
        t0 = time.time()
        with out_jsonl.open("a", encoding="utf-8") as fo:
            for i, r in enumerate(todo.itertuples(), 1):
                mask_path = work / "masks" / f"{r.image_id}.npz"
                base = {"image_id": r.image_id, "path": r.path, "parcel_id": r.parcel_id}
                try:
                    res = sam3.run_one(runner, Path(r.path), mask_path)
                    rec = {**base, "mask_path": str(mask_path), **res, "error": ""}
                except Exception as e:
                    rec = {**base, "filter_result": "fail", "filter_reason": "sam_error",
                           "error": f"{type(e).__name__}:{e}"[:300]}
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 50 == 0:
                    fo.flush()
                    print(f"  [{i}/{len(todo)}] {time.time() - t0:.0f}s ({(time.time() - t0) / i:.2f}s/img)", flush=True)
    df = pd.DataFrame(read_jsonl(out_jsonl))
    if df.empty:
        df = pd.DataFrame(columns=["image_id", "path", "parcel_id", "mask_path", "filter_result", "filter_reason", "error"])
    df.to_csv(out_jsonl.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    if len(df):
        print(df["filter_reason"].value_counts().to_string())


def run_align(images: pd.DataFrame, parcels: dict, work: Path) -> None:
    """Parcel band, mask scores and top-1 mask for the images that passed the SAM 3 step."""
    sam = read_table(work / "sam.csv")
    mask_of = dict(zip(sam.loc[sam["filter_result"] == "pass", "image_id"],
                       sam.loc[sam["filter_result"] == "pass", "mask_path"]))
    out_jsonl = work / "alignment.jsonl"
    done = {r["image_id"] for r in read_jsonl(out_jsonl)}
    todo = images[images["image_id"].isin(mask_of) & ~images["image_id"].isin(done)]
    print(f"align: {len(mask_of)} images with masks, {len(todo)} to do", flush=True)
    with out_jsonl.open("a", encoding="utf-8") as fo:
        for r in todo.itertuples():
            mask_path = mask_of[r.image_id]
            camera = [getattr(r, c) for c in CAMERA_COLUMNS]
            parcel = parcels.get(r.parcel_id)
            if any(pd.isna(v) for v in camera):
                rec = {"error": "missing_camera_values"}
            else:
                rec = score_image(Path(mask_path), parcel["geometry"] if parcel else None, *camera)
            rec = {"image_id": r.image_id, "path": r.path, "parcel_id": r.parcel_id, "mask_path": mask_path, **rec}
            if not rec.get("error"):
                gate = band_gate(rec["cands"])
                if not gate:
                    rec["aligned"] = False
                    rec["align_note"] = "no_band_overlapping_mask"
                else:
                    top = pick_top1_overlap(gate)
                    stack = load_mask_stack(Path(mask_path))
                    bbox = tight_bbox_from_mask(stack[top["idx"]])
                    rec.update({"aligned": True, "top1_idx": top["idx"], "top1_f_overlap": top["f_overlap"],
                                "top1_s": top["s"], "bbox_tight": ",".join(map(str, bbox))})
            else:
                rec["aligned"] = False
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
    recs = read_jsonl(out_jsonl)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "cands"} for r in recs])
    if df.empty:
        df = pd.DataFrame(columns=["image_id", "path", "parcel_id", "mask_path", "aligned", "top1_idx", "bbox_tight", "error"])
    df.to_csv(out_jsonl.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    if len(df):
        print(df["error"].replace("", "ok").value_counts().to_string())
        print("aligned:", int(df["aligned"].astype(bool).sum()), "of", len(df))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", type=Path, required=True, help="image table (CSV)")
    ap.add_argument("--parcels", type=Path, required=True, help="parcel file (GeoJSON)")
    ap.add_argument("--work-dir", type=Path, required=True, help="folder with filtering.csv of step 1")
    ap.add_argument("--part", choices=["sam", "align", "all"], default="all")
    args = ap.parse_args()

    images = load_images(args.images)
    if args.part in ("sam", "all"):
        run_sam(images, args.work_dir)
    if args.part in ("align", "all"):
        run_align(images, load_parcels(args.parcels), args.work_dir)


if __name__ == "__main__":
    main()
