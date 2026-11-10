#!/usr/bin/env python3
"""Step 3: dataset of the classification experiment (Section 3.4).

Builds, from the outputs of steps 1 and 2 and the labels of the parcel file, the
images of the three input conditions, the mask files, the equal image counts and
the folds.

Conditions (every path is an unmarked frame)
    raw        every image of the parcel
    filtered   images that passed group-specific filtering and the SAM 3 step
    aligned    filtered images with a top-1 house mask. ``mask_path`` is that mask as
               an 8-bit PNG of the frame's size, ``mask_shuf_path`` the mask of another
               house in the same frame (drawn at random among the other masks that
               cover at least 1% of the frame; if there is none, the top-1 mask
               mirrored left to right)

Split units: parcels that share a ``pair_id`` in the parcel file form one unit, and
a parcel without ``pair_id`` is a unit of its own. A unit is kept when every one of
its parcels has images in all three conditions. All images of a unit stay in the
same split.

Equal image count: every parcel keeps, in raw and filtered, as many images as it has
in aligned. They are drawn at random with a seed that depends only on the condition
and the parcel (``draw_equal``).

Folds: the units, sorted by id, are shuffled with a fixed seed and dealt into ten
groups. Fold k (0 to 4) tests on groups 2k and 2k+1, validates on group
(2k+2) mod 10 and trains on the other seven, so every unit is tested once.

Outputs in ``--output-dir`` (default ``<work-dir>/dataset``)
    split_units.csv                         one row per unit with its split in each fold
    manifests/<cond>_full.csv, _eq.csv      all images, and the equal-count selection
    masks/, masks_shuf/                     mask PNGs of the aligned images
    folds/fold<k>/<cond>_<train|val|test>.csv
    dataset_summary.json

In the tables ``pair_id`` is the split unit, ``parcel_key`` the parcel id,
``view_id`` the image id, ``crop_path`` the image file and ``source`` the platform.

Usage:
    python scripts/3.4_experimental_setup/03_build_dataset.py \
        --images data/images.csv --parcels data/parcels.geojson --work-dir work
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import svi_pipeline  # noqa: F401
except ImportError:  # running from a checkout without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from svi_pipeline.alignment.masks import load_mask_stack
from svi_pipeline.inputs import load_images, load_parcels

SEED = 2026
CONDS = ("raw", "filtered", "aligned")
N_FOLDS = 5
N_GROUPS = 10


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------
# equal image count
# ---------------------------------------------------------------------------

def draw_equal(df: pd.DataFrame, n_ref: pd.Series, cond: str, seed: int = SEED) -> tuple[pd.DataFrame, list]:
    """Keep, for every parcel_key, ``n_ref[parcel_key]`` rows of ``df``.

    The parcel's rows are sorted by view_id and crop_path, then
    ``random.Random('{seed}:{cond}:{parcel_key}').sample`` picks the rows, which stay
    in sorted order. A parcel with fewer rows than its reference count is kept whole
    and returned in the second value.
    """
    keep, short = [], []
    for pk, g in df.groupby("parcel_key", sort=True):
        g = g.sort_values(["view_id", "crop_path"])
        k = int(n_ref.get(pk, 0))
        if len(g) < k:
            short.append(pk)
            keep.append(g)
            continue
        idx = sorted(random.Random(f"{seed}:{cond}:{pk}").sample(range(len(g)), k))
        keep.append(g.iloc[idx])
    return pd.concat(keep).sort_index(), short


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

def write_mask(stack: np.ndarray, idx: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((stack[idx].astype(np.uint8) * 255)).save(path, optimize=True)


def shuffled_index(stack: np.ndarray, top: int, key: str) -> int | None:
    """Index of another house mask of the frame for the shuffled-mask control."""
    area = stack.reshape(len(stack), -1).mean(axis=1)
    others = [i for i in range(len(stack)) if i != top and area[i] >= 0.01]
    if not others:
        return None
    return random.Random(f"{SEED}:shuf:{key}").choice(others)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", type=Path, required=True, help="image table (CSV)")
    ap.add_argument("--parcels", type=Path, required=True, help="parcel file (GeoJSON) with labels")
    ap.add_argument("--work-dir", type=Path, required=True, help="folder with the outputs of steps 1 and 2")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    work = args.work_dir
    R = args.output_dir or (work / "dataset")

    images = load_images(args.images)
    parcels = load_parcels(args.parcels)
    labelled = {pid: p for pid, p in parcels.items() if p["label"]}
    if not labelled:
        raise SystemExit("the parcel file has no label property (vacant or non_vacant)")
    unit_of = {pid: (p["pair_id"] or pid) for pid, p in labelled.items()}
    members = defaultdict(set)
    for pid, unit in unit_of.items():
        members[unit].add(pid)

    filt = read_table(work / "filtering.csv")
    sam = read_table(work / "sam.csv")
    aln = read_table(work / "alignment.csv")
    filtered_ok = set(filt.loc[filt["passed"] == "True", "image_id"]) & set(sam.loc[sam["filter_result"] == "pass", "image_id"])
    aligned = aln[aln["aligned"] == "True"].set_index("image_id")

    # ---------------- images of every condition, in the order of the image table
    rows = {c: [] for c in CONDS}
    for r in images.itertuples():
        p = labelled.get(r.parcel_id)
        if p is None:
            continue
        b = {"pair_id": unit_of[r.parcel_id], "label": p["label"], "label_id": int(p["label"] == "vacant"),
             "source": r.platform, "parcel_key": r.parcel_id, "crop_path": r.path, "view_id": r.image_id}
        rows["raw"].append(b)
        if r.image_id in filtered_ok:
            rows["filtered"].append(b)
        if r.image_id in aligned.index:
            a = aligned.loc[r.image_id]
            rows["aligned"].append({**b, "npz": a["mask_path"], "top1": int(float(a["top1_idx"])), "bbox": a["bbox_tight"]})

    columns = ["pair_id", "label", "label_id", "source", "parcel_key", "crop_path", "view_id"]
    full = {c: pd.DataFrame(rows[c], columns=columns + (["npz", "top1", "bbox"] if c == "aligned" else []))
            for c in CONDS}
    # a unit needs every one of its parcels in every condition
    ok_units = set(members)
    for c in CONDS:
        present = full[c].groupby("pair_id")["parcel_key"].nunique()
        ok_units &= {u for u, n in present.items() if n == len(members[u])}
    dropped = sorted(set(members) - ok_units)
    if len(ok_units) < N_GROUPS:
        raise SystemExit(f"{len(ok_units)} split units have images in all three conditions, "
                         f"at least {N_GROUPS} are needed for the folds")
    P = pd.DataFrame({"pair_id": sorted(ok_units)})
    P["n_parcels"] = P["pair_id"].map(lambda u: len(members[u]))
    full = {c: d[d["pair_id"].isin(ok_units)].reset_index(drop=True) for c, d in full.items()}

    # ---------------- masks of the aligned images
    A = full["aligned"]
    mpaths, spaths, shuf_kind, n_masks, areas = [], [], [], [], []
    for i, r in enumerate(A.itertuples()):
        stack = load_mask_stack(Path(r.npz))
        W, H = Image.open(r.crop_path).size
        if stack.shape[1:] != (H, W):
            raise SystemExit(f"mask {stack.shape[1:]} != frame {(H, W)}: {r.npz}")
        tag = f"{r.label[0].upper()}{i:06d}"
        mp = R / "masks" / f"{tag}.png"
        write_mask(stack, r.top1, mp)
        j = shuffled_index(stack, r.top1, r.view_id)
        sp = R / "masks_shuf" / f"{tag}.png"
        if j is None:
            sp.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(stack[r.top1][:, ::-1].astype(np.uint8) * 255).save(sp, optimize=True)
            shuf_kind.append("mirrored_top1")
        else:
            write_mask(stack, j, sp)
            shuf_kind.append("other_house")
        mpaths.append(str(mp.resolve())); spaths.append(str(sp.resolve())); n_masks.append(len(stack))
        areas.append(float(stack[r.top1].mean()))
    A = A.assign(mask_path=mpaths, mask_shuf_path=spaths, shuf_kind=shuf_kind, n_masks=n_masks, mask_area=areas)
    full["aligned"] = A

    # ---------------- equal image count and folds
    n_ref = A.groupby("parcel_key").size()
    eq = {"aligned": A}
    short = {}
    for c in ("raw", "filtered"):
        eq[c], short[c] = draw_equal(full[c], n_ref, c, SEED)
    order = list(range(len(P)))
    random.Random(SEED).shuffle(order)
    group = {P.loc[i, "pair_id"]: pos % N_GROUPS for pos, i in enumerate(order)}
    P["group10"] = P["pair_id"].map(group)
    (R / "manifests").mkdir(parents=True, exist_ok=True)
    for c in CONDS:
        full[c].to_csv(R / "manifests" / f"{c}_full.csv", index=False, encoding="utf-8-sig")
        eq[c].insert(0, "example_id", [f"{c}_{i:06d}" for i in range(len(eq[c]))])
        eq[c].to_csv(R / "manifests" / f"{c}_eq.csv", index=False, encoding="utf-8-sig")
    fold_sizes = {}
    for k in range(N_FOLDS):
        test_g, val_g = {2 * k, 2 * k + 1}, {(2 * k + 2) % N_GROUPS}
        sp = P["group10"].map(lambda g: "test" if g in test_g else ("val" if g in val_g else "train"))
        P[f"fold{k}"] = sp.values
        fd = R / "folds" / f"fold{k}"
        fd.mkdir(parents=True, exist_ok=True)
        smap = dict(zip(P["pair_id"], sp))
        for c in CONDS:
            d = eq[c].assign(split=eq[c]["pair_id"].map(smap))
            for s in ("train", "val", "test"):
                d[d["split"] == s].to_csv(fd / f"{c}_{s}.csv", index=False, encoding="utf-8-sig")
        fold_sizes[k] = sp.value_counts().to_dict()
    P.to_csv(R / "split_units.csv", index=False, encoding="utf-8-sig")

    summ = {
        "split_units": len(P), "units_dropped_missing_a_condition": len(dropped),
        "images_full": {c: full[c]["label"].value_counts().to_dict() for c in CONDS},
        "images_eq": {c: eq[c]["label"].value_counts().to_dict() for c in CONDS},
        "parcels_short": {c: len(v) for c, v in short.items()},
        "shuffled_mask_kind": pd.Series(shuf_kind).value_counts().to_dict(),
        "masks_per_aligned_frame_median": float(np.median(n_masks)),
        "fold_sizes_units": fold_sizes,
    }
    (R / "dataset_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
