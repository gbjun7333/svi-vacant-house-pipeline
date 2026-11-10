"""Readers for the image table (CSV) and the parcel file (GeoJSON). README.md lists the columns."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict

import pandas as pd

from .filtering.groups import DIVISIONS, URBAN

CAMERA_COLUMNS = ("camera_lat", "camera_lng", "heading", "fov")
LABELS = {"vacant": "vacant", "1": "vacant", "true": "vacant",
          "non_vacant": "non_vacant", "non-vacant": "non_vacant", "nonvacant": "non_vacant",
          "0": "non_vacant", "false": "non_vacant"}


def _text(value) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


def default_image_id(image_path: str) -> str:
    """Image path without its extension, safe to use as a file name."""
    parts = [p for p in Path(_text(image_path)).with_suffix("").as_posix().split("/") if p not in ("", ".")]
    return "/".join("_" if p == ".." else p.replace(":", "_") for p in parts)


def load_images(images_csv: Path) -> pd.DataFrame:
    """Image table with image_id, path, parcel_id, platform and the camera values.

    Relative image paths are resolved against the folder of the CSV. image_id defaults
    to the image path without its extension. Missing camera values become NaN.
    """
    images_csv = Path(images_csv)
    df = pd.read_csv(images_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = [c for c in ("image_path", "parcel_id") if c not in df.columns]
    if missing:
        raise SystemExit(f"{images_csv}: missing column(s) {', '.join(missing)}")
    base = images_csv.resolve().parent
    out = pd.DataFrame()
    out["image_id"] = df["image_id"].map(_text) if "image_id" in df.columns else df["image_path"].map(default_image_id)
    out["path"] = [str(Path(p) if Path(p).is_absolute() else base / p) for p in df["image_path"]]
    out["parcel_id"] = df["parcel_id"].map(_text)
    out["platform"] = df["platform"].map(_text).replace("", "default") if "platform" in df.columns else "default"
    for c in CAMERA_COLUMNS:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else float("nan")
    if (out["image_id"] == "").any():
        raise SystemExit(f"{images_csv}: empty image_id")
    dup = out.loc[out["image_id"].duplicated(), "image_id"]
    if len(dup):
        raise SystemExit(f"{images_csv}: image_id is not unique, for example {dup.iloc[0]!r}. "
                         "Add an image_id column with unique values.")
    return out


def _division(value) -> str:
    if value in (None, ""):
        return URBAN
    for d in DIVISIONS:
        if _text(value).lower() == d.lower():
            return d
    raise SystemExit(f"parcel file: division must be one of {', '.join(DIVISIONS)}, found {value!r}")


def _label(value) -> str:
    if value in (None, ""):
        return ""
    key = _text(value).lower()
    if key not in LABELS:
        raise SystemExit(f"parcel file: label must be vacant or non_vacant, found {value!r}")
    return LABELS[key]


def load_parcels(parcels_geojson: Path) -> Dict[str, Dict]:
    """{parcel_id: {"geometry", "division", "label", "pair_id"}}, division defaulting to Urban."""
    path = Path(parcels_geojson)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("type") != "FeatureCollection":
        raise SystemExit(f"{path}: expected a GeoJSON FeatureCollection")
    crs = json.dumps(data.get("crs", ""))
    if data.get("crs") and "CRS84" not in crs and "4326" not in crs:
        raise SystemExit(f"{path}: coordinates must be longitude and latitude (EPSG:4326), found crs {crs}")
    parcels: Dict[str, Dict] = {}
    for feature in data["features"]:
        props = feature.get("properties") or {}
        if props.get("parcel_id") in (None, ""):
            raise SystemExit(f"{path}: a feature has no parcel_id property")
        pid = _text(props["parcel_id"])
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in ("Polygon", "MultiPolygon"):
            raise SystemExit(f"{path}: parcel {pid} is not a Polygon or MultiPolygon")
        if pid in parcels:
            raise SystemExit(f"{path}: parcel_id {pid} appears more than once")
        ring = geometry["coordinates"][0] if geometry["type"] == "Polygon" else geometry["coordinates"][0][0]
        lng, lat = ring[0][0], ring[0][1]
        if not (-180 <= lng <= 180 and -90 <= lat <= 90):
            raise SystemExit(f"{path}: parcel {pid} is not in longitude and latitude (EPSG:4326)")
        parcels[pid] = {"geometry": geometry, "division": _division(props.get("division")),
                        "label": _label(props.get("label")),
                        "pair_id": "" if props.get("pair_id") in (None, "") else _text(props["pair_id"])}
    return parcels
