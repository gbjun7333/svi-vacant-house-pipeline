"""Scores of each house mask against the parcel band (Section 3.3).

    f_align    exp(-(c_k - c_B)^2 / sigma^2), mask centre against band centre
    f_overlap  1-D IoU of the mask's column range and the band
    f_size     exp(-|log(A_k / A_expected)| / TAU_SIZE), mask area against a reference house
    s          f_align * f_overlap * f_size
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import CAMERA_HEIGHT_M, parcel_band, polygon_to_local, target_distance_m
from .masks import load_all_masks

SIGMA_BAND_FRAC = 0.20          # sigma of f_align as a share of the band width
TAU_SIZE = 0.5                  # scale of f_size
STANDARD_HOUSE_M = (8.0, 4.0)   # (width, height) of the reference house


def score_masks(masks, Bt: Tuple[float, float], d_pano_target: float, fx: float,
                sigma_frac: float = SIGMA_BAND_FRAC, tau: float = TAU_SIZE) -> List[Dict]:
    """One dict per non-empty mask with its column extent, area and scores."""
    u_min, u_max = Bt
    c_B = (u_min + u_max) / 2.0
    band_w = max(u_max - u_min, 1.0)
    sigma = sigma_frac * band_w
    # pixel area of the reference house at distance d
    W_h, H_h = STANDARD_HOUSE_M
    d_eff = max(d_pano_target, 1.0)
    A_hat = (W_h * H_h * fx * fx) / (d_eff * d_eff)

    out = []
    for k, M in enumerate(masks):
        ys, xs = np.where(M)
        if len(xs) == 0:
            continue
        cx_k = float(xs.mean())
        l_k, r_k = float(xs.min()), float(xs.max())
        area = int(M.sum())
        f_a = math.exp(-((cx_k - c_B) ** 2) / max(sigma * sigma, 1.0))
        inter = max(0.0, min(r_k, u_max) - max(l_k, u_min))
        union = max(r_k, u_max) - min(l_k, u_min)
        f_o = inter / union if union > 0 else 0.0
        ratio = area / max(A_hat, 1.0)
        f_s = math.exp(-abs(math.log(max(ratio, 1e-9))) / tau)
        out.append({
            "idx": k,
            "cx": cx_k, "l": l_k, "r": r_k, "area": area,
            "f_align": round(f_a, 4),
            "f_overlap": round(f_o, 4),
            "f_size": round(f_s, 4),
            "s": round(f_a * f_o * f_s, 4),
        })
    return out


def score_image(mask_path: Path, parcel_geometry: Optional[Dict], camera_lat: float, camera_lng: float,
                heading: float, fov: float) -> Dict:
    """Parcel band and candidate scores of one image.

    error is "" on success. Otherwise it names the reason and there are no candidates.
    """
    try:
        masks, scores_sam, _bboxes = load_all_masks(mask_path)
        if not masks:
            return {"error": "no_mask"}
        if parcel_geometry is None:
            return {"n_masks": len(masks), "error": "no_parcel_polygon"}
        heading = float(heading) % 360.0
        fov = float(fov)
        pano_xy = (0.0, 0.0)
        H, W = masks[0].shape
        fx = (W / 2.0) / math.tan(math.radians(fov) / 2.0)
        target_poly = polygon_to_local(parcel_geometry, float(camera_lat), float(camera_lng))
        Bt = parcel_band(target_poly, pano_xy, CAMERA_HEIGHT_M, heading, fov, W, H)
        if Bt is None:
            return {"n_masks": len(masks), "error": "target_behind_camera"}
        d = target_distance_m(target_poly, pano_xy)
        scored = score_masks(masks, Bt, d, fx)
        if not scored:
            return {"n_masks": len(masks), "error": "no_valid_mask"}
        cands = []
        for c in scored:
            M = masks[c["idx"]]
            rows_ = np.flatnonzero(M.any(axis=1))
            cands.append({"idx": c["idx"], "area": c["area"], "cx": round(c["cx"], 1), "l": c["l"], "r": c["r"],
                          "bot": int(rows_.max()), "sam": round(scores_sam[c["idx"]], 3), "f_align": c["f_align"],
                          "f_overlap": c["f_overlap"], "f_size": c["f_size"], "s": c["s"]})
        return {"n_masks": len(masks), "W": W, "H": H,
                "bt_umin": round(Bt[0], 1), "bt_umax": round(Bt[1], 1), "d_m": round(d, 1),
                "heading": heading, "fov": fov, "cands": cands, "error": ""}
    except Exception as e:
        return {"error": f"{type(e).__name__}:{e}"[:300]}
